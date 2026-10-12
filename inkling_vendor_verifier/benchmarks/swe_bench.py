"""SWE-bench source suites and the pinned Signal selection with one agent interface."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tarfile
import tempfile
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path

from inkling_vendor_verifier.benchmarks.swe_sources import (
    _digest as _manifest_digest,
)
from inkling_vendor_verifier.benchmarks.swe_sources import (
    _file_hashes,
    _validate_tasks,
    load_source,
    prepare_source,
)
from inkling_vendor_verifier.cli import evaluate, parser
from inkling_vendor_verifier.sandbox import DockerSandbox, phase_hosts
from inkling_vendor_verifier.types import (
    Evaluation,
    Example,
    Generation,
    Message,
    Provider,
    Text,
    Tool,
)

MAX_TURNS = 4096
MAX_TOOL_CALLS = 4096


def release() -> dict:
    return json.loads(files(__package__).joinpath("swe_signal_selection.json").read_text())


def selection() -> list[dict[str, str]]:
    return [
        {"id": task_id, "suite": suite}
        for suite, task_ids in release()["tasks"].items()
        for task_id in task_ids
    ]


def bash_tool(hosts: list[str]) -> Tool:
    description = (
        "Executes a bash command in the SandboxInterface. Returns stdout, stderr "
        "(truncated to 16KB each), exit code, and execution time."
    )
    if not hosts:
        description += "\n\nCommands run with this tool have no internet access."
    elif hosts != ["*"]:
        description += (
            "\n\nCommands run with this tool can only reach a restricted set of allowlisted "
            "internet hosts."
        )
    return Tool(
        name="bash",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command (default: container WORKDIR)",
                },
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 300)"},
            },
            "required": ["command"],
        },
    )


def _validate(directory: Path) -> None:
    _validate_tasks(directory, [row["id"] for row in selection()])


def prepare(data_dir: Path, bundle: Path) -> Path:
    with bundle.open("rb") as source:
        archive_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if archive_hash != release()["bundle_sha256"]:
        raise ValueError("SWE archive differs from the pinned release bundle")
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "swe-bench-signal"
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".swe-prepare-", dir=data_dir) as temporary:
        staged = Path(temporary) / "swe-bench-signal"
        staged.mkdir()
        with tarfile.open(bundle) as archive:
            if any(member.issym() or member.islnk() for member in archive):
                raise ValueError("Archive links are forbidden")
            archive.extractall(staged, filter="data")
        _validate(staged)
        hashes = _file_hashes(staged)
        if _manifest_digest(hashes) != release()["files_sha256"]:
            raise ValueError("SWE bundle assets differ from the pinned release manifest")
        if (manifest := staged / "manifest.json").exists():
            recorded = json.loads(manifest.read_text())
            if recorded["files"] != hashes:
                raise ValueError("Task bundle file checksums do not match its manifest")
        manifest.write_text(
            json.dumps({"benchmark": "swe-bench-signal", "files": hashes}, indent=2)
        )
        staged.rename(destination)
    return destination


class SWEBench:
    def __init__(self, subset: str = "signal"):
        if subset not in {"signal", "verified", "pro", "multilingual"}:
            raise ValueError(f"Unknown SWE-bench subset: {subset}")
        self.subset = subset
        self.name = f"swe-bench-{subset}"
        self.cache_name = self.name

    def load(self, data_dir: Path) -> list[Example]:
        if self.subset != "signal":
            return load_source(data_dir, self.subset)
        directory = data_dir / self.name
        _validate(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["files"] != _file_hashes(directory):
            raise ValueError("Installed SWE task files have changed; prepare the bundle again")
        bundle_digest = _manifest_digest(manifest["files"])
        if bundle_digest != release()["files_sha256"]:
            raise ValueError("Installed SWE manifest differs from the pinned release")
        return [
            Example(
                id=row["id"],
                messages=(
                    Message(
                        "user",
                        (Text((directory / "tasks" / row["id"] / "instruction.md").read_text()),),
                    ),
                ),
                target="",
                metadata={
                    "task_dir": str(directory / "tasks" / row["id"]),
                    "suite": row["suite"],
                    "bundle_sha256": bundle_digest,
                },
            )
            for row in selection()
        ]

    async def evaluate(
        self,
        example: Example,
        provider: Provider,
        generation: Generation,
        work_dir: Path,
        *,
        judge: Provider | None = None,
    ) -> Evaluation:
        messages = list(example.messages)
        remaining = generation.max_tokens
        calls = 0
        turns = 0
        stop = "max_turns"
        answer = ""
        async with DockerSandbox(Path(example.metadata["task_dir"]), work_dir) as sandbox:
            tool = bash_tool(phase_hosts(sandbox.config, "agent"))
            deadline = asyncio.timeout(
                float(sandbox.config.get("agent", {}).get("timeout_sec", 28800))
            )
            try:
                async with deadline:
                    for turns in range(1, MAX_TURNS + 1):
                        response = await provider.generate(
                            messages, replace(generation, max_tokens=remaining), tools=(tool,)
                        )
                        if response.usage.output_tokens <= 0:
                            raise RuntimeError(
                                "SWE generation requires positive reported output token usage"
                            )
                        remaining -= response.usage.output_tokens
                        messages.append(response.message)
                        answer = response.text
                        if response.stop_reason == "length" or remaining <= 0:
                            stop = "max_tokens"
                            break
                        if not response.message.tool_calls:
                            stop = response.stop_reason
                            break
                        if calls + len(response.message.tool_calls) > MAX_TOOL_CALLS:
                            stop = "max_tool_calls"
                            break
                        for call in response.message.tool_calls:
                            calls += 1
                            try:
                                if call.name != "bash":
                                    raise ValueError(f"Unknown tool: {call.name}")
                                args = json.loads(call.arguments)
                                if not isinstance(args, dict) or args.keys() - {
                                    "command",
                                    "workdir",
                                    "timeout",
                                }:
                                    raise ValueError(
                                        "Expected bash command, optional workdir and timeout"
                                    )
                                if not isinstance(args.get("command"), str) or not args["command"]:
                                    raise ValueError("command must be a nonempty string")
                                if "workdir" in args and not isinstance(args["workdir"], str):
                                    raise ValueError("workdir must be a string")
                                timeout = args.get("timeout", 300)
                                if type(timeout) is not int or timeout <= 0:
                                    raise ValueError(
                                        "timeout must be a positive integer in seconds"
                                    )
                            except (ValueError, TypeError) as error:
                                result_text = json.dumps({"error": str(error)})
                            else:
                                result = await sandbox.run(
                                    args["command"], workdir=args.get("workdir"), timeout=timeout
                                )
                                result_text = json.dumps(asdict(result))
                            messages.append(
                                Message("tool", (Text(result_text),), tool_call_id=call.id)
                            )
                        if calls == MAX_TOOL_CALLS:
                            stop = "max_tool_calls"
                            break
            except TimeoutError:
                if not deadline.expired():
                    raise
                stop = "timeout"
            reward, grade = await sandbox.grade()
            return Evaluation(
                score=0.0
                if stop in {"max_tokens", "max_turns", "max_tool_calls", "timeout"}
                else reward,
                answer=answer,
                details={
                    "suite": example.metadata["suite"],
                    "stop_reason": stop,
                    "turns": turns,
                    "tool_calls": calls,
                    "generated_tokens": generation.max_tokens - remaining,
                    "token_budget_kind": "cumulative_generated_tokens",
                    "verifier_reward": reward,
                    "solve_allowed_hosts": phase_hosts(sandbox.config, "agent"),
                    "verifier_allowed_hosts": phase_hosts(sandbox.config, "verifier"),
                    **grade,
                },
            )


def main() -> None:
    root, prepare_parser, run_parser = parser("swe-bench")
    for subparser in (prepare_parser, run_parser):
        subparser.add_argument(
            "--subset", choices=("signal", "verified", "pro", "multilingual"), default="signal"
        )
    prepare_parser.add_argument(
        "--bundle", type=Path, help="Pinned SWE Signal task archive (.tar.gz); required for signal"
    )
    args = root.parse_args()
    benchmark = SWEBench(args.subset)
    if args.command == "prepare":
        if args.subset == "signal":
            if args.bundle is None:
                root.error("--bundle is required when preparing --subset signal")
            prepare(args.data_dir.resolve(), args.bundle.resolve())
        else:
            if args.bundle is not None:
                root.error("--bundle is only supported with --subset signal")
            prepare_source(args.data_dir.resolve(), args.subset)
        print(f"Prepared {benchmark.name} in {args.data_dir}")
    else:
        asyncio.run(evaluate(benchmark, args))
