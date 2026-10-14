"""Bash agent loop for DeepSWE across serving APIs."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

from inkling_vendor_verifier.sandbox import DockerSandbox
from inkling_vendor_verifier.tool_loop import run_tools
from inkling_vendor_verifier.types import (
    Evaluation,
    Example,
    Generation,
    Provider,
    Text,
    Tool,
)

MAX_TURNS = 4096
MAX_TOOL_CALLS = 4096
AGENT_TIMEOUT_SECONDS = 12 * 3600


def bash_tool(*, allow_network: bool = False) -> Tool:
    return Tool(
        name="bash",
        description=(
            "Executes a bash command in the SandboxInterface. Returns stdout, stderr "
            "(truncated to 16KB each), exit code, and execution time."
            + ("" if allow_network else "\n\nCommands run with this tool have no internet access.")
        ),
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


async def run_bash(sandbox: DockerSandbox, arguments: str) -> tuple[Text, ...]:
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict) or args.keys() - {"command", "workdir", "timeout"}:
            raise ValueError("Expected bash command, optional workdir and timeout")
        if not isinstance(args.get("command"), str) or not args["command"]:
            raise ValueError("command must be a nonempty string")
        if "workdir" in args and not isinstance(args["workdir"], str):
            raise ValueError("workdir must be a string")
        for name in ("command", "workdir"):
            if "\x00" in args.get(name, ""):
                raise ValueError(f"{name} contains a NUL character; command was not executed")
        timeout = args.get("timeout", 300)
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("timeout must be a positive integer in seconds")
    except (ValueError, TypeError) as error:
        return (Text(json.dumps({"error": str(error)})),)
    result = await sandbox.run(args["command"], workdir=args.get("workdir"), timeout=timeout)
    return (Text(json.dumps(asdict(result))),)


class CodeBenchmark:
    def __init__(self, *, allow_network: bool = False):
        self.allow_network = allow_network

    async def evaluate(
        self,
        example: Example,
        provider: Provider,
        generation: Generation,
        work_dir: Path,
        *,
        judge: Provider | None = None,
    ) -> Evaluation:
        async with DockerSandbox(
            Path(example.metadata["task_dir"]), work_dir, allow_network=self.allow_network
        ) as sandbox:
            answer, details = await run_tools(
                provider,
                generation,
                example.messages,
                bash_tool(allow_network=self.allow_network),
                partial(run_bash, sandbox),
                max_turns=MAX_TURNS,
                max_tool_calls=MAX_TOOL_CALLS,
                max_turn_tokens=generation.max_tokens,
                timeout=AGENT_TIMEOUT_SECONDS,
            )
            reward, grade = await sandbox.grade()
            return Evaluation(
                score=reward,
                answer=answer,
                details={
                    "suite": example.metadata["suite"],
                    **details,
                    "verifier_reward": reward,
                    "allow_network": self.allow_network,
                    "solve_allowed_hosts": ["*"] if self.allow_network else [],
                    "verifier_allowed_hosts": ["*"] if self.allow_network else [],
                    **grade,
                },
            )
