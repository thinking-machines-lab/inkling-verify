"""SWE-bench source suites and the pinned Signal selection with one agent interface."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tarfile
import tempfile
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
from inkling_vendor_verifier.code_agent import CodeBenchmark
from inkling_vendor_verifier.types import Example, Message, Text


def release() -> dict:
    return json.loads(files(__package__).joinpath("swe_signal_selection.json").read_text())


def selection() -> list[dict[str, str]]:
    return [
        {"id": task_id, "suite": suite}
        for suite, task_ids in release()["tasks"].items()
        for task_id in task_ids
    ]


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


class SWEBench(CodeBenchmark):
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


def main() -> None:
    root, prepare_parser, run_parser = parser("swe-bench")
    for subparser in (prepare_parser, run_parser):
        subparser.add_argument(
            "--subset",
            choices=("signal", "verified", "pro", "multilingual"),
            default="signal",
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
