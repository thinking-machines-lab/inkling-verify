"""SWE-bench source suites and the pinned Signal selection with one agent interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

from inkling_vendor_verifier.benchmarks import task_bundles
from inkling_vendor_verifier.cli import evaluate, parser
from inkling_vendor_verifier.code_agent import CodeBenchmark
from inkling_vendor_verifier.types import Example


class SWEBench(CodeBenchmark):
    def __init__(self, subset: str = "signal"):
        if subset not in {"signal", "verified", "pro", "multilingual"}:
            raise ValueError(f"Unknown SWE-bench subset: {subset}")
        self.subset = subset
        self.name = self.cache_name = f"swe-bench-{subset}"

    def prepare(self, data_dir: Path, bundle: Path) -> Path:
        return task_bundles.prepare(data_dir, self.name, bundle)

    def load(self, data_dir: Path) -> list[Example]:
        return task_bundles.load(data_dir, self.name)


def main() -> None:
    root, prepare_parser, run_parser = parser("swe-bench")
    for subparser in (prepare_parser, run_parser):
        subparser.add_argument(
            "--subset",
            choices=("signal", "verified", "pro", "multilingual"),
            default="signal",
        )
    prepare_parser.add_argument("--bundle", type=Path, required=True, help="Pinned task archive")
    args = root.parse_args()
    benchmark = SWEBench(args.subset)
    if args.command == "prepare":
        benchmark.prepare(args.data_dir.resolve(), args.bundle.resolve())
        print(f"Prepared {benchmark.name} in {args.data_dir}")
    else:
        asyncio.run(evaluate(benchmark, args))
