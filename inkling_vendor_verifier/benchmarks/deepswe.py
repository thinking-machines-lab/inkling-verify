"""Datacurve DeepSWE with committed patches and offline, separate verification."""

from __future__ import annotations

import asyncio
from pathlib import Path

from inkling_vendor_verifier.benchmarks import task_bundles
from inkling_vendor_verifier.cli import evaluate, parser
from inkling_vendor_verifier.code_agent import CodeBenchmark
from inkling_vendor_verifier.types import Example


class DeepSWE(CodeBenchmark):
    name = cache_name = "deepswe"
    zero_reward_on_limit = False

    def prepare(self, data_dir: Path, bundle: Path) -> Path:
        return task_bundles.prepare(data_dir, self.name, bundle)

    def load(self, data_dir: Path) -> list[Example]:
        return task_bundles.load(data_dir, self.name)


def main() -> None:
    root, prepare_parser, _ = parser("deepswe")
    prepare_parser.add_argument("--bundle", type=Path, required=True, help="Pinned task archive")
    args = root.parse_args()
    benchmark = DeepSWE()
    if args.command == "prepare":
        benchmark.prepare(args.data_dir.resolve(), args.bundle.resolve())
        print(f"Prepared {benchmark.name} in {args.data_dir}")
    else:
        asyncio.run(evaluate(benchmark, args))
