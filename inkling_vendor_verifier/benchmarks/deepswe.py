"""Datacurve DeepSWE with committed patches and offline, separate verification."""

from __future__ import annotations

import asyncio
from pathlib import Path

from inkling_vendor_verifier.benchmarks.swe_sources import load_source, prepare_source
from inkling_vendor_verifier.cli import evaluate, parser
from inkling_vendor_verifier.code_agent import CodeBenchmark
from inkling_vendor_verifier.types import Example


class DeepSWE(CodeBenchmark):
    name = cache_name = "deepswe"
    zero_reward_on_limit = False

    def prepare(self, data_dir: Path) -> Path:
        return prepare_source(data_dir, self.name)

    def load(self, data_dir: Path) -> list[Example]:
        return load_source(data_dir, self.name)


def main() -> None:
    root, _, _ = parser("deepswe")
    args = root.parse_args()
    benchmark = DeepSWE()
    if args.command == "prepare":
        benchmark.prepare(args.data_dir.resolve())
        print(f"Prepared {benchmark.name} in {args.data_dir}")
    else:
        asyncio.run(evaluate(benchmark, args))
