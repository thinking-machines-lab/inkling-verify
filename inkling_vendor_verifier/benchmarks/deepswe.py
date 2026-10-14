"""Datacurve DeepSWE with committed patches and separate verification."""

from __future__ import annotations

import asyncio
from pathlib import Path

from inkling_vendor_verifier.benchmarks import task_bundles
from inkling_vendor_verifier.cli import evaluate, parser
from inkling_vendor_verifier.code_agent import CodeBenchmark
from inkling_vendor_verifier.types import Example


class DeepSWE(CodeBenchmark):
    name = cache_name = "deepswe"

    def prepare(self, data_dir: Path) -> Path:
        return task_bundles.prepare(data_dir, self.name)

    def load(self, data_dir: Path) -> list[Example]:
        return task_bundles.load(data_dir, self.name)


def main() -> None:
    root, _, run = parser("deepswe")
    run.add_argument(
        "--with-network",
        action="store_true",
        help="enable network access for both solver and grader (default: disabled)",
    )
    args = root.parse_args()
    if args.command == "prepare":
        benchmark = DeepSWE()
        benchmark.prepare(args.data_dir.resolve())
        print(f"Prepared {benchmark.name} in {args.data_dir}")
    else:
        asyncio.run(evaluate(DeepSWE(allow_network=args.with_network), args))


if __name__ == "__main__":
    main()
