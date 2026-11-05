"""Datacurve DeepSWE with committed patches and separate verification."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

from inkling_verify.benchmarks import deepswe_data
from inkling_verify.cli import evaluate, parser, positive
from inkling_verify.deepswe_images import IMAGE_MANIFEST_SHA256
from inkling_verify.mini_swe import AGENT, run_agent
from inkling_verify.sandbox import DeepSWESandbox
from inkling_verify.types import (
    Evaluation,
    Example,
    Generation,
    Provider,
)

AGENT_TIMEOUT_SECONDS = 4 * 3600


class DeepSWE:
    name = cache_name = "deepswe"

    def __init__(
        self, *, allow_network: bool = False, timeout_seconds: float = AGENT_TIMEOUT_SECONDS
    ):
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        self.allow_network = allow_network
        self.timeout_seconds = timeout_seconds

    def prepare(self, data_dir: Path) -> Path:
        return deepswe_data.prepare(data_dir)

    def load(self, data_dir: Path) -> list[Example]:
        return deepswe_data.load(data_dir)

    async def evaluate(
        self,
        example: Example,
        provider: Provider,
        generation: Generation,
        work_dir: Path,
        *,
        judge: Provider | None = None,
    ) -> Evaluation:
        async with DeepSWESandbox(
            Path(example.metadata["task_dir"]), work_dir, allow_network=self.allow_network
        ) as sandbox:
            if len(example.messages) != 1 or example.messages[0].role != "user":
                raise ValueError("DeepSWE requires a single task instruction")
            answer, details = await run_agent(
                provider,
                generation,
                example.messages[0].text,
                sandbox,
                work_dir,
                self.timeout_seconds,
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


def main() -> None:
    root, _, run = parser("deepswe")
    run.set_defaults(max_output_tokens=1_000_000, concurrency=4)
    run.add_argument(
        "--timeout-seconds",
        type=positive,
        default=AGENT_TIMEOUT_SECONDS,
        help="solver time limit in seconds; excludes setup and grading (default: %(default)s)",
    )
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
        asyncio.run(
            evaluate(
                DeepSWE(allow_network=args.with_network, timeout_seconds=args.timeout_seconds),
                args,
                benchmark_configuration={
                    "agent": AGENT,
                    "agent_timeout_seconds": args.timeout_seconds,
                    "image_manifest_sha256": IMAGE_MANIFEST_SHA256,
                    "allow_network": args.with_network,
                },
            )
        )


if __name__ == "__main__":
    main()
