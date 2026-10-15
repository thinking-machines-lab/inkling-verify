"""One resumable, fully accounted execution loop for every benchmark and API."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .types import Benchmark, Example, Generation, Message, Provider, Response, Tool


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_directory(output: Path, example_id: str, sample_index: int) -> Path:
    return output / "samples" / f"{hashlib.sha256(example_id.encode()).hexdigest()}-{sample_index}"


class RecordingProvider:
    def __init__(
        self,
        provider: Provider,
        directory: Path,
        total_usage: dict[str, int],
        calls: list[dict[str, Any]],
        *,
        role: Literal["candidate", "judge"] | None = None,
    ) -> None:
        self.provider = provider
        self.directory = directory
        self.calls = calls
        self.total_usage = total_usage
        self.role = role

    async def generate(
        self,
        messages: Sequence[Message],
        generation: Generation,
        *,
        tools: Sequence[Tool] = (),
        response_schema: dict[str, Any] | None = None,
    ) -> Response:
        started = time.monotonic()
        response = await self.provider.generate(
            messages, generation, tools=tools, response_schema=response_schema
        )
        call = {
            "usage": asdict(response.usage),
            "stop_reason": response.stop_reason,
            "seconds": time.monotonic() - started,
        }
        if self.role is not None:
            call["role"] = self.role
        write_json(
            self.directory / f"call-{len(self.calls):04d}.json",
            {"request": response.request, "response": response.raw, **call},
        )
        self.calls.append(call)
        for key, value in call["usage"].items():
            self.total_usage[key] += value
        return response


async def run(
    benchmark: Benchmark,
    examples: Sequence[Example],
    provider: Provider,
    generation: Generation,
    output_dir: Path,
    *,
    configuration: dict[str, Any],
    concurrency: int,
    samples_per_problem: int = 1,
    resume: bool = False,
    judge: Provider | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("The selected dataset is empty")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if samples_per_problem < 1:
        raise ValueError("samples_per_problem must be positive")
    ids = [example.id for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset contains duplicate example IDs")
    if not isinstance(configuration["dataset"], dict) or not configuration["dataset"]:
        raise ValueError("The run configuration must include the prepared dataset manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another evaluation is using this output directory") from error
        manifest = {
            "schema_version": 4,
            "benchmark": benchmark.name,
            "configuration": configuration,
            "generation": asdict(generation),
            "concurrency": concurrency,
            "samples_per_problem": samples_per_problem,
            "examples": ids,
        }
        manifest_path = output_dir / "run.json"
        if resume:
            if not manifest_path.is_file():
                raise ValueError("Cannot resume without a saved run.json")
            saved = json.loads(manifest_path.read_text())
            if saved["schema_version"] != manifest["schema_version"]:
                raise ValueError("Unsupported saved run schema; start a new output directory")
            if saved != manifest:
                raise ValueError(
                    "Resume requires identical data, selected cases, model, API, and settings"
                )
        else:
            if any(path.name != ".lock" for path in output_dir.iterdir()):
                raise ValueError(
                    "Output directory must be empty; use --resume for an identical run"
                )
            write_json(manifest_path, manifest)

        expected = len(examples) * samples_per_problem
        results: dict[tuple[str, int], dict[str, Any]] = {}
        pending: asyncio.Queue[tuple[Example, int]] = asyncio.Queue()
        for example in examples:
            for sample_index in range(samples_per_problem):
                result_path = sample_directory(output_dir, example.id, sample_index) / "result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    if (
                        result["id"] != example.id
                        or result["sample_index"] != sample_index
                        or (
                            result["score"] is not None
                            and (
                                not math.isfinite(result["score"]) or not 0 <= result["score"] <= 1
                            )
                        )
                    ):
                        raise ValueError(
                            f"Mismatched saved result for {example.id}, sample {sample_index}"
                        )
                    results[example.id, sample_index] = result
                else:
                    pending.put_nowait((example, sample_index))

        started = time.monotonic()
        previous_output_tokens = sum(
            result["usage"]["output_tokens"]
            for result in results.values()
            if result["score"] is not None
        )
        recorded_usage = dict.fromkeys(("input_tokens", "output_tokens", "cached_input_tokens"), 0)
        for path in output_dir.glob("samples/*/attempt-*/call-*.json"):
            for key, value in json.loads(path.read_text())["usage"].items():
                recorded_usage[key] += value

        def summary(status: str) -> dict[str, Any]:
            graded = [result for result in results.values() if result["score"] is not None]
            completed = len(graded)
            score_sum = sum(result["score"] for result in graded)
            seconds = time.monotonic() - started
            usage = {
                key: sum(result["usage"][key] for result in graded)
                for key in ("input_tokens", "output_tokens", "cached_input_tokens")
            }
            value = {
                "status": status,
                "updated_utc": datetime.now(UTC).isoformat(),
                "expected": expected,
                "completed": completed,
                "failed": len(results) - completed,
                "processed": len(results),
                "coverage": completed / expected,
                "accuracy": score_sum / completed if status == "completed" else None,
                "provisional_accuracy": score_sum / completed if completed else None,
                "usage": usage,
                "current_session_seconds": seconds,
                "current_session_completed_output_tokens_per_second": (
                    (usage["output_tokens"] - previous_output_tokens) / seconds if seconds else 0
                ),
                "recorded_usage": dict(recorded_usage),
                "full_dataset": configuration.get("full_dataset", True),
            }
            write_json(output_dir / "summary.json", value)
            return value

        summary("running")

        async def worker() -> None:
            while not pending.empty():
                example, sample_index = pending.get_nowait()
                directory = sample_directory(output_dir, example.id, sample_index)
                directory.mkdir(parents=True, exist_ok=True)
                attempts = sorted(directory.glob("attempt-*"))
                attempt = directory / f"attempt-{len(attempts) + 1:04d}"
                attempt.mkdir()
                calls: list[dict[str, Any]] = []
                recording = RecordingProvider(
                    provider,
                    attempt,
                    recorded_usage,
                    calls,
                    role="candidate" if judge is not None else None,
                )
                case_started = time.monotonic()
                try:
                    judge_recording = (
                        RecordingProvider(judge, attempt, recorded_usage, calls, role="judge")
                        if judge is not None
                        else None
                    )
                    evaluation = await benchmark.evaluate(
                        example, recording, generation, attempt, judge=judge_recording
                    )
                    if evaluation.score is not None and (
                        not math.isfinite(evaluation.score) or not 0 <= evaluation.score <= 1
                    ):
                        raise ValueError(f"Invalid score for {example.id}: {evaluation.score}")
                    result = {
                        "id": example.id,
                        "sample_index": sample_index,
                        **asdict(evaluation),
                        "seconds": time.monotonic() - case_started,
                        "attempt": attempt.name,
                        "calls": calls,
                        "usage": {
                            key: sum(call["usage"][key] for call in calls)
                            for key in ("input_tokens", "output_tokens", "cached_input_tokens")
                        },
                    }
                    write_json(directory / "result.json", result)
                    results[example.id, sample_index] = result
                    progress = summary("running")
                    print(
                        f"{benchmark.name}: {progress['completed']}/{expected} completed, "
                        f"{progress['failed']} unscored",
                        flush=True,
                    )
                except Exception as error:
                    write_json(
                        attempt / "error.json",
                        {
                            "id": example.id,
                            "sample_index": sample_index,
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    )
                    raise
                finally:
                    pending.task_done()

        try:
            async with asyncio.TaskGroup() as tasks:
                for _ in range(min(concurrency, pending.qsize())):
                    tasks.create_task(worker())
        except BaseException:
            summary("incomplete")
            raise
        if len(results) != expected:
            raise RuntimeError("Evaluation ended without complete coverage")
        return summary(
            "completed_with_errors"
            if any(result["score"] is None for result in results.values())
            else "completed"
        )
