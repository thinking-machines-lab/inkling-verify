import asyncio
import hashlib
import json
from dataclasses import replace
from functools import partial

import pytest

from inkling_vendor_verifier.runner import run
from inkling_vendor_verifier.types import (
    Evaluation,
    Example,
    Generation,
    Image,
    Message,
    Response,
    Text,
    Usage,
)

GENERATION = Generation(100)
CONFIGURATION = {
    "api": "openai-completions",
    "model": "model",
    "base_url": "https://candidate.example/v1",
    "full_dataset": True,
    "dataset": {"repository": "test/data", "revision": "pinned", "examples_sha256": "a" * 64},
}


@pytest.fixture
def execute(tmp_path):
    return partial(
        run, generation=GENERATION, output_dir=tmp_path, configuration=CONFIGURATION, concurrency=1
    )


def example(identifier="one"):
    return Example(identifier, (Message("user", (Text("Question"),)),), "4")


def response(text="4"):
    return Response(
        Message("assistant", (Text(text),)),
        Usage(10, 7, 3),
        "stop",
        {"answer": text},
        {"model": "model"},
    )


class OnceBenchmark:
    name = "test"

    async def evaluate(self, example, provider, generation, work_dir, *, judge=None):
        value = await provider.generate(example.messages, generation)
        return Evaluation(float(value.text == example.target), value.text)


class JudgedBenchmark:
    name = "judged"

    async def evaluate(self, example, provider, generation, work_dir, *, judge=None):
        candidate = await provider.generate(example.messages, generation)
        verdict = await (judge if judge is not None else provider).generate(
            (Message("user", (Text(candidate.text),)),),
            Generation(25),
            response_schema={"type": "object"},
        )
        return Evaluation(float(verdict.text == "correct"), candidate.text)


class ConstantProvider:
    calls = 0

    async def generate(self, messages, generation, **kwargs):
        self.calls += 1
        return response()


async def test_concurrency_slot_includes_generation_and_judging(execute):
    both_judging = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class BlockingProvider:
        async def generate(self, messages, generation, **kwargs):
            phase = "judge" if kwargs["response_schema"] else "candidate"
            calls.append(phase)
            if phase == "judge" and not release.is_set():
                if calls.count("judge") == 2:
                    both_judging.set()
                await asyncio.wait_for(release.wait(), timeout=3)
            return response("correct" if phase == "judge" else "4")

    task = asyncio.create_task(
        execute(
            JudgedBenchmark(),
            [example(str(i)) for i in range(5)],
            BlockingProvider(),
            concurrency=2,
        )
    )
    try:
        await asyncio.wait_for(both_judging.wait(), timeout=3)
        assert calls == ["candidate", "judge", "candidate", "judge"]
    finally:
        release.set()
    summary = await asyncio.wait_for(task, timeout=3)
    assert summary["completed"] == 5
    assert summary["recorded_usage"]["output_tokens"] == 70


@pytest.mark.parametrize("external", [False, True])
async def test_failed_judging_preserves_candidate_artifact_and_usage_on_resume(
    execute, tmp_path, external
):
    class JudgeProvider:
        def __init__(self, *, fail):
            self.fail = fail

        async def generate(self, messages, generation, **kwargs):
            if kwargs["response_schema"]:
                if self.fail:
                    raise TimeoutError("Judge unavailable")
                return replace(response("correct"), usage=Usage(20, 5, 2))
            return response()

    failing = JudgeProvider(fail=True)
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await execute(
            JudgedBenchmark(),
            [example()],
            ConstantProvider() if external else failing,
            judge=failing if external else None,
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["completed"] == 0
    assert summary["accuracy"] is None
    assert summary["recorded_usage"] == {
        "input_tokens": 10,
        "output_tokens": 7,
        "cached_input_tokens": 3,
    }
    assert summary["usage"] == dict.fromkeys(summary["recorded_usage"], 0)
    candidate_path = next(tmp_path.glob("samples/*/attempt-0001/call-0000.json"))
    original_artifact = candidate_path.read_bytes()
    candidate = json.loads(original_artifact)
    assert candidate["response"] == {"answer": "4"}
    assert (candidate["role"] == "candidate") if external else "role" not in candidate
    assert not list(tmp_path.glob("samples/*/attempt-0001/call-0001.json"))
    assert not list(tmp_path.glob("samples/*/result.json"))
    error = json.loads(next(tmp_path.glob("samples/*/attempt-0001/error.json")).read_text())
    assert error["type"] == "TimeoutError"

    working = JudgeProvider(fail=False)
    summary = await execute(
        JudgedBenchmark(),
        [example()],
        ConstantProvider() if external else working,
        judge=working if external else None,
        resume=True,
    )
    assert summary["status"] == "completed"
    assert summary["usage"] == {
        "input_tokens": 30,
        "output_tokens": 12,
        "cached_input_tokens": 5,
    }
    assert summary["recorded_usage"] == {
        "input_tokens": 40,
        "output_tokens": 19,
        "cached_input_tokens": 8,
    }
    assert candidate_path.read_bytes() == original_artifact
    assert len(list(tmp_path.glob("samples/*/attempt-0002/call-*.json"))) == 2


async def test_completed_resume_makes_no_calls_and_has_zero_session_throughput(execute, tmp_path):
    provider = ConstantProvider()
    args = (OnceBenchmark(), [example()], provider)
    first = await execute(*args)
    second = await execute(*args, resume=True)
    assert provider.calls == 1
    assert second["recorded_usage"] == first["recorded_usage"]
    assert second["current_session_completed_output_tokens_per_second"] == 0
    assert second["coverage"] == 1
    assert json.loads((tmp_path / "run.json").read_text()) == {
        "schema_version": 2,
        "benchmark": "test",
        "configuration": CONFIGURATION,
        "generation": {
            "max_tokens": 100,
            "temperature": 1.0,
            "top_p": 1.0,
            "reasoning_effort": "high",
        },
        "concurrency": 1,
        "examples": ["one"],
    }


@pytest.mark.parametrize(
    "change",
    [
        "api",
        "model",
        "base_url",
        "dataset",
        "selection",
        "order",
        "generation",
        "judge-mode",
        "judge-api",
        "judge-model",
        "judge-base_url",
        "judge-generation",
        "allow_network",
    ],
)
async def test_resume_rejects_changed_workload(execute, change):
    provider = ConstantProvider()
    configuration = CONFIGURATION | {
        "judge": {
            "mode": "external",
            "api": "openai-responses",
            "model": "judge-model",
            "base_url": "https://judge.example/v1",
            "generation": {"max_tokens": 25},
        },
    }
    examples = [example("one"), example("two")]
    generation = GENERATION
    await execute(OnceBenchmark(), examples, provider, configuration=configuration)
    if change in ("api", "model", "base_url"):
        configuration = configuration | {change: "different"}
    elif change == "dataset":
        configuration = configuration | {
            "dataset": configuration["dataset"] | {"examples_sha256": "b" * 64}
        }
    elif change in ("selection", "order"):
        examples = examples[:1] if change == "selection" else list(reversed(examples))
    elif change == "generation":
        generation = replace(GENERATION, max_tokens=99)
    elif change == "allow_network":
        configuration = configuration | {"allow_network": True}
    else:
        field = change.removeprefix("judge-")
        value = (
            {"max_tokens": 26}
            if field == "generation"
            else ("self" if field == "mode" else "different")
        )
        configuration = configuration | {"judge": configuration["judge"] | {field: value}}
    with pytest.raises(ValueError, match="identical"):
        await execute(
            OnceBenchmark(),
            examples,
            provider,
            configuration=configuration,
            generation=generation,
            resume=True,
        )
    assert provider.calls == 2


@pytest.mark.parametrize("dataset", [None, {}])
async def test_run_requires_prepared_dataset_manifest(execute, dataset):
    provider = ConstantProvider()
    with pytest.raises(ValueError, match="prepared dataset manifest"):
        await execute(
            OnceBenchmark(),
            [example()],
            provider,
            configuration=CONFIGURATION | {"dataset": dataset},
        )
    assert provider.calls == 0


@pytest.mark.parametrize("version", [1, 3])
async def test_resume_explicitly_rejects_incompatible_manifest_schema(execute, tmp_path, version):
    (tmp_path / "run.json").write_text(json.dumps({"schema_version": version}))
    provider = ConstantProvider()
    with pytest.raises(ValueError, match="Unsupported saved run schema"):
        await execute(OnceBenchmark(), [example()], provider, resume=True)
    assert provider.calls == 0


async def test_resume_uses_dataset_manifest_instead_of_local_asset_paths(execute):
    original = replace(
        example(),
        messages=(Message("user", (Image(b"pixels", "image/png"),)),),
        metadata={"images": {"image_1": {"image_path": "/old/image.png"}}},
    )
    relocated = replace(
        original, metadata={"images": {"image_1": {"image_path": "/new/image.png"}}}
    )
    provider = ConstantProvider()
    await execute(OnceBenchmark(), [original], provider)
    await execute(OnceBenchmark(), [relocated], provider, resume=True)
    assert provider.calls == 1


async def test_simultaneous_runs_cannot_share_output_directory(execute):
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider:
        async def generate(self, messages, generation, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=3)
            return response()

    first = asyncio.create_task(execute(OnceBenchmark(), [example()], BlockingProvider()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        with pytest.raises(ValueError, match="Another evaluation"):
            await execute(OnceBenchmark(), [example()], ConstantProvider(), resume=True)
    finally:
        release.set()
    assert (await asyncio.wait_for(first, timeout=3))["status"] == "completed"


async def test_partial_dataset_completion_remains_labeled_as_smoke(execute):
    result = await execute(
        OnceBenchmark(),
        [example()],
        ConstantProvider(),
        configuration=CONFIGURATION | {"full_dataset": False},
    )
    assert result["coverage"] == 1.0
    assert result["full_dataset"] is False


@pytest.mark.parametrize("field, value", [("id", "other"), ("score", -1), ("score", float("nan"))])
async def test_invalid_saved_result_is_rejected(execute, tmp_path, field, value):
    await execute(OnceBenchmark(), [example()], ConstantProvider())
    path = tmp_path / "samples" / hashlib.sha256(b"one").hexdigest() / "result.json"
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Mismatched saved result"):
        await execute(OnceBenchmark(), [example()], ConstantProvider(), resume=True)


async def test_external_judge_records_distinct_calls_and_combined_usage(execute, tmp_path):
    observed = []

    class Candidate:
        async def generate(self, messages, generation, **kwargs):
            observed.append("candidate")
            assert messages == example().messages
            assert generation == GENERATION
            assert kwargs["response_schema"] is None
            return response()

    class Judge:
        async def generate(self, messages, generation, **kwargs):
            observed.append("judge")
            assert messages == (Message("user", (Text("4"),)),)
            assert generation == Generation(25)
            assert kwargs["response_schema"] == {"type": "object"}
            return replace(
                response("correct"), usage=Usage(20, 5, 2), request={"model": "judge-model"}
            )

    summary = await execute(JudgedBenchmark(), [example()], Candidate(), judge=Judge())
    assert observed == ["candidate", "judge"]
    expected_usage = {"input_tokens": 30, "output_tokens": 12, "cached_input_tokens": 5}
    assert summary["usage"] == summary["recorded_usage"] == expected_usage
    assert summary["accuracy"] == 1.0
    paths = sorted(tmp_path.glob("samples/*/attempt-0001/call-*.json"))
    assert [path.name for path in paths] == ["call-0000.json", "call-0001.json"]
    candidate, judge = [json.loads(path.read_text()) for path in paths]
    assert candidate["role"] == "candidate"
    assert candidate["request"] == {"model": "model"}
    assert candidate["response"] == {"answer": "4"}
    assert judge["role"] == "judge"
    assert judge["request"] == {"model": "judge-model"}
    assert judge["response"] == {"answer": "correct"}
    result = json.loads(next(tmp_path.glob("samples/*/result.json")).read_text())
    assert [call["role"] for call in result["calls"]] == ["candidate", "judge"]
    assert result["usage"] == expected_usage


async def test_default_call_records_keep_original_unlabeled_shape(execute, tmp_path):
    await execute(OnceBenchmark(), [example()], ConstantProvider())
    call = json.loads(next(tmp_path.glob("samples/*/attempt-0001/call-0000.json")).read_text())
    assert set(call) == {"request", "response", "usage", "stop_reason", "seconds"}
    result = json.loads(next(tmp_path.glob("samples/*/result.json")).read_text())
    assert set(result["calls"][0]) == {"usage", "stop_reason", "seconds"}
