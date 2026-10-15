import hashlib
import json

import pytest

from inkling_verify.benchmarks import hle
from inkling_verify.benchmarks.hle import (
    HLE,
    JUDGE_PROMPT,
    SCHEMA_SYSTEM_INSTRUCTION,
    _has_image,
    _prepare_row,
    parse_verdict,
)
from inkling_verify.runner import run, sample_directory
from inkling_verify.types import (
    Example,
    Generation,
    Message,
    Response,
    Text,
    Usage,
)

VERDICT = {
    "extracted_final_answer": "4",
    "reasoning": "The answers agree.",
    "correct": "yes",
    "confidence": 100,
    "strict": True,
}


def test_official_prompt_and_schema_instruction_bytes():
    assert hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest() == (
        "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835"
    )
    assert hashlib.sha256(SCHEMA_SYSTEM_INSTRUCTION.encode()).hexdigest() == (
        "d22a9a11137aa949eb9ccd089f089c7712a8cbd2d7b476cce4e90402797db539"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "```json\n" + json.dumps(VERDICT) + "\n```",
        json.dumps({k: v for k, v in VERDICT.items() if k != "strict"}),
        json.dumps(VERDICT | {"strict": False}),
        json.dumps(VERDICT | {"confidence": True}),
        json.dumps(VERDICT | {"confidence": "100"}),
        json.dumps(VERDICT | {"confidence": 1.5}),
        json.dumps(VERDICT | {"correct": "YES"}),
        json.dumps(VERDICT | {"extra": "field"}),
    ],
)
def test_invalid_judge_output_is_an_error(text):
    with pytest.raises(ValueError, match="invalid ExtractedAnswer"):
        parse_verdict(text)


def test_hle_data_preserves_question_bytes_and_filters_only_images():
    row = {
        "id": "synthetic-1",
        "question": "  What is two plus two?\n",
        "answer": "4",
        "answer_type": "exactMatch",
        "category": "math",
    }
    assert _prepare_row(row)["question"] == row["question"]
    assert not _has_image({"image": ""})
    assert not _has_image({"image": None})
    assert _has_image({"image": "data:image/png;base64,abc"})
    with pytest.raises(ValueError, match="image"):
        _has_image({"image": []})
    with pytest.raises(ValueError, match="answer type"):
        _prepare_row(row | {"answer_type": "guess"})


def test_prepare_load_filters_images_preserves_prompts_and_reuses_pinned_cache(
    tmp_path, monkeypatch
):
    import datasets

    rows = [
        {
            "id": "exact",
            "question": "  What is 2 × 2?\n",
            "answer": "4",
            "answer_type": "exactMatch",
            "category": "math",
            "image": None,
        },
        {
            "id": "choice",
            "question": "\tChoose four.\nA. 4\nB. 5\n",
            "answer": "A",
            "answer_type": "multipleChoice",
            "category": "math",
            "image": "",
        },
    ]
    calls = []

    def load_dataset(repository, *, split, revision):
        calls.append((repository, split, revision))
        return [
            rows[0],
            rows[0] | {"id": "visual", "image": "data:image/png;base64,abc"},
            rows[1],
        ]

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(hle, "EXPECTED_COUNT", 2)
    benchmark = HLE()
    benchmark.prepare(tmp_path)
    examples = benchmark.load(tmp_path)
    assert len(examples) == 2
    for example, row in zip(examples, rows, strict=True):
        assert example.messages == (
            Message("system", (Text(hle.SYSTEM_PROMPTS[row["answer_type"]]),)),
            Message("user", (Text(row["question"]),)),
        )
        assert example.target == row["answer"]
        assert example.metadata == {
            "question": row["question"],
            "source_id": row["id"],
            "category": row["category"],
            "answer_type": row["answer_type"],
        }
    assert (tmp_path / "hle" / "manifest.json").is_file()
    benchmark.prepare(tmp_path)
    assert benchmark.load(tmp_path) == examples
    assert calls == [(hle.REPOSITORY, "test", hle.REVISION)]


@pytest.mark.parametrize(
    "failure", ["length", "context_limit", "refusal", "tool_calls", "invalid_json"]
)
@pytest.mark.parametrize("all_failed", [False, True])
async def test_judge_failures_are_unscored_terminal_and_do_not_abort_other_cases(
    tmp_path, failure, all_failed
):
    class Provider:
        calls = 0

        async def generate(self, messages, generation, **kwargs):
            self.calls += 1
            judging = kwargs["response_schema"] is not None
            failing = judging and (self.calls == 2 or all_failed)
            text = "unfinished" if failing else json.dumps(VERDICT) if judging else "candidate"
            stop = failure if failing and failure != "invalid_json" else "stop"
            return Response(
                Message("assistant", (Text(text),)), Usage(1, 2), stop, {"text": text}, {}
            )

    provider = Provider()
    examples = [Example(str(i), (), "4", {"question": "Question"}) for i in range(2)]
    options = dict(configuration={"dataset": {"fixture": "hle"}}, concurrency=1)
    summary = await run(HLE(), examples, provider, Generation(100), tmp_path, **options)
    expected_failed = 2 if all_failed else 1
    assert summary["status"] == "completed_with_errors"
    assert summary["completed"] == 2 - expected_failed
    assert summary["failed"] == expected_failed
    assert summary["processed"] == 2
    assert summary["coverage"] == (2 - expected_failed) / 2
    assert summary["accuracy"] is None
    assert summary["provisional_accuracy"] == (None if all_failed else 1.0)
    assert summary["recorded_usage"]["output_tokens"] == 8
    assert summary["usage"]["output_tokens"] == (0 if all_failed else 4)
    assert provider.calls == 4
    first = sample_directory(tmp_path, "0", 0)
    saved = json.loads((first / "result.json").read_text())
    assert saved["score"] is None
    assert saved["answer"] == "candidate"
    assert saved["usage"]["output_tokens"] == 4
    assert saved["details"]["reason"].startswith(
        "HLE judge returned an invalid"
        if failure == "invalid_json"
        else "HLE judge did not complete"
    )
    assert saved["details"]["judge_stop_reason"] == (
        "stop" if failure == "invalid_json" else failure
    )
    assert not list(first.rglob("error.json"))
    assert not (first / "failure.json").exists()
    original = {p: p.read_bytes() for p in first.rglob("*.json")}
    resumed = await run(
        HLE(), examples, provider, Generation(100), tmp_path, **options, resume=True
    )
    assert resumed["status"] == "completed_with_errors"
    assert provider.calls == 4
    assert resumed["recorded_usage"] == summary["recorded_usage"]
    assert resumed["current_session_completed_output_tokens_per_second"] == 0
    assert original == {p: p.read_bytes() for p in first.rglob("*.json")}
