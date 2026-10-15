import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from inkling_verify.benchmarks import niah

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "niah_release.json").read_text())
GRID = {
    "context_lengths": REFERENCE["contexts"],
    "depths": REFERENCE["depths"],
    "samples_per_cell": REFERENCE["samples_per_cell"],
}
ANSWER = "972a8469-1641-4f82-8b9d-2434e465e150"


@pytest.fixture
def tokenizer_path(tmp_path):
    encoder = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    encoder.pre_tokenizer = pre_tokenizers.Whitespace()
    encoder.enable_truncation(max_length=4)
    encoder.enable_padding(length=16)
    path = tmp_path / "tokenizer.json"
    encoder.save(str(path))
    return path


@pytest.fixture
def prepared(tmp_path, tokenizer_path):
    benchmark = niah.NIAH()
    benchmark.prepare(tmp_path / "data", str(tokenizer_path), **GRID)
    return benchmark, tmp_path / "data", benchmark.load(tmp_path / "data")


@pytest.mark.parametrize("source", ["file", "directory", "huggingface"])
def test_prepare_matches_release_reference_and_reuses_cache(
    tmp_path, tokenizer_path, monkeypatch, source
):
    calls = []

    def download(repository, filename):
        calls.append((repository, filename))
        return str(tokenizer_path)

    monkeypatch.setattr(niah, "hf_hub_download", download)
    location = {
        "file": str(tokenizer_path),
        "directory": str(tmp_path),
        "huggingface": "vendor/model",
    }[source]
    benchmark = niah.NIAH()
    directory = tmp_path / "data"
    benchmark.prepare(directory, location, **GRID)
    examples = benchmark.load(directory)
    assert calls == ([("vendor/model", "tokenizer.json")] if source == "huggingface" else [])
    for example, expected in zip(examples, REFERENCE["cases"], strict=True):
        assert example.id == expected["id"] and example.target == expected["expected"]
        assert example.messages == ()
        assert {
            key: value for key, value in example.metadata.items() if key != "prompt_path"
        } == expected
        prompt = Path(example.metadata["prompt_path"]).read_bytes()
        assert hashlib.sha256(prompt).hexdigest() == expected["prompt_sha256"]
        assert prompt.endswith(b"mentioned in the provided text is")
        assert example.metadata["budgeted_total_tokens"] <= example.metadata["context_length"]
    manifest = json.loads((directory / "niah" / "manifest.json").read_text())
    assert manifest["grid"] == GRID
    assert manifest["tokenizer_sha256"] == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    def regenerate(**kwargs):
        pytest.fail("A valid cache must not regenerate prompts")

    monkeypatch.setattr(niah, "generate_sample", regenerate)
    benchmark.prepare(directory, str(tokenizer_path), **GRID)
    assert benchmark.load(directory) == examples


@pytest.mark.parametrize("changed", ["prompt", "examples", "revision"])
def test_load_rejects_corrupt_cache(prepared, changed):
    benchmark, directory, examples = prepared
    cache = directory / "niah"
    if changed == "prompt":
        Path(examples[0].metadata["prompt_path"]).write_text("changed")
    elif changed == "examples":
        (cache / "examples.jsonl").write_text("{}\n")
    else:
        path = cache / "manifest.json"
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps(manifest | {"revision": "changed"}))
    with pytest.raises(ValueError, match="checksum|revision"):
        benchmark.load(directory)


@pytest.mark.parametrize("changed", ["grid", "tokenizer"])
def test_prepare_rejects_different_grid_or_tokenizer(prepared, tokenizer_path, changed):
    benchmark, directory, _ = prepared
    grid = GRID
    if changed == "grid":
        grid = GRID | {"depths": [0]}
    else:
        tokenizer_path.write_text(tokenizer_path.read_text() + "\n")
    with pytest.raises(ValueError, match="grid or tokenizer differs"):
        benchmark.prepare(directory, str(tokenizer_path), **grid)


def test_too_small_context_fails_without_publishing_cache(tmp_path, tokenizer_path):
    directory = tmp_path / "data"
    with pytest.raises(ValueError, match="exceeds its context budget"):
        niah.NIAH().prepare(
            directory, str(tokenizer_path), context_lengths=[128], depths=[0], samples_per_cell=1
        )
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    "context_lengths, depths, samples_per_cell",
    [
        ([], [0], 1),
        ([512, 512], [0], 1),
        ([True], [0], 1),
        ([0], [0], 1),
        ([512], [], 1),
        ([512], [0, 0], 1),
        ([512], [101], 1),
        ([512], [-1], 1),
        ([512], [0], 0),
        ([512], [0], True),
    ],
)
def test_invalid_grid_is_rejected(context_lengths, depths, samples_per_cell):
    with pytest.raises(ValueError):
        niah.validate_grid(context_lengths, depths, samples_per_cell)


@pytest.mark.parametrize(
    "response, correct, extracted",
    [
        ("", False, ""),
        ("  \n ", False, ""),
        ("No match\nStill no match", False, "No match"),
        (ANSWER.upper(), True, ANSWER.upper()),
        ("Explanation.\n  " + ANSWER + "  \nDone.", True, ANSWER),
        ("prefix" + ANSWER + "suffix", True, "prefix" + ANSWER + "suffix"),
        (ANSWER.replace("-", ""), False, ANSWER.replace("-", "")),
    ],
)
def test_release_grading_searches_visible_text_without_punctuation_normalization(
    response, correct, extracted
):
    assert niah.grade_response(ANSWER, response) == (correct, extracted)


def test_grid_preserves_unfinished_and_unstarted_cells(tmp_path):
    ids = [
        f"niah_uuid_key_to_value_{context}_{depth}_{index}"
        for context, depth, index in [(512, 0, 1), (512, 0, 2), (512, 50, 3), (1024, 0, 4)]
    ]
    (tmp_path / "run.json").write_text(json.dumps({"examples": ids}))
    for index, context, score in [(0, 512, 1.0), (3, 1024, 0.0)]:
        directory = tmp_path / "samples" / hashlib.sha256(ids[index].encode()).hexdigest()
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "score": score,
                    "details": {"context_length": context, "depth_percent": 0},
                }
            )
        )
    niah.write_grid(tmp_path)
    cells = json.loads((tmp_path / "grid.json").read_text())
    fields = (
        "context_length",
        "depth_percent",
        "expected",
        "completed",
        "correct",
        "accuracy",
        "provisional_accuracy",
    )
    assert [tuple(cell[field] for field in fields) for cell in cells] == [
        (512, 0, 2, 1, 1.0, None, 1.0),
        (512, 50, 1, 0, 0, None, None),
        (1024, 0, 1, 1, 0.0, 0.0, 0.0),
    ]
