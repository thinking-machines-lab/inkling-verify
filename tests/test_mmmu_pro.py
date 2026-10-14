import hashlib
import json
import random

import pytest
from PIL import Image as PILImage

from inkling_vendor_verifier.benchmarks import mmmu_pro
from inkling_vendor_verifier.benchmarks.data import read_cache, write_cache
from inkling_vendor_verifier.benchmarks.mmmu_pro import (
    PROMPT,
    MMMUPro,
    _prepare_row,
    parse_answer,
)
from inkling_vendor_verifier.types import Generation, Image, Message, Response, Text, Usage

OPTIONS = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Answer: A\nI reconsidered. Answer: B", ("B", "answer_prefix")),
        ("<think>Answer: A</think>Answer: C", ("C", "answer_prefix")),
        ("First (A), ultimately (D).", ("D", "last_match")),
        ("B", ("B", "single_match")),
        ("B. is selected", ("B", "single_match")),
        ("We can infer that the color is blue", ("B", "single_match")),
        ("B. or C. please", ("B", "last_match")),
    ],
)
def test_reference_extraction_order(response, expected):
    assert parse_answer(response, list(OPTIONS), OPTIONS, task_id="task") == expected


def test_random_fallback_is_task_stable_and_does_not_change_global_rng():
    state = random.getstate()
    seed = int.from_bytes(hashlib.sha256(b"task/0").digest()[:8], "big")
    expected = random.Random(seed).choice(list(OPTIONS))
    assert parse_answer("unknown", list(OPTIONS), OPTIONS, task_id="task") == (
        expected,
        "random_fallback",
    )
    assert random.getstate() == state


def test_cache_rejects_changed_data_and_wrong_revision(tmp_path):
    write_cache(tmp_path, [{"id": "one"}], repository="test", revision="pinned")
    assert read_cache(tmp_path, repository="test", revision="pinned", count=1) == [{"id": "one"}]
    with pytest.raises(ValueError, match="revision"):
        read_cache(tmp_path, repository="test", revision="wrong", count=1)
    (tmp_path / "examples.jsonl").write_text(json.dumps({"id": "two"}) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        read_cache(tmp_path, repository="test", revision="pinned", count=1)


def test_load_rejects_image_corruption_before_completed_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    directory = tmp_path / MMMUPro().name
    (directory / "images").mkdir(parents=True)
    row = _prepare_row(
        {
            "id": "synthetic-1",
            "subject": "testing",
            "options": "['a', 'b']",
            "answer": "A",
            "image": PILImage.new("RGB", (2, 2)),
        },
        directory,
    )
    write_cache(directory, [row], repository=mmmu_pro.REPOSITORY, revision=mmmu_pro.REVISION)
    assert len(MMMUPro().load(tmp_path)) == 1
    assert MMMUPro(with_tools=True).load(tmp_path) == MMMUPro().load(tmp_path)
    (directory / row["image"]).write_bytes(b"corrupted after completion")
    with pytest.raises(ValueError, match="Image checksum mismatch"):
        MMMUPro().load(tmp_path)


def standard_row():
    return {
        "id": "standard-synthetic-1",
        "subject": "testing",
        "question": "Compare <image 2> with <image 1>. Again: <image 2>.",
        "options": "['<image 2>', '<image 1>', 'neither']",
        "answer": "B",
        "explanation": "SECRET REFERENCE EXPLANATION",
        **{f"image_{index}": None for index in range(1, 8)},
        "image_1": PILImage.new("RGB", (2, 3), (3, 5, 7)),
        "image_2": PILImage.new("RGBA", (4, 2), (11, 13, 17, 99)),
    }


def cache_row(tmp_path, source, subset):
    benchmark = MMMUPro(subset)
    directory = tmp_path / benchmark.name
    (directory / "images").mkdir(parents=True)
    row = _prepare_row(source, directory, subset)
    write_cache(directory, [row], repository=mmmu_pro.REPOSITORY, revision=mmmu_pro.REVISION)
    return benchmark, directory, row


@pytest.mark.parametrize(
    "subset, config",
    [
        ("vision", "vision"),
        ("standard-4", "standard (4 options)"),
        ("standard-10", "standard (10 options)"),
    ],
)
def test_prepare_downloads_pinned_subset_and_reuses_only_its_cache(
    tmp_path, monkeypatch, subset, config
):
    import datasets

    source = standard_row() | {"image": PILImage.new("RGB", (2, 3))}
    calls = []

    def load_dataset(repository, name, *, split, revision):
        calls.append((repository, name, split, revision))
        return [source]

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    benchmark = MMMUPro(subset)
    benchmark.prepare(tmp_path)
    benchmark.prepare(tmp_path)
    assert benchmark.name == f"mmmu-pro-{subset}"
    assert calls == [(mmmu_pro.REPOSITORY, config, "test", mmmu_pro.REVISION)]
    example = benchmark.load(tmp_path)[0]
    assert example.target == "B"
    assert example.metadata["source_id"] == source["id"]
    assert "SECRET" not in repr(example)


def test_default_vision_retains_original_uuid_cache_and_prompt(tmp_path, monkeypatch):
    import io
    import uuid

    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    image = PILImage.new("RGBA", (7, 9), (3, 5, 7, 99))
    source = {
        "id": "synthetic-1",
        "subject": "testing",
        "options": "['a', 'b']",
        "answer": " B ",
        "image": image,
    }
    benchmark, directory, row = cache_row(tmp_path, source, "vision")
    task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "MMMU/MMMU_Pro/vision/test/synthetic-1"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    assert row == {
        "id": task_id,
        "source_id": "synthetic-1",
        "subject": "testing",
        "options": {"A": "a", "B": "b"},
        "answer": "B",
        "image": f"images/{task_id}.png",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    assert (directory / row["image"]).read_bytes() == image_bytes
    assert MMMUPro().name == benchmark.name == "mmmu-pro-vision"
    example = benchmark.load(tmp_path)[0]
    assert example.messages == (Message("user", (Text(PROMPT),)),)
    assert example.metadata == {
        "subject": "testing",
        "source_id": "synthetic-1",
        "options": {"A": "a", "B": "b"},
        "image_path": str((directory / row["image"]).resolve()),
        "image_sha256": row["image_sha256"],
    }


@pytest.mark.parametrize("subset", ["standard-4", "standard-10"])
async def test_standard_interleaves_question_and_shuffled_option_images_in_reference_order(
    tmp_path, monkeypatch, subset
):
    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    source = standard_row()
    benchmark, directory, row = cache_row(tmp_path, source, subset)
    example = benchmark.load(tmp_path)[0]
    assert set(row["images"]) == {"image_1", "image_2"}
    image_parts = {}
    for key in ("image_1", "image_2"):
        path = directory / row["images"][key]["image"]
        encoded = path.read_bytes()
        assert hashlib.sha256(encoded).hexdigest() == row["images"][key]["image_sha256"]
        with PILImage.open(path) as decoded:
            assert decoded.mode == source[key].mode
            assert decoded.size == source[key].size
            assert decoded.tobytes() == source[key].tobytes()
        image_parts[key] = Image(encoded, "image/png")
    expected = (
        Text("Compare "),
        image_parts["image_2"],
        Text(" with "),
        image_parts["image_1"],
        Text(". Again: "),
        image_parts["image_2"],
        Text(".\nA. "),
        image_parts["image_2"],
        Text("\nB. "),
        image_parts["image_1"],
        Text("\nC. neither\n" + mmmu_pro.STANDARD_PROMPT),
    )

    class FakeProvider:
        async def generate(self, messages, generation, **kwargs):
            assert messages == (Message("user", expected),)
            assert kwargs == {}
            return Response(Message("assistant", (Text("Answer: B"),)), Usage(1, 1), "stop", {}, {})

    result = await benchmark.evaluate(example, FakeProvider(), Generation(20), tmp_path)
    assert result.score == 1.0
    assert result.answer == "B"
    assert result.details["parse_method"] == "answer_prefix"
    image_path = directory / row["images"]["image_2"]["image"]
    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image checksum mismatch"):
        benchmark.load(tmp_path)
    with pytest.raises(ValueError, match="Image checksum mismatch"):
        await benchmark.evaluate(example, FakeProvider(), Generation(20), tmp_path)


@pytest.mark.parametrize("subset, choices", [("standard-4", 5), ("standard-10", 9)])
async def test_standard_accepts_nullable_images_and_dataset_variable_option_counts(
    tmp_path, monkeypatch, subset, choices
):
    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    source = standard_row() | {
        "question": "Which label is correct?",
        "options": repr([f"label {index}" for index in range(choices)]),
        "answer": chr(ord("A") + choices - 1),
        **{f"image_{index}": None for index in range(1, 8)},
    }
    benchmark, _, row = cache_row(tmp_path, source, subset)
    example = benchmark.load(tmp_path)[0]
    assert row["images"] == {}
    assert len(example.metadata["options"]) == choices

    class FakeProvider:
        async def generate(self, messages, generation, **kwargs):
            assert messages == example.messages
            assert all(isinstance(part, Text) for part in messages[0].content)
            assert "A. label 0\nB. label 1" in messages[0].text
            assert "SECRET" not in messages[0].text
            return Response(
                Message("assistant", (Text(f"Answer: {source['answer']}"),)),
                Usage(1, 1),
                "stop",
                {},
                {},
            )

    result = await benchmark.evaluate(example, FakeProvider(), Generation(20), tmp_path)
    assert result.score == 1.0
    assert result.answer == source["answer"]


@pytest.mark.parametrize(
    "subset, overrides, missing",
    [
        ("standard-10", {"question": "<image 0>"}, "image_0"),
        ("standard-10", {"question": "<image 3>"}, "image_3"),
        ("standard-10", {"question": "<image 8>"}, "image_8"),
        ("standard-4", {"image_2": None}, "image_2"),
    ],
)
def test_standard_missing_image_reference_fails(tmp_path, subset, overrides, missing):
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match=f"Missing {missing}"):
        _prepare_row(standard_row() | overrides, tmp_path, subset)


@pytest.mark.parametrize("source_subset", ["vision", "standard-4", "standard-10"])
def test_cache_cannot_be_reused_for_another_subset(tmp_path, monkeypatch, source_subset):
    import shutil

    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    source = standard_row() | {"image": PILImage.new("RGB", (2, 3))}
    _, directory, row = cache_row(tmp_path, source, source_subset)
    ids = {row["id"]}
    for subset in mmmu_pro.SUBSETS:
        if subset == source_subset:
            continue
        benchmark = MMMUPro(subset)
        shutil.copytree(directory, tmp_path / benchmark.name)
        with pytest.raises(ValueError, match="example ID differs"):
            benchmark.load(tmp_path)
        (tmp_path / "images").mkdir(exist_ok=True)
        ids.add(_prepare_row(source, tmp_path, subset)["id"])
    assert len(ids) == 3


@pytest.mark.parametrize("subset", ["standard", "standard-5", "Vision", ""])
def test_unknown_subset_is_rejected(subset):
    with pytest.raises(ValueError, match="Unknown MMMU-Pro subset"):
        MMMUPro(subset)
