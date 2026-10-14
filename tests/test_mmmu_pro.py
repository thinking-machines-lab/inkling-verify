import hashlib
import io
import random
import uuid

import pytest
from PIL import Image as PILImage

from inkling_vendor_verifier.benchmarks import mmmu_pro
from inkling_vendor_verifier.benchmarks.data import write_cache
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


def source_row():
    return {
        "id": "standard-synthetic-1",
        "subject": "testing",
        "question": "Compare <image 2> with <image 1>. Again: <image 2>.",
        "options": "['<image 2>', '<image 1>', 'neither']",
        "answer": "B",
        "explanation": "SECRET REFERENCE EXPLANATION",
        "image": PILImage.new("RGBA", (7, 9), (3, 5, 7, 99)),
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

    source = source_row()
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


@pytest.mark.parametrize("subset", mmmu_pro.SUBSETS)
async def test_subsets_preserve_ids_prompt_and_image_bytes_and_reject_corruption(
    tmp_path, monkeypatch, subset
):
    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    source = source_row() | {"answer": " B "}
    benchmark, directory, row = cache_row(tmp_path, source, subset)
    example = benchmark.load(tmp_path)[0]
    assert example.id == str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"MMMU/MMMU_Pro/{subset}/test/{source['id']}")
    )
    assert MMMUPro(subset, with_tools=True).load(tmp_path) == [example]
    originals = (
        {"image_1": source["image"]}
        if subset == "vision"
        else {key: source[key] for key in ("image_1", "image_2")}
    )
    assert set(row["images"]) == set(originals)
    image_parts = {}
    for key, image in originals.items():
        path = directory / row["images"][key]["image"]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = buffer.getvalue()
        assert path.read_bytes() == encoded
        assert hashlib.sha256(encoded).hexdigest() == row["images"][key]["image_sha256"]
        image_parts[key] = Image(encoded, "image/png")
    expected = (
        (image_parts["image_1"], Text(PROMPT))
        if subset == "vision"
        else (
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
    image_path = directory / row["images"]["image_1"]["image"]
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
    source = source_row() | {
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
        _prepare_row(source_row() | overrides, tmp_path, subset)


@pytest.mark.parametrize("source_subset", ["vision", "standard-4", "standard-10"])
def test_cache_rejects_another_subset_or_stale_format(tmp_path, monkeypatch, source_subset):
    import shutil

    monkeypatch.setattr(mmmu_pro, "EXPECTED_COUNT", 1)
    source = source_row()
    original, directory, row = cache_row(tmp_path, source, source_subset)
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
    row.pop("prompt")
    write_cache(directory, [row], repository=mmmu_pro.REPOSITORY, revision=mmmu_pro.REVISION)
    with pytest.raises(ValueError, match="cache format; prepare in a fresh data-dir"):
        original.load(tmp_path)
    with pytest.raises(ValueError, match="cache format; prepare in a fresh data-dir"):
        original.prepare(tmp_path)


@pytest.mark.parametrize("subset", ["standard", "standard-5", "Vision", ""])
def test_unknown_subset_is_rejected(subset):
    with pytest.raises(ValueError, match="Unknown MMMU-Pro subset"):
        MMMUPro(subset)
