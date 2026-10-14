import hashlib
import io
import json
import struct
import wave
from pathlib import Path

import huggingface_hub
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from inkling_verify.benchmarks import mmau


def pcm_wav(samples, sample_rate=16000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


@pytest.mark.parametrize("count", [2, 4, 5, 8])
def test_prompt_preserves_release_instruction_and_variable_choice_count(count):
    choices = [f"choice {index}" for index in range(count)]
    assert mmau.build_prompt("What is audible?", choices) == (
        "What is audible?\n\nChoice: \n"
        + "\n".join(choices)
        + f"\nChoose a choices from the given {count} choices. Do not provide any "
        "additional explanations or content. Output must match exactly one of the listed choices."
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("(A) Dog", "Dog"),
        ("(h) Piano", "Piano"),
        ("A. Dog", "A. Dog"),
        ("(A)Dog", "(A)Dog"),
        ("(1) Dog", "(1) Dog"),
        ("", ""),
    ],
)
def test_letter_prefix_matches_reference_exactly(source, expected):
    assert mmau.strip_letter_prefix(source) == expected


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("A DOG is barking!", True),
        ("dog dog barking", True),
        ("It is a dog barking nearby.", True),
        ("dog barking or cat meowing", False),
        ("dog", False),
        ("A", False),
        ("", False),
    ],
)
def test_official_matcher_requires_answer_words_and_excludes_distractor_words(prediction, expected):
    choices = ["dog barking", "cat meowing", "dog whining", "bird singing"]
    assert mmau.mmau_string_match(choices[0], prediction, choices) is expected


def test_official_matcher_uses_sets_even_when_choices_only_differ_in_order():
    assert mmau.mmau_string_match("red blue", "BLUE, red", ["red blue", "blue red"])


def test_mono_16k_preserves_pcm_samples():
    samples = [-32768, -12345, -1, 0, 1, 12345, 32767]
    original = pcm_wav(samples)
    assert mmau.audio_to_wav(original) == original


def test_stereo_48k_downmixes_before_resampling():
    source = np.random.default_rng(7).integers(-32768, 32768, (480, 2), dtype=np.int16)
    result = mmau.audio_to_wav(pcm_wav(source.ravel(), 48000, 2))
    assert hashlib.sha256(result).hexdigest() == (
        "7f30d988017f65be94c256fff7ef2cd4c62711292ad9db139c10193f21dff271"
    )
    with wave.open(io.BytesIO(result)) as audio:
        assert (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) == (1, 16000, 2)
        assert audio.getnframes() == 160


@pytest.fixture
def source_dataset(tmp_path, monkeypatch):
    rows = []
    audio = pcm_wav([-1000, 0, 1000])
    for index, (task, count) in enumerate(zip(("sound", "sound", "speech", "music"), (2, 4, 5, 8))):
        choices = [f"({chr(65 + option)}) choice {option}" for option in range(count)]
        rows.append(
            {
                "context": {"bytes": audio, "path": None},
                "instruction": "What is audible?",
                "choices": choices,
                "answer": choices[-1],
                "other_attributes": json.dumps({"id": f"source/{index}", "task": task}),
            }
        )
    path = tmp_path / "source.parquet"
    calls = []

    def download(repository, filename, *, repo_type, revision):
        calls.append((repository, filename, repo_type, revision))
        pq.write_table(pa.Table.from_pylist(rows), path)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(mmau, "EXPECTED_COUNT", len(rows))
    monkeypatch.setattr(mmau, "TASKS", {"sound": 2, "speech": 1, "music": 1})
    return rows, calls, audio


def test_prepare_pins_download_preserves_metadata_ids_and_reuses_shared_cache(
    source_dataset, tmp_path
):
    rows, calls, audio = source_dataset
    data_dir = tmp_path / "data"
    mmau.MMAU("speech").prepare(data_dir)
    mmau.MMAU("music").prepare(data_dir)
    assert calls == [(mmau.REPOSITORY, "test_mini.parquet", "dataset", mmau.REVISION)]
    examples = mmau.MMAU().load(data_dir)
    assert [item.metadata["task"] for item in examples[:3]] == ["sound", "speech", "music"]
    assert {item.id for item in examples} == {f"source/{index}" for index in range(4)}
    assert {len(item.metadata["choices"]) for item in examples} == {2, 4, 5, 8}
    for example in examples:
        source = rows[int(example.id.rsplit("/", 1)[1])]
        assert example.target == source["answer"][4:]
        assert Path(example.metadata["audio_path"]).read_bytes() == audio
        assert example.metadata["audio_sha256"] == hashlib.sha256(audio).hexdigest()
        assert example.messages[0].text == mmau.build_prompt(
            source["instruction"], [choice[4:] for choice in source["choices"]]
        )
    assert [item.id for item in mmau.MMAU("sound").load(data_dir)] == ["source/0", "source/1"]


@pytest.mark.parametrize("corruption", ["revision", "examples"])
def test_existing_cache_rejects_stale_revision_or_corrupted_rows(
    source_dataset, tmp_path, corruption
):
    _, calls, _ = source_dataset
    benchmark = mmau.MMAU()
    benchmark.prepare(tmp_path)
    directory = tmp_path / "mmau"
    if corruption == "revision":
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps(manifest | {"revision": "wrong-revision"}))
    else:
        with (directory / "examples.jsonl").open("ab") as file:
            file.write(b"\n")
    with pytest.raises(ValueError, match="revision differs|checksum mismatch"):
        benchmark.prepare(tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("corruption", ["duplicate_id", "task_count", "answer"])
def test_prepare_rejects_invalid_source_without_publishing_cache(
    source_dataset, tmp_path, corruption
):
    rows, _, _ = source_dataset
    if corruption == "answer":
        rows[-1]["answer"] = "(A) not a choice"
    else:
        metadata = json.loads(rows[-1]["other_attributes"])
        metadata["id" if corruption == "duplicate_id" else "task"] = (
            "source/0" if corruption == "duplicate_id" else "sound"
        )
        rows[-1]["other_attributes"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="distinct MMAU|task counts|Invalid MMAU"):
        mmau.MMAU().prepare(tmp_path)
    assert not (tmp_path / "mmau").exists()
