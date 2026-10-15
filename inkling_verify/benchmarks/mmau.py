"""MMAU v05.15.25 audio understanding with the official token matcher."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from inkling_verify.benchmarks.data import read_cache, required_text, write_cache
from inkling_verify.cli import evaluate, parser
from inkling_verify.types import (
    Audio,
    Evaluation,
    Example,
    Generation,
    Message,
    Provider,
    Text,
)

REPOSITORY = "gamma-lab-umd/MMAU-test-mini"
REVISION = "ccd9696c0111ea7060827598f310558df0b71b0a"
TASKS = {"sound": 333, "speech": 333, "music": 334}
EXPECTED_COUNT = sum(TASKS.values())


def strip_letter_prefix(text: str) -> str:
    if len(text) >= 4 and text[0] == "(" and text[2:4] == ") " and text[1].isalpha():
        return text[4:]
    return text


def build_prompt(question: str, choices: list[str]) -> str:
    choices_text = "\n".join(choices)
    return (
        f"{question}\n\nChoice: \n{choices_text}\n"
        f"Choose a choices from the given {len(choices)} choices. Do not provide any "
        "additional explanations or content. Output must match exactly one of the listed choices."
    )


def mmau_string_match(answer: str, prediction: str, choices: list[str]) -> bool:
    def tokens(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    prediction_tokens = tokens(prediction)
    answer_tokens = tokens(answer)
    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = tokens(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    return (
        bool(prediction_tokens)
        and answer_tokens.issubset(prediction_tokens)
        and prediction_tokens.isdisjoint(incorrect_tokens)
    )


def audio_to_wav(raw: bytes) -> bytes:
    import librosa
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(raw))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _audio_path(directory: Path, source_id: str) -> Path:
    return directory / "audio" / (hashlib.sha256(source_id.encode()).hexdigest() + ".wav")


class MMAU:
    cache_name = "mmau"

    def __init__(self, subset: str = "all") -> None:
        if subset not in ("all", *TASKS):
            raise ValueError(f"Unknown MMAU subset: {subset}")
        self.subset = subset
        self.name = "mmau" if subset == "all" else f"mmau-{subset}"

    def prepare(self, data_dir: Path) -> None:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        destination = data_dir / self.cache_name
        if destination.exists():
            self.load(data_dir)
            return
        data_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            REPOSITORY, "test_mini.parquet", repo_type="dataset", revision=REVISION
        )
        with tempfile.TemporaryDirectory(prefix="mmau-prepare-", dir=data_dir) as temporary:
            staging = Path(temporary) / self.cache_name
            (staging / "audio").mkdir(parents=True)
            rows = []
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
                row = batch.to_pylist()[0]
                attributes = json.loads(row["other_attributes"])
                source_id = required_text(attributes, "id")
                task = attributes["task"]
                choices = [strip_letter_prefix(choice) for choice in row["choices"]]
                answer = strip_letter_prefix(required_text(row, "answer"))
                if task not in TASKS or len(choices) < 2 or answer not in choices:
                    raise ValueError(f"Invalid MMAU task or choices: {source_id}")
                audio = audio_to_wav(row["context"]["bytes"])
                _audio_path(staging, source_id).write_bytes(audio)
                rows.append(
                    {
                        "id": source_id,
                        "task": task,
                        "prompt": build_prompt(required_text(row, "instruction"), choices),
                        "choices": choices,
                        "answer": answer,
                        "audio_sha256": hashlib.sha256(audio).hexdigest(),
                    }
                )
            if len(rows) != EXPECTED_COUNT or len({row["id"] for row in rows}) != EXPECTED_COUNT:
                raise ValueError(f"Expected {EXPECTED_COUNT} distinct MMAU examples")
            if Counter(row["task"] for row in rows) != TASKS:
                raise ValueError("MMAU task counts differ from the pinned dataset")
            write_cache(staging, rows, repository=REPOSITORY, revision=REVISION)
            staging.rename(destination)

    def load(self, data_dir: Path) -> list[Example]:
        directory = data_dir / self.cache_name
        rows = read_cache(directory, repository=REPOSITORY, revision=REVISION, count=EXPECTED_COUNT)
        if self.subset == "all":
            groups = [[row for row in rows if row["task"] == task] for task in TASKS]
            rows = [row for group in zip_longest(*groups) for row in group if row is not None]
        return [
            Example(
                row["id"],
                (Message("user", (Text(row["prompt"]),)),),
                row["answer"],
                {
                    "task": row["task"],
                    "choices": row["choices"],
                    "audio_path": str(_audio_path(directory, row["id"]).resolve()),
                    "audio_sha256": row["audio_sha256"],
                },
            )
            for row in rows
            if self.subset == "all" or row["task"] == self.subset
        ]

    async def evaluate(
        self,
        example: Example,
        provider: Provider,
        generation: Generation,
        work_dir: Path,
        *,
        judge: Provider | None = None,
    ) -> Evaluation:
        audio = Path(example.metadata["audio_path"]).read_bytes()
        if hashlib.sha256(audio).hexdigest() != example.metadata["audio_sha256"]:
            raise ValueError(f"Audio checksum mismatch: {example.id}")
        messages = (Message("user", (Text(example.messages[0].text), Audio(audio))),)
        response = await provider.generate(messages, generation)
        if response.stop_reason == "tool_calls" or response.message.tool_calls:
            raise ValueError("MMAU requested no tools but received a tool call")
        return Evaluation(
            float(mmau_string_match(example.target, response.text, example.metadata["choices"])),
            response.text,
            {"task": example.metadata["task"], "stop_reason": response.stop_reason},
        )


def main() -> None:
    root, _, run = parser("MMAU audio understanding")
    run.add_argument("--subset", choices=("all", *TASKS), default="all")
    args = root.parse_args()
    if args.command == "prepare":
        MMAU().prepare(args.data_dir.resolve())
        print(f"Prepared MMAU in {args.data_dir}")
    else:
        if args.api != "openai-completions":
            root.error("MMAU requires an audio-capable Chat Completions endpoint")
        asyncio.run(evaluate(MMAU(args.subset), args))


if __name__ == "__main__":
    main()
