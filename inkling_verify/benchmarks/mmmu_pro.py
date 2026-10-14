"""MMMU-Pro vision and standard questions with shared multiple-choice extraction."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import random
import re
import string
import tempfile
import uuid
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from inkling_verify.benchmarks.data import read_cache, required_text, write_cache
from inkling_verify.cli import evaluate, parser
from inkling_verify.tool_loop import run_tools
from inkling_verify.types import (
    Evaluation,
    Example,
    Generation,
    Image,
    Message,
    Provider,
    Text,
)

REPOSITORY = "MMMU/MMMU_Pro"
REVISION = "1ba55708b8588a8f9b180b8fec9e6435c88ce363"
EXPECTED_COUNT = 1730
SUBSETS = {
    "vision": "vision",
    "standard-4": "standard (4 options)",
    "standard-10": "standard (10 options)",
}
PROMPT = (
    "Write out the multiple-choice question in the image and then solve it. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER "
    "is one of options. Think step by step before answering."
)
STANDARD_PROMPT = (
    "Answer the preceding multiple choice question. The last line of your response should be of "
    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. "
    "Think step by step before answering."
)
IMAGE_TOKEN = re.compile(r"<image\s+(\d+)>")
TOOLS_MAX_TURNS = 128
TOOLS_MAX_CALLS = 128
TOOLS_MAX_TURN_TOKENS = 128 * 1024


def parse_answer(
    response: str, choices: list[str], options: dict[str, str], *, task_id: str
) -> tuple[str, str]:
    if not choices or set(options) != set(choices):
        raise ValueError("Choices and option text must be nonempty and agree")
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    last_answer = response.rfind("Answer:")
    if last_answer != -1:
        suffix = response[last_answer + len("Answer:") :].strip()
        matching = [choice for choice in choices if choice in suffix]
        if len(matching) == 1:
            return matching[0], "answer_prefix"
    for character in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(character)
    response = " " + response + " "
    index_answer = True
    bracketed = False
    candidates = [choice for choice in choices if f"({choice})" in response]
    if candidates:
        bracketed = True
    if not candidates:
        candidates = [choice for choice in choices if f"{choice} " in response]
    if not candidates:
        candidates = [choice for choice in choices if f"{choice}." in response]
    if not candidates and len(response.split()) > 5:
        candidates = [key for key, value in options.items() if value.lower() in response.lower()]
        index_answer = False
    if not candidates:
        seed = int.from_bytes(hashlib.sha256(f"{task_id}/0".encode()).digest()[:8], "big")
        return random.Random(seed).choice(choices), "random_fallback"
    if len(candidates) > 1:
        if index_answer:
            positions = [
                response.rfind(f"({choice})" if bracketed else f" {choice} ")
                for choice in candidates
            ]
        else:
            positions = [response.lower().rfind(options[choice].lower()) for choice in candidates]
        return candidates[max(range(len(positions)), key=positions.__getitem__)], "last_match"
    return candidates[0], "single_match"


def _task_id(source_id: str, subset: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{REPOSITORY}/{subset}/test/{source_id}"))


def _standard_prompt(question: str, options: dict[str, str]) -> str:
    choices = "\n".join(f"{label}. {option}" for label, option in options.items())
    return f"{question}\n{choices}\n{STANDARD_PROMPT}"


def _validate_image_references(prompt: str, images: dict[str, Any], source_id: str) -> None:
    for match in IMAGE_TOKEN.finditer(prompt):
        key = f"image_{int(match[1])}"
        if key not in images:
            raise ValueError(f"Missing {key} referenced by {source_id}")


def _save_image(image: Any, directory: Path, filename: str, source_id: str) -> dict[str, str]:
    if not isinstance(image, PILImage.Image):
        raise ValueError(f"Expected decoded image for {source_id}")
    image = image.convert("RGBA" if image.has_transparency_data else "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = buffer.getvalue()
    relative = f"images/{filename}"
    (directory / relative).write_bytes(encoded)
    return {"image": relative, "image_sha256": hashlib.sha256(encoded).hexdigest()}


def _prepare_row(row: dict[str, Any], directory: Path, subset: str = "vision") -> dict[str, Any]:
    if subset not in SUBSETS:
        raise ValueError(f"Unknown MMMU-Pro subset: {subset}")
    source_id = required_text(row, "id")
    subject = required_text(row, "subject")
    options = ast.literal_eval(required_text(row, "options"))
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise ValueError(f"Invalid options for {source_id}")
    options = {label: str(option) for label, option in zip(string.ascii_uppercase, options)}
    answer = required_text(row, "answer").strip().upper()
    if answer not in options:
        raise ValueError(f"Invalid answer for {source_id}")
    task_id = _task_id(source_id, subset)
    if subset == "vision":
        prompt = "<image 1>" + PROMPT
        images = {"image_1": row["image"]}
    else:
        prompt = _standard_prompt(required_text(row, "question"), options)
        images = {
            f"image_{index}": row[f"image_{index}"]
            for index in range(1, 8)
            if row[f"image_{index}"] is not None
        }
    _validate_image_references(prompt, images, source_id)
    return {
        "id": task_id,
        "source_id": source_id,
        "subject": subject,
        "options": options,
        "answer": answer,
        "prompt": prompt,
        "images": {
            key: _save_image(image, directory, f"{task_id}-{key}.png", source_id)
            for key, image in images.items()
        },
    }


def _load_image(directory: Path, record: dict[str, str], filename: str) -> dict[str, str]:
    image_path = directory / "images" / filename
    if record["image"] != f"images/{filename}" or not image_path.is_file():
        raise ValueError(f"Invalid or missing image: {filename}")
    with image_path.open("rb") as image_file:
        image_hash = hashlib.file_digest(image_file, "sha256").hexdigest()
    if image_hash != record["image_sha256"]:
        raise ValueError(f"Image checksum mismatch: {filename}")
    return {
        "image_path": str(image_path.resolve()),
        "image_sha256": record["image_sha256"],
    }


def _read_image(record: dict[str, str]) -> Image:
    image = Path(record["image_path"]).read_bytes()
    if hashlib.sha256(image).hexdigest() != record["image_sha256"]:
        raise ValueError(f"Image checksum mismatch: {record['image_path']}")
    return Image(image, "image/png")


class MMMUPro:
    def __init__(self, subset: str = "vision", *, with_tools: bool = False) -> None:
        if subset not in SUBSETS:
            raise ValueError(f"Unknown MMMU-Pro subset: {subset}")
        self.subset = subset
        self.with_tools = with_tools
        self.cache_name = f"mmmu-pro-{subset}"
        self.name = self.cache_name + ("-with-tools" if with_tools else "")

    def prepare(self, data_dir: Path) -> None:
        from datasets import load_dataset

        destination = data_dir / self.cache_name
        if destination.exists():
            self.load(data_dir)
            return
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(REPOSITORY, SUBSETS[self.subset], split="test", revision=REVISION)
        if len(dataset) != EXPECTED_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_COUNT} MMMU-Pro {self.subset} rows, got {len(dataset)}"
            )
        with tempfile.TemporaryDirectory(prefix="mmmu-prepare-", dir=data_dir) as temporary:
            staging = Path(temporary) / self.cache_name
            (staging / "images").mkdir(parents=True)
            rows = [_prepare_row(row, staging, self.subset) for row in dataset]
            if len({row["id"] for row in rows}) != EXPECTED_COUNT:
                raise ValueError("Duplicate MMMU-Pro IDs")
            write_cache(staging, rows, repository=REPOSITORY, revision=REVISION)
            staging.rename(destination)

    def load(self, data_dir: Path) -> list[Example]:
        directory = data_dir / self.cache_name
        rows = read_cache(directory, repository=REPOSITORY, revision=REVISION, count=EXPECTED_COUNT)
        examples = []
        for row in rows:
            if "prompt" not in row:
                raise ValueError("Unsupported MMMU-Pro cache format; prepare in a fresh data-dir")
            if row["id"] != _task_id(required_text(row, "source_id"), self.subset):
                raise ValueError(f"MMMU-Pro {self.subset} example ID differs: {row['id']}")
            options = row["options"]
            if not isinstance(options, dict) or row["answer"] not in options:
                raise ValueError(f"Invalid choices for {row['id']}")
            prompt = required_text(row, "prompt")
            images = row["images"]
            if not isinstance(images, dict) or not set(images) <= {
                f"image_{index}" for index in range(1, 8)
            }:
                raise ValueError(f"Invalid images for {row['id']}")
            _validate_image_references(prompt, images, row["id"])
            examples.append(
                Example(
                    id=row["id"],
                    messages=(Message("user", (Text(prompt),)),),
                    target=row["answer"],
                    metadata={
                        "subject": row["subject"],
                        "source_id": row["source_id"],
                        "options": options,
                        "images": {
                            key: _load_image(directory, record, f"{row['id']}-{key}.png")
                            for key, record in images.items()
                        },
                    },
                )
            )
        return examples

    async def evaluate(
        self,
        example: Example,
        provider: Provider,
        generation: Generation,
        work_dir: Path,
        *,
        judge: Provider | None = None,
    ) -> Evaluation:
        prompt = example.messages[0].text
        images = example.metadata["images"]
        _validate_image_references(prompt, images, example.id)
        parts: list[Text | Image] = []
        start = 0
        for match in IMAGE_TOKEN.finditer(prompt):
            if match.start() > start:
                parts.append(Text(prompt[start : match.start()]))
            parts.append(_read_image(images[f"image_{int(match[1])}"]))
            start = match.end()
        if start < len(prompt):
            parts.append(Text(prompt[start:]))
        messages = (Message("user", tuple(parts)),)
        if self.with_tools:
            from inkling_verify.python_tools import SYSTEM_PROMPT, TOOL, PythonSandbox

            async with PythonSandbox(work_dir) as sandbox:
                staged = await sandbox.stage(messages)
                text, details = await run_tools(
                    provider,
                    generation,
                    [Message("system", (Text(SYSTEM_PROMPT),)), *staged],
                    TOOL,
                    sandbox.execute,
                    max_turns=TOOLS_MAX_TURNS,
                    max_tool_calls=TOOLS_MAX_CALLS,
                    max_turn_tokens=TOOLS_MAX_TURN_TOKENS,
                )
            details["with_tools"] = True
            details["allow_network"] = False
        else:
            response = await provider.generate(messages, generation)
            if response.stop_reason == "tool_calls" or response.message.tool_calls:
                raise ValueError("MMMU-Pro requested no tools but received a tool call")
            text = response.text
            details = {"stop_reason": response.stop_reason}
        text = text.strip()
        if not text:
            return Evaluation(0.0, "", {"parse_method": "empty", **details})
        options = example.metadata["options"]
        predicted, method = parse_answer(text, list(options), options, task_id=example.id)
        return Evaluation(
            float(predicted == example.target),
            predicted,
            {"parse_method": method, **details},
        )


def main() -> None:
    root, prepare, run = parser("mmmu-pro")
    for command in (prepare, run):
        command.add_argument("--subset", choices=SUBSETS, default="vision")
    run.add_argument(
        "--with-tools", action="store_true", help="enable an offline, persistent Python image tool"
    )
    args = root.parse_args()
    if args.command == "prepare":
        benchmark = MMMUPro(args.subset)
        benchmark.prepare(args.data_dir.resolve())
        print(f"Prepared {benchmark.name} in {args.data_dir}")
    else:
        if args.with_tools:
            if args.max_output_tokens is None:
                args.max_output_tokens = 512 * 1024
            if args.concurrency is None:
                args.concurrency = 4
        asyncio.run(evaluate(MMMUPro(args.subset, with_tools=args.with_tools), args))


if __name__ == "__main__":
    main()
