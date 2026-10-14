"""Deterministic UUID retrieval across context lengths and needle depths."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from inkling_vendor_verifier.benchmarks.data import read_cache, write_cache
from inkling_vendor_verifier.cli import evaluate, parser, positive
from inkling_vendor_verifier.runner import write_json
from inkling_vendor_verifier.types import Evaluation, Example, Generation, Message, Provider, Text

REPOSITORY = "synthetic/niah-uuid-key-to-value"
REVISION = "1"
CONTEXT_LENGTHS = (4000, 16000, 64000, 256000, 1024000)
DEPTHS = tuple(range(0, 101, 10))
# Keep prepared prompts identical to the original release grid.
REFERENCE_OUTPUT_RESERVE = 128


NEEDLE_TEMPLATE = "One of the special magic uuids for {key} is: {value}."


KEY_TO_VALUE_QUERY_TEMPLATE = (
    "A special magic uuid is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the uuid afterwards.\n"
    "{context}\n"
    "What is the special magic uuid for {query_key} mentioned in the provided text? "
    "The special magic uuid for {query_key} mentioned in the provided text is"
)


def generate_uuid(rng: Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def measure_tokens_per_line(rng: Random, count_tokens, num_samples: int) -> float:
    total = 0
    for _ in range(num_samples):
        line = NEEDLE_TEMPLATE.format(key=generate_uuid(rng), value=generate_uuid(rng))
        total += count_tokens(line + "\n")
    return total / max(num_samples, 1)


def build_prompt(
    key_value_pairs: list[tuple[str, str]],
    target_key: str,
) -> str:
    context = "\n".join(
        NEEDLE_TEMPLATE.format(key=key, value=value) for key, value in key_value_pairs
    )
    return KEY_TO_VALUE_QUERY_TEMPLATE.format(context=context, query_key=target_key)


def generate_sample(
    *,
    context_length: int,
    depth_percent: int,
    sample_index: int,
    rng: Random,
    count_tokens,
    tokens_per_line: float,
    max_tokens: int,
) -> dict[str, Any]:
    dummy_key = generate_uuid(rng)
    generate_uuid(rng)
    template_overhead = count_tokens(
        KEY_TO_VALUE_QUERY_TEMPLATE.format(context="", query_key=dummy_key)
    )
    available_tokens = context_length - template_overhead - max_tokens
    estimated_lines = max(3, int(available_tokens / tokens_per_line))

    key_value_pairs = [(generate_uuid(rng), generate_uuid(rng)) for _ in range(estimated_lines)]
    target_index = int((depth_percent / 100) * max(len(key_value_pairs) - 1, 0))
    target_key, target_value = key_value_pairs[target_index]

    prompt = build_prompt(key_value_pairs, target_key)
    prompt_tokens = count_tokens(prompt)
    total_tokens = prompt_tokens + max_tokens

    remaining_adjustments = 50
    while total_tokens > context_length and len(key_value_pairs) > 3 and remaining_adjustments > 0:
        if len(key_value_pairs) - 1 != target_index:
            key_value_pairs.pop()
        else:
            key_value_pairs.pop(-2)
            target_index = len(key_value_pairs) - 1
        prompt = build_prompt(key_value_pairs, target_key)
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_tokens
        remaining_adjustments -= 1

    while total_tokens < context_length - tokens_per_line and remaining_adjustments > 0:
        key_value_pairs.append((generate_uuid(rng), generate_uuid(rng)))
        prompt = build_prompt(key_value_pairs, target_key)
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_tokens
        remaining_adjustments -= 1

    if total_tokens > context_length and len(key_value_pairs) > 3:
        if len(key_value_pairs) - 1 != target_index:
            key_value_pairs.pop()
        prompt = build_prompt(key_value_pairs, target_key)
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_tokens

    sample_id = f"niah_uuid_key_to_value_{context_length}_{depth_percent}_{rng.getrandbits(32)}"
    return {
        "id": sample_id,
        "context_length": context_length,
        "depth_percent": depth_percent,
        "sample_index": sample_index,
        "prompt": prompt,
        "expected": target_value,
        "target_key": target_key,
        "prompt_tokens": prompt_tokens,
        "budgeted_total_tokens": total_tokens,
        "num_haystack_lines": len(key_value_pairs),
    }


def grade_response(expected: str, response_text: str) -> tuple[bool, str]:
    text = response_text.strip()
    hit = expected.lower() in text.lower()
    matched_line = next(
        (line.strip() for line in text.splitlines() if expected.lower() in line.lower()),
        text.splitlines()[0].strip() if text else "",
    )
    return hit, matched_line


def validate_grid(context_lengths, depths, samples_per_cell) -> dict[str, Any]:
    for values, minimum, maximum in ((context_lengths, 1, None), (depths, 0, 100)):
        if (
            not values
            or len(set(values)) != len(values)
            or any(
                type(x) is not int or x < minimum or (maximum is not None and x > maximum)
                for x in values
            )
        ):
            raise ValueError(
                "Context lengths must be positive; depths must be in [0, 100]; no duplicates"
            )
    if type(samples_per_cell) is not int or samples_per_cell < 1:
        raise ValueError("samples-per-cell must be positive")
    return {
        "context_lengths": list(context_lengths),
        "depths": list(depths),
        "samples_per_cell": samples_per_cell,
    }


class NIAH:
    name = cache_name = "niah"

    def prepare(
        self,
        data_dir: Path,
        tokenizer: str,
        *,
        context_lengths=CONTEXT_LENGTHS,
        depths=DEPTHS,
        samples_per_cell: int = 10,
    ) -> None:
        grid = validate_grid(context_lengths, depths, samples_per_cell)
        path = Path(tokenizer)
        if path.is_dir():
            path /= "tokenizer.json"
        elif not path.is_file():
            path = Path(hf_hub_download(tokenizer, "tokenizer.json"))
        encoded = path.read_bytes()
        tokenizer_hash = hashlib.sha256(encoded).hexdigest()
        destination = data_dir / self.cache_name
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text())
            if manifest["grid"] != grid or manifest["tokenizer_sha256"] != tokenizer_hash:
                raise ValueError("Prepared NIAH grid or tokenizer differs; use another data-dir")
            self.load(data_dir)
            return
        encoder = Tokenizer.from_str(encoded.decode())
        encoder.no_truncation()
        encoder.no_padding()

        def count_tokens(text):
            return len(encoder.encode(text, add_special_tokens=False).ids)

        tokens_per_line = measure_tokens_per_line(Random(43), count_tokens, 20)
        rng = Random(42)
        data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="niah-prepare-", dir=data_dir) as temporary:
            staging = Path(temporary) / self.cache_name
            (staging / "prompts").mkdir(parents=True)
            rows = []
            for context_length in context_lengths:
                for depth in depths:
                    for index in range(samples_per_cell):
                        sample = generate_sample(
                            context_length=context_length,
                            depth_percent=depth,
                            sample_index=index,
                            rng=rng,
                            count_tokens=count_tokens,
                            tokens_per_line=tokens_per_line,
                            max_tokens=REFERENCE_OUTPUT_RESERVE,
                        )
                        if sample["budgeted_total_tokens"] > context_length:
                            raise ValueError(
                                f"NIAH prompt exceeds its context budget: {sample['id']}"
                            )
                        prompt = sample.pop("prompt").encode()
                        sample["prompt_sha256"] = hashlib.sha256(prompt).hexdigest()
                        (staging / "prompts" / f"{sample['prompt_sha256']}.txt").write_bytes(prompt)
                        rows.append(sample)
                print(f"Prepared context {context_length}: {len(rows)} cases", flush=True)
            write_cache(staging, rows, repository=REPOSITORY, revision=REVISION)
            manifest_path = staging / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            write_json(manifest_path, manifest | {"grid": grid, "tokenizer_sha256": tokenizer_hash})
            staging.rename(destination)

    def load(self, data_dir: Path) -> list[Example]:
        directory = data_dir / self.cache_name
        manifest = json.loads((directory / "manifest.json").read_text())
        grid = validate_grid(**manifest["grid"])
        count = len(grid["context_lengths"]) * len(grid["depths"]) * grid["samples_per_cell"]
        rows = read_cache(directory, repository=REPOSITORY, revision=REVISION, count=count)
        examples = []
        for row in rows:
            digest = row["prompt_sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid prompt checksum")
            path = directory / "prompts" / f"{digest}.txt"
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Prompt checksum mismatch: {row['id']}")
            examples.append(
                Example(row["id"], (), row["expected"], row | {"prompt_path": str(path)})
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
        prompt = Path(example.metadata["prompt_path"]).read_bytes()
        if hashlib.sha256(prompt).hexdigest() != example.metadata["prompt_sha256"]:
            raise ValueError(f"Prompt checksum mismatch: {example.id}")
        response = await provider.generate((Message("user", (Text(prompt.decode()),)),), generation)
        if response.stop_reason == "tool_calls" or response.message.tool_calls:
            raise ValueError("NIAH requested no tools but received a tool call")
        correct, answer = grade_response(example.target, response.text)
        return Evaluation(
            float(correct),
            answer,
            {
                key: example.metadata[key]
                for key in (
                    "context_length",
                    "depth_percent",
                    "sample_index",
                    "prompt_tokens",
                    "target_key",
                )
            }
            | {"stop_reason": response.stop_reason},
        )


def write_grid(output: Path) -> None:
    manifest = json.loads((output / "run.json").read_text())
    counts = Counter(tuple(map(int, case_id.split("_")[-3:-1])) for case_id in manifest["examples"])
    cells = {
        cell: {
            "context_length": cell[0],
            "depth_percent": cell[1],
            "expected": count,
            "completed": 0,
            "correct": 0,
        }
        for cell, count in counts.items()
    }
    for case_id in manifest["examples"]:
        path = output / "samples" / hashlib.sha256(case_id.encode()).hexdigest() / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            details = result["details"]
            cell = cells[details["context_length"], details["depth_percent"]]
            cell["completed"] += 1
            cell["correct"] += result["score"]
    for cell in cells.values():
        score = cell["correct"] / cell["completed"] if cell["completed"] else None
        cell["accuracy"] = score if cell["completed"] == cell["expected"] else None
        cell["provisional_accuracy"] = score
    write_json(output / "grid.json", list(cells.values()))


def main() -> None:
    root, prepare, run = parser("niah")
    prepare.add_argument(
        "--tokenizer", required=True, help="HF model ID, tokenizer.json, or its directory"
    )
    prepare.add_argument("--context-lengths", nargs="+", type=positive, default=CONTEXT_LENGTHS)
    prepare.add_argument("--depths", nargs="+", type=int, default=DEPTHS)
    prepare.add_argument("--samples-per-cell", type=positive, default=10)
    run.set_defaults(
        max_output_tokens=4096,
        temperature=0.0,
        top_p=None,
        reasoning_effort="none",
        concurrency=4,
    )
    args = root.parse_args()
    benchmark = NIAH()
    if args.command == "prepare":
        benchmark.prepare(
            args.data_dir.resolve(),
            args.tokenizer,
            context_lengths=args.context_lengths,
            depths=args.depths,
            samples_per_cell=args.samples_per_cell,
        )
    else:
        asyncio.run(evaluate(benchmark, args))
        write_grid(args.output.resolve())
