"""Text-only Humanity's Last Exam with self-judging or an independent judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema

from inkling_vendor_verifier.benchmarks.data import read_cache, required_text, write_cache
from inkling_vendor_verifier.cli import (
    add_api_arguments,
    evaluate,
    optional_float,
    parser,
    positive,
)
from inkling_vendor_verifier.types import Evaluation, Example, Generation, Message, Provider, Text

REPOSITORY = "cais/hle"
REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
EXPECTED_COUNT = 2158
DATASET_NAME = "hle-self-judge"
SYSTEM_PROMPTS = {
    "exactMatch": """Your response should be in the following format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}""",
    "multipleChoice": """Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}""",
}

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""  # noqa: E501

JUDGE_SCHEMA = {
    "additionalProperties": False,
    "properties": {
        "extracted_final_answer": {"title": "Extracted Final Answer", "type": "string"},
        "reasoning": {"title": "Reasoning", "type": "string"},
        "correct": {"enum": ["yes", "no"], "title": "Correct", "type": "string"},
        "confidence": {"title": "Confidence", "type": "integer"},
        "strict": {"const": True, "title": "Strict", "type": "boolean"},
    },
    "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
    "title": "ExtractedAnswer",
    "type": "object",
}
SCHEMA_SYSTEM_INSTRUCTION = (
    "Return exactly one JSON object matching the response schema below. Include every "
    'required field, including "strict": true, then close the object. Do not include '
    "markdown or any text outside the JSON object.\n"
    + json.dumps(
        {"name": "ExtractedAnswer", "schema": JUDGE_SCHEMA, "strict": True}, separators=(",", ":")
    )
)
JUDGE_GENERATION = Generation(max_tokens=32768, temperature=0.0, top_p=1.0, reasoning_effort="none")


def parse_verdict(text: str) -> dict[str, Any]:
    try:
        verdict = json.loads(text)
        jsonschema.validate(verdict, JUDGE_SCHEMA)
    except (json.JSONDecodeError, jsonschema.ValidationError) as error:
        raise ValueError(f"HLE judge returned an invalid ExtractedAnswer: {error}") from error
    return verdict


def _has_image(row: dict[str, Any]) -> bool:
    image = row["image"]
    if image is not None and not isinstance(image, str):
        raise ValueError("HLE image must be a string or null")
    return bool(image)


def _prepare_row(row: dict[str, Any]) -> dict[str, Any]:
    source_id = required_text(row, "id")
    answer_type = required_text(row, "answer_type")
    if answer_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown HLE answer type: {answer_type}")
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"hle::{source_id}")),
        "source_id": source_id,
        "question": required_text(row, "question"),
        "answer": required_text(row, "answer"),
        "answer_type": answer_type,
        "category": required_text(row, "category"),
    }


class HLE:
    name = "hle"
    cache_name = DATASET_NAME

    def __init__(self, judge_generation: Generation | None = None) -> None:
        self.judge_generation = judge_generation

    def prepare(self, data_dir: Path) -> None:
        from datasets import load_dataset

        destination = data_dir / DATASET_NAME
        if destination.exists():
            self.load(data_dir)
            return
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(REPOSITORY, split="test", revision=REVISION)
        rows = [_prepare_row(row) for row in dataset if not _has_image(row)]
        if len(rows) != EXPECTED_COUNT or len({row["id"] for row in rows}) != EXPECTED_COUNT:
            raise ValueError(f"Expected {EXPECTED_COUNT} distinct text-only HLE examples")
        with tempfile.TemporaryDirectory(prefix="hle-prepare-", dir=data_dir) as temporary:
            staging = Path(temporary) / DATASET_NAME
            staging.mkdir()
            write_cache(staging, rows, repository=REPOSITORY, revision=REVISION)
            staging.rename(destination)

    def load(self, data_dir: Path) -> list[Example]:
        rows = read_cache(
            data_dir / DATASET_NAME, repository=REPOSITORY, revision=REVISION, count=EXPECTED_COUNT
        )
        examples = []
        for row in rows:
            if row["answer_type"] not in SYSTEM_PROMPTS:
                raise ValueError(f"Unknown HLE answer type: {row['answer_type']}")
            question = required_text(row, "question")
            examples.append(
                Example(
                    id=row["id"],
                    messages=(
                        Message("system", (Text(SYSTEM_PROMPTS[row["answer_type"]]),)),
                        Message("user", (Text(question),)),
                    ),
                    target=required_text(row, "answer"),
                    metadata={
                        "question": question,
                        "source_id": row["source_id"],
                        "category": row["category"],
                        "answer_type": row["answer_type"],
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
        candidate = await provider.generate(example.messages, generation)
        if candidate.stop_reason == "tool_calls" or candidate.message.tool_calls:
            raise ValueError("HLE requested no tools but received a tool call")
        prompt = JUDGE_PROMPT.format(
            question=example.metadata["question"],
            correct_answer=example.target,
            response=candidate.text,
        )
        judge_generation = self.judge_generation or replace(
            JUDGE_GENERATION,
            temperature=None if judge is None and generation.temperature is None else 0.0,
            top_p=None if judge is None and generation.top_p is None else 1.0,
        )
        judgment = await (judge if judge is not None else provider).generate(
            (
                Message("system", (Text(SCHEMA_SYSTEM_INSTRUCTION),)),
                Message("user", (Text(prompt),)),
            ),
            judge_generation,
            response_schema=JUDGE_SCHEMA,
        )
        if judgment.stop_reason != "stop" or judgment.message.tool_calls:
            raise ValueError(f"HLE judge did not complete a verdict: {judgment.stop_reason}")
        verdict = parse_verdict(judgment.text)
        return Evaluation(
            float(verdict["correct"] == "yes"),
            verdict["extracted_final_answer"],
            {
                "verdict": verdict,
                "candidate_stop_reason": candidate.stop_reason,
                "judge_temperature": judge_generation.temperature,
                "judge_top_p": judge_generation.top_p,
            },
        )


def judge_settings(args: argparse.Namespace) -> tuple[argparse.Namespace | None, Generation]:
    fields = ("api", "model", "base_url", "api_key_env", "extra_body")
    separate = any(getattr(args, f"judge_{field}") is not None for field in fields)
    judge_args = None
    if separate:
        api = args.judge_api or args.api
        same_api = api == args.api
        judge_args = argparse.Namespace(
            api=api,
            model=args.judge_model or args.model,
            base_url=args.judge_base_url or (args.base_url if same_api else None),
            api_key_env=args.judge_api_key_env or (args.api_key_env if same_api else None),
            extra_body=args.judge_extra_body,
        )
    generation = Generation(
        max_tokens=args.judge_max_output_tokens,
        temperature=getattr(
            args, "judge_temperature", None if not separate and args.temperature is None else 0.0
        ),
        top_p=getattr(args, "judge_top_p", None if not separate and args.top_p is None else 1.0),
        reasoning_effort=(
            None if args.judge_reasoning_effort == "default" else args.judge_reasoning_effort
        ),
    )
    if generation.top_p is not None and not 0 < generation.top_p <= 1:
        raise ValueError("judge-top-p must be in (0, 1], or 'none'")
    return judge_args, generation


def main() -> None:
    root, _, run_parser = parser(HLE.name)
    add_api_arguments(run_parser, prefix="judge-")
    run_parser.add_argument("--judge-max-output-tokens", type=positive, default=32768)
    run_parser.add_argument(
        "--judge-temperature", type=optional_float, default=argparse.SUPPRESS, metavar="FLOAT|none"
    )
    run_parser.add_argument(
        "--judge-top-p", type=optional_float, default=argparse.SUPPRESS, metavar="FLOAT|none"
    )
    run_parser.add_argument(
        "--judge-reasoning-effort",
        choices=("default", "none", "low", "medium", "high"),
        default="none",
    )
    args = root.parse_args()
    if args.command == "prepare":
        HLE().prepare(args.data_dir.resolve())
        print(f"Prepared {HLE.name} in {args.data_dir}")
    else:
        judge_args, judge_generation = judge_settings(args)
        asyncio.run(
            evaluate(
                HLE(judge_generation),
                args,
                judge_args=judge_args,
                judge_generation=judge_generation,
            )
        )
