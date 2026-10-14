"""Prepare pinned datasets and run the same workload through any supported API."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path
from typing import get_args
from urllib.parse import urlsplit

from .code_agent import CodeBenchmark
from .providers import APIProvider
from .runner import run
from .types import Api, Benchmark, Generation

APIS = get_args(Api)


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def optional_float(value: str) -> float | None:
    if value == "none":
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number or 'none'")
    return result


def add_api_arguments(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    parser.add_argument(f"--{prefix}api", choices=APIS, required=not prefix)
    parser.add_argument(f"--{prefix}model", required=not prefix)
    parser.add_argument(
        f"--{prefix}base-url", help="API base URL, including /v1 for OpenAI endpoints"
    )
    parser.add_argument(
        f"--{prefix}api-key-env", help="environment variable containing the API key"
    )


def parser(
    benchmark: str,
) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
    root = argparse.ArgumentParser(description=f"Prepare or evaluate {benchmark}.")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download or install a pinned dataset")
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    evaluate = commands.add_parser("run", help="run or resume an evaluation")
    add_api_arguments(evaluate)
    evaluate.add_argument("--data-dir", type=Path, default=Path("data"))
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--concurrency", type=positive)
    evaluate.add_argument("--max-output-tokens", type=positive)
    evaluate.add_argument("--temperature", type=optional_float, default=1.0, metavar="FLOAT|none")
    evaluate.add_argument("--top-p", type=optional_float, default=1.0, metavar="FLOAT|none")
    evaluate.add_argument(
        "--reasoning-effort",
        choices=("default", "none", "low", "medium", "high", "max"),
        default="high",
    )
    evaluate.add_argument(
        "--limit", type=positive, help="first N cases for a partial-dataset smoke test"
    )
    evaluate.add_argument(
        "--resume", action="store_true", help="resume identical data and run settings"
    )
    return root, prepare, evaluate


def endpoint(api: str, base_url: str | None) -> str:
    if api == "anthropic":
        value = base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    else:
        value = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Base URL must be HTTP(S), with no embedded credentials, query or fragment"
        )
    return value.rstrip("/")


def provider_configuration(args: argparse.Namespace) -> tuple[dict, str]:
    base_url = endpoint(args.api, args.base_url)
    key_name = args.api_key_env or (
        "ANTHROPIC_API_KEY" if args.api == "anthropic" else "OPENAI_API_KEY"
    )
    if not (api_key := os.environ.get(key_name)):
        raise ValueError(
            f"Set {key_name}; use an arbitrary value for endpoints without authentication"
        )
    configuration = {
        "api": args.api,
        "model": args.model,
        "base_url": base_url,
        "api_key_env": key_name,
    }
    return configuration, api_key


async def evaluate(
    benchmark: Benchmark,
    args: argparse.Namespace,
    *,
    judge_args: argparse.Namespace | None = None,
    judge_generation: Generation | None = None,
) -> None:
    examples = benchmark.load(args.data_dir.resolve())
    dataset_size = len(examples)
    if args.limit is not None:
        examples = examples[: args.limit]
    code = isinstance(benchmark, CodeBenchmark)
    generation = Generation(
        max_tokens=args.max_output_tokens or (262144 if code else 131072),
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=None if args.reasoning_effort == "default" else args.reasoning_effort,
    )
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1], or 'none'")
    candidate_config, api_key = provider_configuration(args)
    configuration = {
        **candidate_config,
        "dataset": json.loads((args.data_dir / benchmark.cache_name / "manifest.json").read_text()),
        "dataset_size": dataset_size,
        "selected_size": len(examples),
        "full_dataset": len(examples) == dataset_size,
    }
    if code and benchmark.allow_network:
        configuration["allow_network"] = True
    async with AsyncExitStack() as stack:
        provider = await stack.enter_async_context(
            APIProvider(
                args.api,
                args.model,
                base_url=candidate_config["base_url"],
                api_key=api_key,
            )
        )
        judge = None
        judge_config = candidate_config
        if judge_args is not None:
            if judge_generation is None:
                raise ValueError(
                    "Judge generation settings must be recorded with the judge provider"
                )
            judge_config, judge_key = provider_configuration(judge_args)
            judge = await stack.enter_async_context(
                APIProvider(
                    judge_args.api,
                    judge_args.model,
                    base_url=judge_config["base_url"],
                    api_key=judge_key,
                )
            )
        if judge_generation is not None:
            configuration["judge"] = {
                "mode": "self" if judge_args is None else "external",
                **judge_config,
                "generation": asdict(judge_generation),
            }
        summary = await run(
            benchmark,
            examples,
            provider,
            generation,
            args.output.resolve(),
            configuration=configuration,
            concurrency=args.concurrency or (32 if code else 64),
            resume=args.resume,
            judge=judge,
        )
    print(json.dumps(summary, indent=2))
