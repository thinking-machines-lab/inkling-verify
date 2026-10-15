"""Prepare pinned datasets and run the same workload through any supported API."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args
from urllib.parse import urlsplit

from .providers import REQUEST_TIMEOUT_SECONDS, APIProvider
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


def add_generation_arguments(
    parser: argparse.ArgumentParser, *, prefix: str = "", defaults: Generation | None = None
) -> None:
    parser.add_argument(
        f"--{prefix}max-output-tokens",
        type=positive,
        default=defaults.max_tokens if defaults else None,
        help="output token budget, cumulative across tool turns",
    )
    defaults = defaults or Generation(262144)
    for name in ("temperature", "top_p"):
        parser.add_argument(
            f"--{prefix}{name.replace('_', '-')}",
            type=optional_float,
            default=getattr(defaults, name),
            metavar="FLOAT|none",
        )
    parser.add_argument(
        f"--{prefix}reasoning-effort",
        choices=("default", "none", "low", "medium", "high", "max"),
        default=defaults.reasoning_effort,
    )


def generation_settings(args: argparse.Namespace, *, prefix: str = "") -> Generation:
    values = {
        name: getattr(args, prefix.replace("-", "_") + name)
        for name in ("max_output_tokens", "temperature", "top_p", "reasoning_effort")
    }
    if values["top_p"] is not None and not 0 < values["top_p"] <= 1:
        raise ValueError(f"{prefix}top-p must be in (0, 1], or 'none'")
    return Generation(
        max_tokens=values["max_output_tokens"] or 262144,
        temperature=values["temperature"],
        top_p=values["top_p"],
        reasoning_effort=None
        if values["reasoning_effort"] == "default"
        else values["reasoning_effort"],
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
    evaluate.add_argument(
        "--samples-per-problem",
        type=positive,
        default=1,
        help="independent attempts per problem, averaged in accuracy (default: %(default)s)",
    )
    add_generation_arguments(evaluate)
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
    benchmark_configuration: dict[str, Any] | None = None,
    judge_args: argparse.Namespace | None = None,
    judge_generation: Generation | None = None,
) -> None:
    examples = benchmark.load(args.data_dir.resolve())
    dataset_size = len(examples)
    if args.limit is not None:
        examples = examples[: args.limit]
    generation = generation_settings(args)
    candidate_config, api_key = provider_configuration(args)
    configuration = {
        **(benchmark_configuration or {}),
        **candidate_config,
        "dataset": json.loads((args.data_dir / benchmark.cache_name / "manifest.json").read_text()),
        "dataset_size": dataset_size,
        "selected_size": len(examples),
        "full_dataset": len(examples) == dataset_size,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    }
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
            concurrency=args.concurrency or 64,
            samples_per_problem=args.samples_per_problem,
            resume=args.resume,
            judge=judge,
        )
    print(json.dumps(summary, indent=2))
