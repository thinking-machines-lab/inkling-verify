import json
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from dataclasses import asdict

import pytest

from inkling_vendor_verifier import cli
from inkling_vendor_verifier.benchmarks import deepswe, niah
from inkling_vendor_verifier.benchmarks import hle as hle
from inkling_vendor_verifier.benchmarks import mmmu_pro as mmmu
from inkling_vendor_verifier.benchmarks import swe_bench as swe
from inkling_vendor_verifier.types import Example, Generation

BENCHMARKS = (
    (niah, niah.NIAH, None),
    (hle, hle.HLE, None),
    (mmmu, mmmu.MMMUPro, None),
    (mmmu, mmmu.MMMUPro, "standard-4"),
    (mmmu, mmmu.MMMUPro, "standard-10"),
    (swe, swe.SWEBench, None),
    (swe, swe.SWEBench, "verified"),
    (swe, swe.SWEBench, "pro"),
    (swe, swe.SWEBench, "multilingual"),
    (deepswe, deepswe.DeepSWE, None),
)


@pytest.mark.parametrize("module,benchmark_type,subset", BENCHMARKS)
@pytest.mark.parametrize("command", ["prepare", "run"])
def test_benchmark_entrypoint_dispatches_prepare_and_run(
    monkeypatch, tmp_path, module, benchmark_type, subset, command
):
    captured = {}

    def prepare(*args, **kwargs):
        captured["prepare"] = (args, kwargs)

    async def evaluate(benchmark, args, **kwargs):
        captured["run"] = (benchmark, args)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "evaluate", evaluate)
    if module is niah:
        monkeypatch.setattr(niah, "write_grid", lambda output: None)
    monkeypatch.setattr(
        benchmark_type,
        "prepare",
        lambda self, data_dir, *args, **kwargs: prepare(self.name, data_dir, *args, **kwargs),
    )
    expected_name = (benchmark_type() if subset is None else benchmark_type(subset)).name
    argv = ["benchmark.py", command, "--data-dir", "prepared"]
    if subset is not None:
        argv.extend(["--subset", subset])
    if command == "run":
        argv.extend(["--api", "anthropic", "--model", "candidate", "--output", "results"])
    elif module is niah:
        argv.extend(["--tokenizer", "model"])
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    if command == "prepare":
        expected = (expected_name, tmp_path / "prepared")
        options = {}
        if module is niah:
            expected += ("model",)
            options = {
                "context_lengths": (4000, 16000, 64000, 256000, 1024000),
                "depths": tuple(range(0, 101, 10)),
                "samples_per_cell": 10,
            }
        assert captured == {"prepare": (expected, options)}
    else:
        assert set(captured) == {"run"}
        benchmark, args = captured["run"]
        assert isinstance(benchmark, benchmark_type)
        assert benchmark.name == expected_name
        assert args.api == "anthropic"
        assert args.model == "candidate"
        assert args.data_dir.resolve() == tmp_path / "prepared"
        assert args.output.resolve() == tmp_path / "results"
        assert args.limit is None
        if module is hle:
            assert args.reasoning_effort == "max"
        if module is niah:
            assert (
                args.max_output_tokens,
                args.temperature,
                args.top_p,
                args.reasoning_effort,
                args.concurrency,
            ) == (128, 0, None, "none", 4)


@pytest.mark.parametrize(
    "module,arguments,message",
    [
        (swe, ["--bundle", "tasks.tar.gz"], "unrecognized arguments: --bundle"),
        (deepswe, ["--bundle", "tasks.tar.gz"], "unrecognized arguments: --bundle"),
        (hle, ["--bundle", "tasks.tar.gz"], "unrecognized arguments: --bundle"),
        (mmmu, ["--bundle", "tasks.tar.gz"], "unrecognized arguments: --bundle"),
        (
            deepswe,
            ["--subset", "deepswe"],
            "unrecognized arguments: --subset",
        ),
        (swe, ["--subset", "deepswe"], "invalid choice: 'deepswe'"),
    ],
)
def test_prepare_arguments_match_benchmark(monkeypatch, capsys, module, arguments, message):
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "prepare", *arguments])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--concurrency", "0"),
        ("--limit", "-1"),
        ("--max-output-tokens", "0"),
        ("--temperature", "nan"),
        ("--top-p", "inf"),
    ],
)
def test_parser_rejects_invalid_numeric_settings(flag, value):
    root, _, _ = cli.parser("hle")
    with pytest.raises(SystemExit):
        root.parse_args(
            [
                "run",
                "--api",
                "anthropic",
                "--model",
                "m",
                "--output",
                "r",
                flag,
                value,
            ]
        )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host/v1",
        "https://user:password@host/v1",
        "https://secret@host/v1",
        "https://host/v1?api_key=secret",
        "https://host/v1#secret",
        "/v1",
    ],
)
def test_endpoint_rejects_embedded_credentials_and_non_http_urls(url):
    with pytest.raises(ValueError, match="Base URL"):
        cli.endpoint("openai-completions", url)


def test_endpoint_environment_defaults_are_explicit_and_protocol_specific(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.example/v1/")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://anthropic.example/")
    assert cli.endpoint("openai-responses", None) == "http://openai.example/v1"
    assert cli.endpoint("anthropic", None) == "http://anthropic.example"
    assert cli.endpoint("anthropic", "http://override/") == "http://override"
    monkeypatch.delenv("OPENAI_BASE_URL")
    monkeypatch.delenv("ANTHROPIC_BASE_URL")
    assert cli.endpoint("openai-completions", None) == "https://api.openai.com/v1"
    assert cli.endpoint("anthropic", None) == "https://api.anthropic.com"


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    captured = {"providers": [], "closed": []}

    @asynccontextmanager
    async def provider(api, model, **kwargs):
        captured["providers"].append((api, model, kwargs))
        try:
            yield model
        finally:
            captured["closed"].append(model)

    async def run(benchmark, examples, provider, generation, output_dir, **kwargs):
        captured["run"] = (examples, provider, generation, output_dir, kwargs)
        return {"status": "completed", "full_dataset": kwargs["configuration"]["full_dataset"]}

    monkeypatch.setattr(cli, "APIProvider", provider)
    monkeypatch.setattr(cli, "run", run)

    def arguments(benchmark, *options):
        monkeypatch.setattr(
            benchmark, "load", lambda data_dir: [Example("one", (), "a"), Example("two", (), "b")]
        )
        directory = tmp_path / "data" / benchmark.cache_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text('{"revision": "test", "examples_sha256": "abc"}')
        root, _, _ = cli.parser(benchmark.name)
        return root.parse_args(
            [
                "run",
                "--api",
                "anthropic",
                "--model",
                "candidate",
                "--data-dir",
                str(tmp_path / "data"),
                "--output",
                str(tmp_path / "results"),
                *options,
            ]
        )

    return arguments, captured


@pytest.mark.parametrize("api", cli.APIS)
@pytest.mark.parametrize(
    "benchmark,concurrency,max_tokens",
    [
        (hle.HLE(), 64, 131072),
        (mmmu.MMMUPro(), 64, 131072),
        (swe.SWEBench(), 32, 262144),
        (deepswe.DeepSWE(), 32, 262144),
    ],
)
async def test_cli_records_dataset_and_endpoint_without_secrets_and_marks_smoke(
    monkeypatch, capsys, evaluation, api, benchmark, concurrency, max_tokens
):
    arguments, captured = evaluation
    prefix = "ANTHROPIC" if api == "anthropic" else "OPENAI"
    monkeypatch.setenv(f"{prefix}_API_KEY", "test-secret")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "http://local.example/v1/")
    monkeypatch.setenv(f"{prefix}_CUSTOM_HEADERS", "X-Vendor-Token: private-header-secret")
    args = arguments(
        benchmark,
        "--api",
        api,
        "--limit",
        "1",
        "--temperature",
        "none",
        "--top-p",
        "none",
        "--reasoning-effort",
        "default",
    )
    await cli.evaluate(benchmark, args)
    assert json.loads(capsys.readouterr().out) == {"status": "completed", "full_dataset": False}
    provider_api, model, provider_kwargs = captured["providers"][0]
    assert (provider_api, model) == (api, "candidate")
    assert provider_kwargs["api_key"] == "test-secret"
    assert provider_kwargs["base_url"] == "http://local.example/v1"
    examples, _, generation, _, kwargs = captured["run"]
    assert len(examples) == 1
    assert generation == Generation(max_tokens, temperature=None, top_p=None, reasoning_effort=None)
    configuration = kwargs["configuration"]
    assert configuration["dataset"] == {"revision": "test", "examples_sha256": "abc"}
    assert configuration["full_dataset"] is False
    assert configuration["dataset_size"] == 2
    assert configuration["base_url"] == "http://local.example/v1"
    assert "secret" not in json.dumps(configuration)
    assert kwargs["concurrency"] == concurrency
    assert captured["closed"] == ["candidate"]
    args.concurrency, args.max_output_tokens = 2, 123
    await cli.evaluate(benchmark, args)
    _, _, generation, _, kwargs = captured["run"]
    assert kwargs["concurrency"] == 2
    assert generation.max_tokens == 123


async def test_missing_api_key_fails_before_provider_creation(monkeypatch, evaluation):
    arguments, captured = evaluation
    benchmark = hle.HLE()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        await cli.evaluate(benchmark, arguments(benchmark))
    assert not captured["providers"]


@pytest.mark.parametrize(
    "candidate_api,arguments,expected_api,expected_url,expected_key_env,expected_generation",
    [
        (
            "openai-completions",
            [],
            None,
            None,
            None,
            Generation(32768, temperature=None, top_p=None, reasoning_effort="none"),
        ),
        (
            "openai-completions",
            ["--judge-model", "judge"],
            "openai-completions",
            "https://candidate/v1",
            "CANDIDATE_KEY",
            Generation(32768, temperature=0, reasoning_effort="none"),
        ),
        (
            "openai-completions",
            ["--judge-api", "anthropic", "--judge-model", "judge"],
            "anthropic",
            None,
            None,
            Generation(32768, temperature=0, reasoning_effort="none"),
        ),
        (
            "openai-completions",
            [
                "--judge-base-url",
                "https://judge/v1",
                "--judge-api-key-env",
                "JUDGE_KEY",
                "--judge-temperature",
                "none",
            ],
            "openai-completions",
            "https://judge/v1",
            "JUDGE_KEY",
            Generation(32768, temperature=None, reasoning_effort="none"),
        ),
        (
            "anthropic",
            [
                "--judge-max-output-tokens",
                "17",
                "--judge-temperature",
                "0.25",
                "--judge-top-p",
                "none",
                "--judge-reasoning-effort",
                "default",
            ],
            None,
            None,
            None,
            Generation(17, temperature=0.25, top_p=None, reasoning_effort=None),
        ),
    ],
)
def test_hle_cli_resolves_judge_independently(
    monkeypatch,
    candidate_api,
    arguments,
    expected_api,
    expected_url,
    expected_key_env,
    expected_generation,
):
    captured = {}

    async def evaluate(benchmark, args, **kwargs):
        captured.update(kwargs)
        assert benchmark.judge_generation == kwargs["judge_generation"]

    monkeypatch.setattr(hle, "evaluate", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hle.py",
            "run",
            "--api",
            candidate_api,
            "--model",
            "candidate",
            "--base-url",
            "https://candidate/v1",
            "--api-key-env",
            "CANDIDATE_KEY",
            "--temperature",
            "none",
            "--top-p",
            "none",
            "--output",
            "results",
            *arguments,
        ],
    )
    hle.main()
    judge = captured["judge_args"]
    if expected_api is None:
        assert judge is None
    else:
        assert judge.api == expected_api
        assert judge.base_url == expected_url
        assert judge.api_key_env == expected_key_env
        assert judge.model == ("judge" if "--judge-model" in arguments else "candidate")
    assert captured["judge_generation"] == expected_generation


@pytest.mark.parametrize("top_p", ["0", "1.1"])
def test_hle_cli_rejects_invalid_judge_top_p(monkeypatch, top_p):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hle.py",
            "run",
            "--api",
            "anthropic",
            "--model",
            "candidate",
            "--output",
            "results",
            "--judge-top-p",
            top_p,
        ],
    )
    with pytest.raises(ValueError, match="judge-top-p"):
        hle.main()


@pytest.mark.parametrize("external", [False, True])
async def test_cli_records_judge_configuration_and_closes_both_providers(
    monkeypatch, evaluation, external
):
    arguments, captured = evaluation
    benchmark = hle.HLE()
    monkeypatch.setenv("CANDIDATE_KEY", "candidate-secret")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    args = arguments(
        benchmark,
        "--api",
        "openai-responses",
        "--base-url",
        "https://candidate/v1",
        "--api-key-env",
        "CANDIDATE_KEY",
    )
    judge_args = (
        Namespace(
            api="anthropic",
            model="judge",
            base_url="https://judge",
            api_key_env="JUDGE_KEY",
        )
        if external
        else None
    )
    judge_generation = Generation(123, temperature=None, top_p=None, reasoning_effort=None)
    await cli.evaluate(benchmark, args, judge_args=judge_args, judge_generation=judge_generation)
    _, provider, _, _, kwargs = captured["run"]
    assert provider == "candidate"
    assert kwargs["judge"] == ("judge" if external else None)
    config = kwargs["configuration"]["judge"]
    assert config["generation"] == asdict(judge_generation)
    assert config["mode"] == ("external" if external else "self")
    assert config["model"] == ("judge" if external else "candidate")
    assert config["api"] == ("anthropic" if external else "openai-responses")
    assert config["base_url"] == ("https://judge" if external else "https://candidate/v1")
    assert "secret" not in json.dumps(kwargs["configuration"])
    assert captured["providers"][0][2]["api_key"] == "candidate-secret"
    if external:
        assert captured["providers"][1][2]["api_key"] == "judge-secret"
    assert captured["closed"] == (["judge", "candidate"] if external else ["candidate"])
