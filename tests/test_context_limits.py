import copy
import json

import httpx
import pytest
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import BadRequestError as OpenAIBadRequestError

from inkling_vendor_verifier import code_agent
from inkling_vendor_verifier.benchmarks.deepswe import DeepSWE
from inkling_vendor_verifier.context_limits import OUTPUT_TOKEN_FIELDS, ContextExhausted
from inkling_vendor_verifier.providers import REQUEST_TIMEOUT_SECONDS
from inkling_vendor_verifier.runner import RecordingProvider
from inkling_vendor_verifier.sandbox import CommandResult
from inkling_vendor_verifier.tool_loop import run_tools
from inkling_vendor_verifier.types import (
    Example,
    Generation,
    Message,
    Response,
    Text,
    Tool,
    Usage,
)

APIS = tuple(OUTPUT_TOKEN_FIELDS)
GENERATION = Generation(1_000_000)
MESSAGES = (Message("user", (Text("Keep the entire conversation."),)),)
TOOL = Tool("bash", "Run a command", {"type": "object"})


def total_message(input_tokens=100_000, requested=1_000_000, context=1_048_576):
    return (
        f"Requested token count exceeds the model's maximum context length of {context} tokens. "
        f"You requested a total of {input_tokens + requested} tokens: {input_tokens} tokens "
        f"from the input messages and {requested} tokens for the completion. Please reduce "
        "the number of tokens in the input messages or the completion to fit within the limit."
    )


def rejection(api, message):
    body = {"message": message, "type": "invalid_request_error"}
    return httpx.Response(400, json=body if api == "openai-completions" else {"error": body})


@pytest.mark.parametrize("api", APIS)
async def test_context_retry_changes_only_output_and_records_actual_request(
    sdk_provider, response_body, api
):
    provider, requests = sdk_provider(
        api,
        [response_body(api, tools=("pwd",)), rejection(api, total_message()), response_body(api)],
    )
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    async with provider:
        first = await provider.generate(MESSAGES, GENERATION, tools=(TOOL,))
        messages = (
            *MESSAGES,
            first.message,
            Message("tool", (Text("/repo"),), tool_call_id="call_1"),
        )
        original = copy.deepcopy((messages, TOOL, schema))
        response = await provider.generate(
            messages, GENERATION, tools=(TOOL,), response_schema=schema
        )
    bodies = [json.loads(request.content) for request in requests]
    field = OUTPUT_TOKEN_FIELDS[api]
    assert len(bodies) == 3
    assert bodies[1][field] == 1_000_000
    assert bodies[1] | {field: 948_575} == bodies[2] == response.request
    assert original == (messages, TOOL, schema)
    assert GENERATION.max_tokens == 1_000_000
    assert response.usage.output_tokens == 7


@pytest.mark.parametrize("api", APIS)
async def test_exact_million_and_twelve_hour_request_timeout(sdk_provider, response_body, api):
    provider, requests = sdk_provider(api, [response_body(api)])
    async with provider:
        await provider.generate(MESSAGES, GENERATION)
    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["read"] == REQUEST_TIMEOUT_SECONDS == 43_200
    assert json.loads(requests[0].content)[OUTPUT_TOKEN_FIELDS[api]] == 1_000_000


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("input_tokens", [1_048_575, 1_048_576, 1_048_600])
async def test_exhausted_context_has_no_retry(sdk_provider, api, input_tokens):
    message = (
        total_message(input_tokens=input_tokens)
        if input_tokens < 1_048_576
        else (
            f"The input ({input_tokens} tokens) is longer than the "
            "model's context length (1048576 tokens)."
        )
    )
    provider, requests = sdk_provider(api, [rejection(api, message)])
    async with provider:
        with pytest.raises(ContextExhausted) as error:
            await provider.generate(MESSAGES, GENERATION)
    assert error.value.input_tokens == input_tokens
    assert error.value.context_limit == 1_048_576
    assert len(requests) == 1


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "message",
    [
        "temperature unsupported",
        "prefix " + total_message(),
        total_message(requested=999999),
        total_message().replace("1100000 tokens:", "1100001 tokens:"),
        total_message(input_tokens=1),
        "The input (12 tokens) is longer than the model's context length (1048576 tokens).",
    ],
)
async def test_unrelated_or_inconsistent_errors_are_not_retried(sdk_provider, api, message):
    provider, requests = sdk_provider(api, [rejection(api, message)])
    error_type = AnthropicBadRequestError if api == "anthropic" else OpenAIBadRequestError
    async with provider:
        with pytest.raises(error_type):
            await provider.generate(MESSAGES, GENERATION)
    assert len(requests) == 1


@pytest.mark.parametrize("body", ["invalid", {"error": "invalid"}, {"message": 12}])
async def test_unknown_error_body_is_not_retried(sdk_provider, body):
    provider, requests = sdk_provider("openai-completions", [httpx.Response(400, json=body)])
    async with provider:
        with pytest.raises(OpenAIBadRequestError):
            await provider.generate(MESSAGES, GENERATION)
    assert len(requests) == 1


@pytest.mark.parametrize("api", APIS)
async def test_context_retry_is_bounded(sdk_provider, api):
    provider, requests = sdk_provider(
        api,
        [
            rejection(api, total_message()),
            rejection(api, total_message(input_tokens=200_000, requested=948_575)),
        ],
    )
    error_type = AnthropicBadRequestError if api == "anthropic" else OpenAIBadRequestError
    async with provider:
        with pytest.raises(error_type):
            await provider.generate(MESSAGES, GENERATION)
    assert len(requests) == 2


class Sandbox:
    config = {"agent": {"timeout_sec": 43200}}

    def __init__(self):
        self.commands = []
        self.graded = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, command, **kwargs):
        self.commands.append(command)
        return CommandResult("patched", "", 0, 0.01)

    async def grade(self):
        self.graded = True
        return 1.0, {"verifier_exit_code": 0}


@pytest.mark.parametrize(
    "api,termination",
    [(api, termination) for api in APIS for termination in ("rejection", "clamped")]
    + [("anthropic", "explicit")],
)
async def test_deepswe_context_limit_grades_existing_patch_and_keeps_recorded_usage(
    sdk_provider, response_body, monkeypatch, tmp_path, api, termination
):
    sandbox = Sandbox()
    monkeypatch.setattr(code_agent, "DockerSandbox", lambda *args, **kwargs: sandbox)
    responses = [response_body(api, "prior answer", tools=("fix",))]
    if termination == "clamped":
        responses.extend(
            [
                rejection(api, total_message(requested=999993)),
                response_body(api, "final answer", length=True),
            ]
        )
    elif termination == "explicit":
        response = response_body(api, "final answer", length=True)
        response["stop_reason"] = "model_context_window_exceeded"
        responses.append(response)
    else:
        responses.append(
            rejection(
                api,
                "The input (1048576 tokens) is longer than the "
                "model's context length (1048576 tokens).",
            )
        )
    provider, requests = sdk_provider(api, responses)
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    recorded = RecordingProvider(provider, tmp_path, usage, [])
    example = Example("task", MESSAGES, "", {"task_dir": str(tmp_path), "suite": "deepswe"})
    async with provider:
        result = await DeepSWE().evaluate(example, recorded, GENERATION, tmp_path)
    turns = 1 if termination == "rejection" else 2
    assert result.score == 1 and sandbox.graded and sandbox.commands == ["fix"]
    assert result.answer == ("prior answer" if termination == "rejection" else "final answer")
    assert result.details["stop_reason"] == "context_limit"
    assert result.details["turns"] == len(recorded.calls) == turns
    assert result.details["tool_calls"] == 1
    assert result.details["generated_tokens"] == usage["output_tokens"] == turns * 7
    assert len(list(tmp_path.glob("call-*.json"))) == turns
    if termination != "rejection":
        call = json.loads((tmp_path / "call-0001.json").read_text())
        assert call["stop_reason"] == "context_limit"
        assert call["request"] == json.loads(requests[-1].content)
        assert call["request"][OUTPUT_TOKEN_FIELDS[api]] == (
            948575 if termination == "clamped" else 999993
        )


@pytest.mark.parametrize(
    "budget,per_turn,stop_reason,used,expected",
    [
        (1000, 100, "length", 100, "max_tokens"),
        (1000, 100, "context_limit", 50, "context_limit"),
        (100, 100, "length", 99, "max_tokens"),
        (100, 100, "length", 100, "max_tokens"),
        (100, 100, "context_limit", 100, "max_tokens"),
    ],
)
async def test_length_distinguishes_per_turn_budget_context_and_generic_providers(
    budget, per_turn, stop_reason, used, expected
):
    class Provider:
        async def generate(self, *args, **kwargs):
            return Response(
                Message("assistant", (Text("answer"),)), Usage(5, used), stop_reason, {}, {}
            )

    async def execute(arguments):
        pytest.fail("Length-limited responses must not execute tools")

    answer, details = await run_tools(
        Provider(),
        Generation(budget),
        MESSAGES,
        TOOL,
        execute,
        max_turns=3,
        max_tool_calls=3,
        max_turn_tokens=per_turn,
    )
    assert answer == "answer" and details["stop_reason"] == expected
    assert details["generated_tokens"] == used and details["tool_calls"] == 0
