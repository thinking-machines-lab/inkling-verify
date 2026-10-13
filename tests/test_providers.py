from __future__ import annotations

import json

import httpx
import pytest
from openai import BadRequestError

from inkling_vendor_verifier.types import Generation, Message, Text

APIS = ("openai-completions", "openai-responses", "anthropic")
GENERATION = Generation(4096, temperature=None, top_p=None, reasoning_effort=None)


@pytest.mark.parametrize("api", APIS)
async def test_usage_and_recorded_request_match_sdk(sdk_provider, response_body, api):
    provider, requests = sdk_provider(api, [response_body(api)])
    async with provider:
        result = await provider.generate([Message("user", (Text("Question"),))], GENERATION)
    assert json.loads(requests[0].content) == result.request
    assert result.text == "answer"
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 7
    assert result.usage.cached_input_tokens == 3
    assert "secret-do-not-log" not in json.dumps(result.raw | result.request)
    assert requests[0].url.host == "vendor.test"
    if api == "openai-responses":
        assert result.request["store"] is False
        assert result.request["include"] == ["reasoning.encrypted_content"]


async def test_responses_phase_and_reasoning_replay(sdk_provider, response_body):
    raw = response_body("openai-responses")
    provider, requests = sdk_provider("openai-responses", [raw, raw])
    async with provider:
        first = await provider.generate([Message("user", (Text("First"),))], GENERATION)
        await provider.generate([first.message, Message("user", (Text("Continue"),))], GENERATION)
    assert json.loads(requests[1].content)["input"][:2] == raw["output"]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "length, tools, stop_reason",
    [(False, (), "stop"), (True, (), "length"), (False, ("pwd",), "tool_calls")],
)
async def test_stop_reason(sdk_provider, response_body, api, length, tools, stop_reason):
    provider, _ = sdk_provider(api, [response_body(api, length=length, tools=tools)])
    async with provider:
        result = await provider.generate([Message("user", (Text("Question"),))], GENERATION)
    assert result.stop_reason == stop_reason


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("native", [False, True])
async def test_missing_or_cross_protocol_native_state_rejected_before_http(
    sdk_provider, api, native
):
    provider, requests = sdk_provider(api, [])
    other_api = "anthropic" if api != "anthropic" else "openai-responses"
    message = Message("assistant", native={"api": other_api, "payload": []} if native else None)
    async with provider:
        with pytest.raises(
            ValueError, match="across API protocols" if native else "requires native"
        ):
            await provider.generate([message], GENERATION)
    assert not requests


async def test_api_error_is_not_retried_with_different_options(sdk_provider):
    provider, requests = sdk_provider(
        "openai-completions",
        [
            httpx.Response(
                400,
                json={
                    "error": {"message": "temperature unsupported", "type": "invalid_request_error"}
                },
            ),
        ],
    )
    async with provider:
        with pytest.raises(BadRequestError, match="temperature unsupported"):
            await provider.generate([Message("user", (Text("Question"),))], Generation(4096))
    assert len(requests) == 1


@pytest.mark.parametrize("api", APIS)
async def test_missing_usage_fails_loudly(sdk_provider, response_body, api):
    raw = response_body(api)
    raw.pop("usage")
    provider, _ = sdk_provider(api, [raw])
    async with provider:
        with pytest.raises(KeyError, match="usage"):
            await provider.generate([Message("user", (Text("Question"),))], GENERATION)
