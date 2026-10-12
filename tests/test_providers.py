from __future__ import annotations

import base64
import json

import httpx
import pytest
from openai import BadRequestError

from inkling_vendor_verifier import providers
from inkling_vendor_verifier.types import Generation, Image, Message, Text, Tool

APIS = ("openai-completions", "openai-responses", "anthropic")
GENERATION = Generation(4096, temperature=None, top_p=None, reasoning_effort=None)
TOOL = Tool(
    "bash",
    "Execute a shell command",
    {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)
SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("api", APIS)
async def test_images_are_byte_exact_and_request_auditable(sdk_provider, response_body, api):
    image = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 43000
    provider, requests = sdk_provider(api, [response_body(api)])
    async with provider:
        result = await provider.generate(
            [
                Message("system", (Text("Instructions"),)),
                Message("user", (Text("Look at this"), Image(image, "image/png"))),
            ],
            GENERATION,
        )
    body = json.loads(requests[0].content)
    if api == "anthropic":
        data = body["messages"][0]["content"][1]["source"]["data"]
        assert body["system"] == [{"type": "text", "text": "Instructions"}]
    elif api == "openai-responses":
        data = body["input"][1]["content"][1]["image_url"].split(",", 1)[1]
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
    else:
        data = body["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(data) == image
    assert body == result.request
    assert result.text == "answer"
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 7
    assert result.usage.cached_input_tokens == 3
    assert "secret-do-not-log" not in json.dumps(result.raw | result.request)
    assert requests[0].url.host == "vendor.test"


@pytest.mark.parametrize("api", APIS)
async def test_native_tool_state_survives_round_trip(sdk_provider, response_body, api):
    first_body = response_body(api, None, tools=("pwd",))
    provider, requests = sdk_provider(api, [first_body, response_body(api)])
    messages = [Message("user", (Text("Use bash"),))]
    async with provider:
        first = await provider.generate(messages, GENERATION, tools=[TOOL])
        assert first.stop_reason == "tool_calls"
        assert first.message.tool_calls[0].id == "call_1"
        messages += [first.message, Message("tool", (Text("/repo"),), tool_call_id="call_1")]
        second = await provider.generate(messages, GENERATION, tools=[TOOL])
    body = json.loads(requests[1].content)
    if api == "openai-completions":
        assert body["messages"][1] == first_body["choices"][0]["message"]
        assert body["messages"][2]["tool_call_id"] == "call_1"
    elif api == "openai-responses":
        assert body["input"][1:3] == first_body["output"]
        assert body["input"][3] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "/repo",
        }
    else:
        assert body["messages"][1]["content"] == first_body["content"]
        assert body["messages"][2]["content"][0]["tool_use_id"] == "call_1"
    assert second.stop_reason == "stop"


async def test_responses_phase_and_reasoning_replay(sdk_provider, response_body):
    raw = response_body("openai-responses")
    provider, requests = sdk_provider("openai-responses", [raw, raw])
    async with provider:
        first = await provider.generate([Message("user", (Text("First"),))], GENERATION)
        await provider.generate([first.message, Message("user", (Text("Continue"),))], GENERATION)
    assert json.loads(requests[1].content)["input"][:2] == raw["output"]


@pytest.mark.parametrize("api", APIS)
async def test_structured_output_and_explicit_sampling(sdk_provider, response_body, api):
    provider, requests = sdk_provider(api, [response_body(api)], extra_body={"vendor_option": True})
    generation = Generation(32768, temperature=0, top_p=1, reasoning_effort="none")
    async with provider:
        result = await provider.generate(
            [Message("user", (Text("Grade"),))], generation, response_schema=SCHEMA
        )
    body = json.loads(requests[0].content)
    assert body == result.request
    assert body["temperature"] == 0
    assert body["top_p"] == 1
    assert body["vendor_option"] is True
    if api == "openai-completions":
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA
        assert body["reasoning_effort"] == "none"
    elif api == "openai-responses":
        assert body["text"]["format"]["schema"] == SCHEMA
        assert body["reasoning"] == {"effort": "none"}
    else:
        assert body["output_config"]["format"]["schema"] == SCHEMA
        assert body["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("api", APIS)
async def test_truncation_is_not_a_complete_answer(sdk_provider, response_body, api):
    provider, _ = sdk_provider(api, [response_body(api, length=True)])
    async with provider:
        result = await provider.generate([Message("user", (Text("Question"),))], GENERATION)
    assert result.stop_reason == "length"


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


@pytest.mark.parametrize("key", ["model", "max_tokens", "reasoning", "store", "tools", "stream"])
def test_extra_body_cannot_override_workload(key):
    with pytest.raises(ValueError, match="cannot override"):
        providers.APIProvider("openai-responses", "vendor", extra_body={key: "bad"})


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
