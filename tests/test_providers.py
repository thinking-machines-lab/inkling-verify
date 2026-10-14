from __future__ import annotations

import base64
import copy
import io
import json

import httpx
import pytest
from openai import BadRequestError
from PIL import Image as PILImage

from inkling_vendor_verifier.types import Generation, Image, Message, Text

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


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("kind", ["text", "images", "mixed"])
async def test_image_tool_results_preserve_pixels_order_and_native_history(
    sdk_provider, response_body, api, kind
):
    images = []
    for color in ("red", "green", "blue"):
        buffer = io.BytesIO()
        PILImage.new("RGB", (2, 3), color=color).save(buffer, format="PNG")
        images.append(Image(buffer.getvalue(), "image/png"))
    if kind == "text":
        contents = [(Text("before"), Text("after")), (Text("second"),)]
    elif kind == "images":
        contents = [(images[0], images[1]), (images[2],)]
    else:
        contents = [
            (Text("before"), images[0], Text("middle"), images[1], Text("after")),
            (Text("second"), images[2]),
        ]
    call_response = response_body(api, tools=("first", "second"))
    provider, requests = sdk_provider(api, [call_response, response_body(api), response_body(api)])
    history = [Message("user", (Text("Inspect the images."),))]
    async with provider:
        first = await provider.generate(history, GENERATION)
        original_native = copy.deepcopy(first.message.native)
        history.append(first.message)
        history.extend(
            Message("tool", content, tool_call_id=call.id)
            for call, content in zip(first.message.tool_calls, contents, strict=True)
        )
        second = await provider.generate(history, GENERATION)
        history.extend((second.message, Message("user", (Text("Continue."),))))
        await provider.generate(history, GENERATION)
    assert first.message.native == original_native
    body = json.loads(requests[1].content)
    conversation = body["input" if api == "openai-responses" else "messages"]
    later = json.loads(requests[2].content)["input" if api == "openai-responses" else "messages"]
    assert later[: len(conversation)] == conversation
    if api == "anthropic":
        assert conversation[1]["content"] == call_response["content"]
        assert len(conversation) == 3
        results = conversation[-1]["content"]
        assert [result["tool_use_id"] for result in results] == ["call_1", "call_2"]
        image_blocks = []
        for result, parts in zip(results, contents, strict=True):
            if kind == "text":
                assert result["content"] == "".join(part.text for part in parts)
            else:
                assert [block["type"] for block in result["content"]] == [
                    "text" if isinstance(part, Text) else "image" for part in parts
                ]
                for block, part in zip(result["content"], parts, strict=True):
                    if isinstance(part, Text):
                        assert block["text"] == part.text
                    else:
                        assert block["source"]["media_type"] == part.media_type
                        image_blocks.append(base64.b64decode(block["source"]["data"]))
    else:
        if api == "openai-completions":
            assert conversation[1] == call_response["choices"][0]["message"]
            outputs = conversation[2:]
            call_key, content_key, image_type = "tool_call_id", "content", "image_url"
            assert [output["role"] for output in outputs[:2]] == ["tool", "tool"]
        else:
            native = call_response["output"]
            assert conversation[1 : 1 + len(native)] == native
            outputs = conversation[1 + len(native) :]
            call_key, content_key, image_type = "call_id", "output", "input_image"
            assert [output["type"] for output in outputs[:2]] == [
                "function_call_output",
                "function_call_output",
            ]
        assert [output[call_key] for output in outputs[:2]] == ["call_1", "call_2"]
        image_blocks = []
        for index, parts in enumerate(contents):
            expected = []
            image_index = 0
            for part in parts:
                if isinstance(part, Text):
                    expected.append(part.text)
                else:
                    image_index += 1
                    expected.append(
                        f"\n[Image {image_index} from tool call call_{index + 1} attached below.]\n"
                    )
            assert outputs[index][content_key] == "".join(expected)
        assert len(outputs) == (2 if kind == "text" else 3)
        if kind != "text":
            assert outputs[-1]["role"] == "user"
            parts = outputs[-1]["content"]
            assert [part["text"] for part in parts[::2]] == [
                "Image 1 from tool call call_1:",
                "Image 2 from tool call call_1:",
                "Image 1 from tool call call_2:",
            ]
            for part in parts[1::2]:
                assert part["type"] == image_type
                uri = part["image_url"]
                if api == "openai-completions":
                    uri = uri["url"]
                prefix, encoded = uri.split(",", 1)
                assert prefix == "data:image/png;base64"
                image_blocks.append(base64.b64decode(encoded))
    assert image_blocks == ([] if kind == "text" else [image.data for image in images])
