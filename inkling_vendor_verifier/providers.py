"""Protocol adapters; benchmark prompts and scoring never depend on an API dialect."""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Self, get_args

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from .types import Api, Generation, Image, Message, Response, Text, Tool, ToolCall, Usage

_APIS = get_args(Api)


def _text(message: Message) -> str:
    if any(not isinstance(part, Text) for part in message.content):
        raise ValueError(f"{message.role} messages must contain only text here")
    return message.text


def _encoded(image: Image) -> str:
    return base64.b64encode(image.data).decode("ascii")


def _data_uri(image: Image) -> str:
    return f"data:{image.media_type};base64,{_encoded(image)}"


def _native(message: Message, api: Api) -> Any | None:
    if message.native is None:
        if message.role == "assistant" or message.tool_calls:
            raise ValueError("Assistant history requires native API state")
        return None
    if message.role != "assistant" or message.native.get("api") != api:
        raise ValueError("Native assistant state cannot be replayed across API protocols")
    return copy.deepcopy(message.native["payload"])


def _tool_id(message: Message) -> str:
    if not message.tool_call_id:
        raise ValueError("A tool result requires tool_call_id")
    return message.tool_call_id


def _openai_tool_images(messages: Sequence[Message]) -> list[Message]:
    result = []
    images: list[Text | Image] = []
    for message in messages:
        if message.role != "tool" and images:
            result.append(Message("user", tuple(images)))
            images = []
        if message.role == "tool" and any(isinstance(part, Image) for part in message.content):
            text = []
            image_index = 0
            for part in message.content:
                if isinstance(part, Image):
                    image_index += 1
                    label = f"Image {image_index} from tool call {_tool_id(message)}"
                    text.append(Text(f"\n[{label} attached below.]\n"))
                    images.extend((Text(f"{label}:"), part))
                else:
                    text.append(part)
            message = replace(message, content=tuple(text))
        result.append(message)
    if images:
        result.append(Message("user", tuple(images)))
    return result


def _chat_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    result = []
    for message in _openai_tool_images(messages):
        native = _native(message, "openai-completions")
        if native is not None:
            result.append(native)
            continue
        if message.role == "tool":
            result.append(
                {"role": "tool", "tool_call_id": _tool_id(message), "content": _text(message)}
            )
            continue
        if message.role == "system":
            content: Any = _text(message)
        else:
            content = [
                {"type": "text", "text": part.text}
                if isinstance(part, Text)
                else {"type": "image_url", "image_url": {"url": _data_uri(part)}}
                for part in message.content
            ]
        result.append({"role": message.role, "content": content})
    return result


def _response_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    result = []
    for message in _openai_tool_images(messages):
        native = _native(message, "openai-responses")
        if native is not None:
            result.extend(native)
            continue
        if message.role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": _tool_id(message),
                    "output": _text(message),
                }
            )
            continue
        if message.content:
            if message.role == "system":
                content: Any = _text(message)
            else:
                content = [
                    {"type": "input_text", "text": part.text}
                    if isinstance(part, Text)
                    else {"type": "input_image", "image_url": _data_uri(part)}
                    for part in message.content
                ]
            result.append({"role": message.role, "content": content})
    return result


def _anthropic_messages(
    messages: Sequence[Message],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system, result = [], []
    for message in messages:
        native = _native(message, "anthropic")
        if message.role == "system":
            if result:
                raise ValueError("Anthropic system messages must precede conversation messages")
            system.append({"type": "text", "text": _text(message)})
            continue
        role = "user" if message.role == "tool" else message.role
        if native is not None:
            content = native
        else:
            content = [
                {"type": "text", "text": part.text}
                if isinstance(part, Text)
                else {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": _encoded(part),
                    },
                }
                for part in message.content
            ]
            if message.role == "tool":
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": _tool_id(message),
                        "content": content
                        if any(isinstance(part, Image) for part in message.content)
                        else _text(message),
                    }
                ]
        if message.role == "tool" and result and result[-1]["role"] == "user":
            result[-1]["content"].extend(content)
        else:
            result.append({"role": role, "content": content})
    return system, result


def _usage(raw: dict[str, Any], api: Api) -> Usage:
    usage = raw["usage"]
    if not isinstance(usage, dict):
        raise ValueError("The API response must include token usage")
    if api == "openai-completions":
        details = usage.get("prompt_tokens_details") or {}
        return Usage(
            usage["prompt_tokens"], usage["completion_tokens"], details.get("cached_tokens", 0)
        )
    if api == "openai-responses":
        details = usage.get("input_tokens_details") or {}
        return Usage(usage["input_tokens"], usage["output_tokens"], details.get("cached_tokens", 0))
    cached = usage.get("cache_read_input_tokens") or 0
    created = usage.get("cache_creation_input_tokens") or 0
    return Usage(usage["input_tokens"] + cached + created, usage["output_tokens"], cached)


def _decode_chat(raw: dict[str, Any]) -> tuple[Message, str]:
    if len(raw["choices"]) != 1:
        raise ValueError("Expected exactly one chat completion choice")
    choice = raw["choices"][0]
    payload = copy.deepcopy(choice["message"])
    if payload.get("audio") or payload.get("function_call"):
        raise ValueError("Audio and legacy function_call outputs are not supported")
    payload.pop("annotations", None)
    payload.pop("audio", None)
    payload.pop("function_call", None)
    calls = []
    for call in payload.get("tool_calls") or []:
        if call["type"] != "function":
            raise ValueError(f"Unsupported chat tool type: {call['type']}")
        calls.append(ToolCall(call["id"], call["function"]["name"], call["function"]["arguments"]))
    text = payload.get("content") or ""
    if not isinstance(text, str):
        raise ValueError("Expected text in the assistant chat completion")
    reason = choice["finish_reason"]
    stop = {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "length": "length",
        "content_filter": "refusal",
    }.get(reason)
    if stop is None:
        raise ValueError(f"Unsupported chat finish reason: {reason}")
    if payload.get("refusal"):
        stop = "refusal"
    return Message(
        "assistant",
        (Text(text),),
        tuple(calls),
        native={
            "api": "openai-completions",
            "payload": payload,
        },
    ), stop


def _decode_responses(raw: dict[str, Any]) -> tuple[Message, str]:
    if raw.get("error"):
        raise ValueError(f"Responses API failed: {raw['error']}")
    status = raw["status"]
    if status not in {"completed", "incomplete"}:
        raise ValueError(f"Unexpected Responses API status: {status}")
    stop = "stop"
    if status == "incomplete":
        reason = (raw.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            stop = "length"
        elif reason == "content_filter":
            stop = "refusal"
        else:
            raise ValueError(f"Unsupported incomplete response: {reason}")
    parts, calls = [], []
    for item in raw["output"]:
        if item["type"] == "message":
            for content in item["content"]:
                if content["type"] == "output_text":
                    parts.append(Text(content["text"]))
                elif content["type"] == "refusal":
                    stop = "refusal"
                else:
                    raise ValueError(f"Unsupported Responses content: {content['type']}")
        elif item["type"] == "function_call":
            calls.append(ToolCall(item["call_id"], item["name"], item["arguments"]))
        elif item["type"] != "reasoning":
            raise ValueError(f"Unsupported Responses output: {item['type']}")
    if calls and stop == "stop":
        stop = "tool_calls"
    return Message(
        "assistant",
        tuple(parts),
        tuple(calls),
        native={
            "api": "openai-responses",
            "payload": copy.deepcopy(raw["output"]),
        },
    ), stop


def _decode_anthropic(raw: dict[str, Any]) -> tuple[Message, str]:
    reason = raw["stop_reason"]
    stop = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
        "refusal": "refusal",
    }.get(reason)
    if stop is None:
        raise ValueError(f"Unsupported Anthropic stop reason: {reason}")
    parts, calls = [], []
    for block in raw["content"]:
        if block["type"] == "text":
            parts.append(Text(block["text"]))
        elif block["type"] == "tool_use":
            calls.append(ToolCall(block["id"], block["name"], json.dumps(block["input"])))
        elif block["type"] not in {"thinking", "redacted_thinking"}:
            raise ValueError(f"Unsupported Anthropic output: {block['type']}")
    return Message(
        "assistant",
        tuple(parts),
        tuple(calls),
        native={
            "api": "anthropic",
            "payload": copy.deepcopy(raw["content"]),
        },
    ), stop


class APIProvider:
    def __init__(
        self,
        api: Api,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 7200,
        max_retries: int = 3,
    ) -> None:
        if api not in _APIS:
            raise ValueError(f"Unknown API: {api}")
        self.api, self.model = api, model
        kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if base_url is not None:
            kwargs["base_url"] = base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        self.client = AsyncAnthropic(**kwargs) if api == "anthropic" else AsyncOpenAI(**kwargs)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.client.close()

    async def generate(
        self,
        messages: Sequence[Message],
        generation: Generation,
        *,
        tools: Sequence[Tool] = (),
        response_schema: dict[str, Any] | None = None,
    ) -> Response:
        request: dict[str, Any] = {"model": self.model}
        if generation.temperature is not None:
            request["temperature"] = generation.temperature
        if generation.top_p is not None:
            request["top_p"] = generation.top_p
        if self.api == "openai-completions":
            request.update(
                messages=_chat_messages(messages), max_completion_tokens=generation.max_tokens
            )
            if generation.reasoning_effort is not None:
                request["reasoning_effort"] = generation.reasoning_effort
            if tools:
                request["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                    for tool in tools
                ]
            if response_schema is not None:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": response_schema,
                        "strict": True,
                    },
                }
            create = self.client.chat.completions.create
            decode = _decode_chat
        elif self.api == "openai-responses":
            request.update(
                input=_response_messages(messages),
                max_output_tokens=generation.max_tokens,
                store=False,
                include=["reasoning.encrypted_content"],
            )
            if generation.reasoning_effort is not None:
                request["reasoning"] = {"effort": generation.reasoning_effort}
            if tools:
                request["tools"] = [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": False,
                    }
                    for tool in tools
                ]
            if response_schema is not None:
                request["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "schema": response_schema,
                        "strict": True,
                    }
                }
            create = self.client.responses.create
            decode = _decode_responses
        else:
            system, items = _anthropic_messages(messages)
            request.update(messages=items, max_tokens=generation.max_tokens)
            if system:
                request["system"] = system
            output_config: dict[str, Any] = {}
            effort = generation.reasoning_effort
            if effort == "none":
                request["thinking"] = {"type": "disabled"}
            elif effort is not None:
                request["thinking"] = {"type": "adaptive"}
                output_config["effort"] = effort
            if response_schema is not None:
                output_config["format"] = {"type": "json_schema", "schema": response_schema}
            if output_config:
                request["output_config"] = output_config
            if tools:
                request["tools"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    }
                    for tool in tools
                ]
            create = self.client.messages.create
            decode = _decode_anthropic
        response = await create(**request)
        raw = response.model_dump(mode="json", exclude_unset=True)
        message, stop = decode(raw)
        return Response(message, _usage(raw, self.api), stop, raw, copy.deepcopy(request))
