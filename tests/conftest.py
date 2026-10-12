import json
from functools import partial

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from inkling_vendor_verifier import providers


@pytest.fixture
def sdk_provider(monkeypatch):
    def create(api, responses, **kwargs):
        requests = []
        pending = iter(responses)

        def handle(request):
            requests.append(request)
            response = next(pending)
            return (
                response
                if isinstance(response, httpx.Response)
                else httpx.Response(200, json=response)
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        monkeypatch.setattr(providers, "AsyncOpenAI", partial(AsyncOpenAI, http_client=client))
        monkeypatch.setattr(
            providers, "AsyncAnthropic", partial(AsyncAnthropic, http_client=client)
        )
        options = {
            "model": "vendor",
            "base_url": "https://vendor.test/api/",
            "api_key": "secret-do-not-log",
            "max_retries": 0,
        } | kwargs
        return providers.APIProvider(api, **options), requests

    return create


@pytest.fixture
def response_body():
    def create(api, text="answer", *, tools=(), length=False, cached=True):
        input_tokens, output_tokens = (13, 7) if cached else (10, 5)
        calls = [
            {"id": f"call_{i}", "name": "bash", "arguments": json.dumps({"command": command})}
            for i, command in enumerate(tools, 1)
        ]
        if api == "openai-completions":
            message = {
                "role": "assistant",
                "content": text,
                "reasoning_content": "native reasoning",
            }
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for call in calls
                ]
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            if cached:
                usage["prompt_tokens_details"] = {"cached_tokens": 3}
            return {
                "id": "chat_1",
                "object": "chat.completion",
                "created": 0,
                "model": "vendor",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "length" if length else "tool_calls" if calls else "stop",
                    }
                ],
                "usage": usage,
            }
        if api == "openai-responses":
            output = [
                {
                    "type": "reasoning",
                    "id": "reason_1",
                    "summary": [],
                    "encrypted_content": "opaque reasoning",
                }
            ]
            if text is not None:
                output.append(
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "phase": "commentary" if calls else "final_answer",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                )
            output.extend(
                {
                    "type": "function_call",
                    "id": f"fc_{call['id']}",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "status": "completed",
                }
                for call in calls
            )
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            if cached:
                usage["input_tokens_details"] = {"cached_tokens": 3}
            return {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "vendor",
                "status": "incomplete" if length else "completed",
                "output": output,
                "incomplete_details": {"reason": "max_output_tokens"} if length else None,
                "usage": usage,
            }
        content = [
            {"type": "thinking", "thinking": "reasoning", "signature": "opaque signature"},
            {"type": "redacted_thinking", "data": "opaque redacted state"},
        ]
        if text is not None:
            content.append({"type": "text", "text": text})
        content.extend(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": json.loads(call["arguments"]),
            }
            for call in calls
        )
        usage = {"input_tokens": 8 if cached else input_tokens, "output_tokens": output_tokens}
        if cached:
            usage.update(cache_read_input_tokens=3, cache_creation_input_tokens=2)
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "vendor",
            "content": content,
            "stop_reason": "max_tokens" if length else "tool_use" if calls else "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }

    return create
