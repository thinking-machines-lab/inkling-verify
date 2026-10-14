"""Shared multi-turn execution for a single sandbox tool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

from .types import Generation, Image, Message, Provider, Text, Tool


async def run_tools(
    provider: Provider,
    generation: Generation,
    messages: Sequence[Message],
    tool: Tool,
    execute: Callable[[str], Awaitable[tuple[Text | Image, ...]]],
    *,
    max_turns: int,
    max_tool_calls: int,
    max_turn_tokens: int,
    timeout: float | None = None,
) -> tuple[str, dict[str, Any]]:
    history = list(messages)
    remaining = generation.max_tokens
    calls = turns = 0
    stop, answer = "max_turns", ""
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            for turns in range(1, max_turns + 1):
                response = await provider.generate(
                    history,
                    replace(generation, max_tokens=min(remaining, max_turn_tokens)),
                    tools=(tool,),
                )
                if response.usage.output_tokens <= 0:
                    raise RuntimeError(
                        "Tool generation requires positive reported output token usage"
                    )
                remaining -= response.usage.output_tokens
                history.append(response.message)
                answer = response.text
                if response.stop_reason == "length" or remaining <= 0:
                    stop = "max_tokens"
                    break
                if not response.message.tool_calls:
                    if response.stop_reason == "tool_calls":
                        raise ValueError("Tool-call response contained no tool calls")
                    stop = response.stop_reason
                    break
                if calls + len(response.message.tool_calls) > max_tool_calls:
                    stop = "max_tool_calls"
                    break
                for call in response.message.tool_calls:
                    calls += 1
                    content = (
                        await execute(call.arguments)
                        if call.name == tool.name
                        else (Text(json.dumps({"error": f"Unknown tool: {call.name}"})),)
                    )
                    history.append(Message("tool", content, tool_call_id=call.id))
                if calls == max_tool_calls:
                    stop = "max_tool_calls"
                    break
    except TimeoutError:
        if not deadline.expired():
            raise
        stop = "timeout"
    return answer, {
        "stop_reason": stop,
        "turns": turns,
        "tool_calls": calls,
        "generated_tokens": generation.max_tokens - remaining,
        "token_budget_kind": "cumulative_generated_tokens",
    }
