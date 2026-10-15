import asyncio

import pytest

from inkling_verify.benchmarks import deepswe
from inkling_verify.benchmarks.deepswe import DeepSWE
from inkling_verify.types import (
    Example,
    Generation,
    Message,
    Response,
    Text,
    ToolCall,
    Usage,
)


def reply(text="", calls=(), tokens=4, stop="stop"):
    return Response(
        Message("assistant", (Text(text),), tool_calls=calls), Usage(20, tokens), stop, {}, {}
    )


class Provider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, messages, generation, *, tools=()):
        self.calls.append((list(messages), generation, tools))
        return next(self.responses)


@pytest.fixture
def task(code_sandbox):
    example = Example(
        "task",
        (Message("user", (Text("fix the bug"),)),),
        "",
        {"task_dir": "/task", "suite": "deepswe"},
    )
    return example, code_sandbox


@pytest.mark.parametrize(
    "limit, tool_count, executed",
    [
        ("length", 0, 0),
        ("max_tokens", 0, 0),
        ("max_turns", 1, 1),
        ("max_tool_calls", 1, 1),
        ("max_tool_calls", 2, 0),
        ("timeout", 0, 0),
    ],
)
async def test_limits_still_grade_committed_patch(
    task, tmp_path, monkeypatch, limit, tool_count, executed
):
    example, sandbox = task
    monkeypatch.setattr(deepswe, "MAX_TURNS", 1 if limit == "max_turns" else 2)
    monkeypatch.setattr(deepswe, "MAX_TOOL_CALLS", 1 if limit == "max_tool_calls" else 2)
    provider = Provider(
        [
            reply(
                "partial",
                calls=tuple(
                    ToolCall(f"call-{i}", "bash", '{"command":"echo ok"}')
                    for i in range(tool_count)
                ),
                tokens=30 if limit == "max_tokens" else 4,
                stop="length" if limit == "length" else "tool_calls" if tool_count else "stop",
            )
        ]
    )
    if limit == "timeout":
        monkeypatch.setattr(deepswe, "AGENT_TIMEOUT_SECONDS", 0)
        generate = provider.generate

        async def wait_for_deadline(*args, **kwargs):
            await generate(*args, **kwargs)
            await asyncio.sleep(0)
            pytest.fail("Agent deadline did not cancel generation")

        monkeypatch.setattr(provider, "generate", wait_for_deadline)
    result = await DeepSWE().evaluate(example, provider, Generation(30), tmp_path)
    assert result.score == 1
    assert result.details["verifier_reward"] == 1
    assert result.details["stop_reason"] == ("max_tokens" if limit == "length" else limit)
    assert result.details["turns"] == len(provider.calls) == 1
    assert result.details["tool_calls"] == executed
    assert sandbox.commands == [("echo ok", None, 300)] * executed
    assert sandbox.graded


@pytest.mark.parametrize("allow_network", [False, True])
async def test_bad_tool_arguments_are_model_feedback_not_host_commands(
    task, tmp_path, allow_network
):
    example, sandbox = task
    provider = Provider(
        [
            reply(calls=(ToolCall("call", "bash", '{"command":"bad","timeout":true}'),)),
            reply("done"),
        ]
    )
    result = await DeepSWE(allow_network=allow_network).evaluate(
        example, provider, Generation(30), tmp_path
    )
    assert not sandbox.commands
    assert "positive integer" in provider.calls[1][0][-1].text
    assert sandbox.allow_network is allow_network
    assert ("no internet access" in provider.calls[0][2][0].description) is not allow_network
    assert result.details["allow_network"] is allow_network
    assert result.details["solve_allowed_hosts"] == (["*"] if allow_network else [])
    assert result.details["verifier_allowed_hosts"] == (["*"] if allow_network else [])


async def test_provider_failure_is_not_a_zero_score(task, tmp_path):
    example, sandbox = task

    class BrokenProvider:
        async def generate(self, *args, **kwargs):
            raise TimeoutError("API failed")

    with pytest.raises(TimeoutError, match="API failed"):
        await DeepSWE().evaluate(example, BrokenProvider(), Generation(30), tmp_path)
    assert not sandbox.graded
