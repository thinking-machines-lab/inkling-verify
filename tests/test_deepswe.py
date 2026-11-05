import asyncio
import json

import pytest

from inkling_verify.benchmarks.deepswe import DeepSWE
from inkling_verify.mini_swe import PodmanEnvironment
from inkling_verify.types import Example, Generation, Message, Response, Text, ToolCall, Usage

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def reply(text="", commands=(), tokens=4, stop="stop"):
    return Response(
        Message(
            "assistant",
            (Text(text),),
            tool_calls=tuple(
                ToolCall(f"call-{i}", "bash", json.dumps({"command": command}))
                for i, command in enumerate(commands)
            ),
        ),
        Usage(20, tokens),
        stop,
        {},
        {},
    )


class Provider:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def generate(self, messages, generation, *, tools=()):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def task(code_sandbox):
    return Example(
        "task",
        (Message("user", (Text("fix the bug"),)),),
        "",
        {"task_dir": "/task", "suite": "deepswe"},
    ), code_sandbox


@pytest.mark.parametrize(
    "responses, stop, turns, commands",
    [
        ([reply(tokens=30, stop="length")], "max_tokens", 1, []),
        ([reply(commands=("echo ok",), tokens=30)], "max_tokens", 1, ["echo ok"]),
        ([reply("done")] * 3, "format_error", 3, []),
        ([reply(commands=(SUBMIT,))], "stop", 1, [SUBMIT]),
    ],
)
async def test_exits_still_grade_committed_patch(task, tmp_path, responses, stop, turns, commands):
    example, sandbox = task
    provider = Provider(responses)
    result = await DeepSWE().evaluate(example, provider, Generation(30), tmp_path)
    assert result.score == result.details["verifier_reward"] == 1
    assert result.details["stop_reason"] == stop
    assert result.details["turns"] == turns
    assert result.details["tool_calls"] == len(commands)
    assert sandbox.commands == commands
    assert sandbox.graded
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["info"]["mini_version"] == "2.4.6"
    assert trajectory["info"]["exit_status"] == result.details["agent_exit_status"]


@pytest.mark.parametrize("timeout_seconds", [0, -1, float("inf"), float("nan"), True])
def test_invalid_agent_deadline(timeout_seconds):
    with pytest.raises(ValueError, match="positive and finite"):
        DeepSWE(timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("operation", ["generation", "command"])
@pytest.mark.parametrize("external_cancel", [False, True])
async def test_cancellation_stops_agent_before_grading(
    task, tmp_path, monkeypatch, operation, external_cancel
):
    example, sandbox = task
    started, stopped = asyncio.Event(), asyncio.Event()

    async def wait(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    provider = Provider([reply(commands=("echo ok",))])
    monkeypatch.setattr(
        provider if operation == "generation" else PodmanEnvironment,
        "generate" if operation == "generation" else "_execute",
        wait,
    )
    original_grade = sandbox.grade

    async def grade():
        assert stopped.is_set(), "Agent must finish cancelling before grading"
        return await original_grade()

    monkeypatch.setattr(sandbox, "grade", grade)
    evaluation = asyncio.create_task(
        DeepSWE(timeout_seconds=10 if external_cancel else 0.2).evaluate(
            example, provider, Generation(30), tmp_path
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    if external_cancel:
        evaluation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(evaluation, timeout=2)
        assert not sandbox.graded
    else:
        result = await asyncio.wait_for(evaluation, timeout=2)
        assert result.details["stop_reason"] == "timeout"
        assert result.score == 1
    assert stopped.is_set()


async def test_agent_deadline_excludes_setup_and_grading(task, tmp_path, monkeypatch):
    example, sandbox = task
    original_grade = sandbox.grade

    async def enter(self):
        await asyncio.sleep(0.1)
        return self

    async def grade():
        await asyncio.sleep(0.1)
        return await original_grade()

    monkeypatch.setattr(type(sandbox), "__aenter__", enter)
    monkeypatch.setattr(sandbox, "grade", grade)
    result = await DeepSWE(timeout_seconds=0.05).evaluate(
        example, Provider([reply(commands=(SUBMIT,))]), Generation(30), tmp_path
    )
    assert result.details["stop_reason"] == "stop"
    assert result.score == 1


@pytest.mark.parametrize("allow_network", [False, True])
async def test_network_policy_applies_to_both_containers(task, tmp_path, allow_network):
    example, sandbox = task
    result = await DeepSWE(allow_network=allow_network).evaluate(
        example, Provider([reply(commands=(SUBMIT,))]), Generation(30), tmp_path
    )
    assert sandbox.allow_network is allow_network
    assert result.details["allow_network"] is allow_network
    assert result.details["solve_allowed_hosts"] == (["*"] if allow_network else [])
    assert result.details["verifier_allowed_hosts"] == (["*"] if allow_network else [])


@pytest.mark.parametrize("error", [TimeoutError("API failed"), ValueError("invalid API data")])
async def test_provider_failure_is_not_a_zero_score(task, tmp_path, error):
    example, sandbox = task
    with pytest.raises(type(error), match=str(error)):
        await DeepSWE().evaluate(example, Provider([error]), Generation(30), tmp_path)
    assert not sandbox.graded


async def test_agent_concurrency_is_not_capped_by_thread_pool(task, tmp_path, monkeypatch):
    example, _ = task
    entered = 0
    ready = asyncio.Event()

    async def generate(*args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == 48:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=10)
        return reply(commands=(SUBMIT,))

    provider = Provider([])
    monkeypatch.setattr(provider, "generate", generate)
    results = await asyncio.gather(
        *(
            DeepSWE().evaluate(example, provider, Generation(30), tmp_path / str(i))
            for i in range(48)
        )
    )
    assert len(results) == 48
    assert all(result.details["agent_exit_status"] == "Submitted" for result in results)
