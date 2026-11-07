import asyncio
import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import local
from minisweagent.exceptions import Submitted

from inkling_verify import mini_swe
from inkling_verify.sandbox import CommandResult
from inkling_verify.types import Generation, ToolCall

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


@pytest.mark.parametrize(
    "shell",
    [
        "printf out; printf err >&2; exit 7",
        "printf '\\377'",
        "printf 'prefix\x00suffix'",
        "printf ready; sleep 5",
        "printf '%s' \"$PAGER\"",
        'python3 -c \'print("head" + "x" * 20000 + "tail")\'',
    ],
)
def test_container_execution_matches_upstream(shell, tmp_path):
    config = {"cwd": str(tmp_path), "env": {"PAGER": "cat"}, "timeout": 1}
    action = {"command": shell}
    expected = local.LocalEnvironment(**config).execute(action)
    result = subprocess.run(
        [sys.executable, "-c", mini_swe.container_script()],
        input=json.dumps({"config": config, "action": action}),
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    assert json.loads(result.stdout) == expected


def test_timeout_kills_descendants(tmp_path):
    marker = tmp_path / "orphan"
    action = {"command": f"(sleep 2; touch {marker}) & wait"}
    payload = {"config": {"cwd": str(tmp_path), "env": {}, "timeout": 0.05}, "action": action}
    result = subprocess.run(
        [sys.executable, "-c", mini_swe.container_script()],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=1,
    )
    assert json.loads(result.stdout)["extra"]["exception_type"] == "TimeoutExpired"
    assert not marker.exists()


async def test_prompt_actions_and_observations_match_upstream(
    sdk_provider, response_body, tmp_path, monkeypatch
):
    from litellm import ModelResponse
    from minisweagent.models.litellm_model import LitellmModel

    responses = [
        response_body("openai-completions", "done"),
        response_body("openai-completions", tools=(ToolCall("bad", "bash", "{"),)),
        response_body("openai-completions", tools=("inspect",)),
        response_body("openai-completions", tools=(SUBMIT,)),
    ]
    variables = {"system": "Linux", "release": "test", "version": "test", "machine": "x86_64"}
    output = "head" + "x" * 20000 + "tail"

    class ReferenceModel(LitellmModel):
        def __init__(self):
            super().__init__(model_name="test", **mini_swe.CONFIG["model"])
            self.requests = []

        def _query(self, messages, **kwargs):
            self.requests.append(copy.deepcopy(messages))
            return ModelResponse(**responses[len(self.requests) - 1])

        def _calculate_cost(self, response):
            return {"cost": 0.0}

    class ReferenceEnvironment(local.LocalEnvironment):
        def get_template_vars(self, **kwargs):
            return self.config.model_dump() | variables | kwargs

        def execute(self, action, **kwargs):
            result = {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
                if action["command"] == SUBMIT
                else output,
                "returncode": 0,
                "exception_info": "",
            }
            self._check_finished(result)
            return result

    reference_model = ReferenceModel()
    reference_env = ReferenceEnvironment(**mini_swe.CONFIG["environment"])
    config = mini_swe.CONFIG["agent"] | {"cost_limit": 0}
    del config["mode"]
    reference = DefaultAgent(reference_model, reference_env, **config)
    assert reference.run("Fix the bug")["exit_status"] == "Submitted"

    monkeypatch.setattr(
        mini_swe, "command", AsyncMock(return_value=CommandResult(json.dumps(variables), "", 0, 0))
    )

    async def execute(self, action):
        try:
            return reference_env.execute(action)
        except Submitted:
            return {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
                "returncode": 0,
                "exception_info": "",
            }

    monkeypatch.setattr(mini_swe.PodmanEnvironment, "_execute", execute)
    provider, requests = sdk_provider("openai-completions", responses)
    async with provider:
        _, details = await mini_swe.run_agent(
            provider, Generation(1000), "Fix the bug", SimpleNamespace(name="test"), tmp_path, 10
        )
    assert details["agent_exit_status"] == "Submitted"

    def protocol(messages):
        normalized = [
            {
                key: message[key]
                for key in ("role", "content", "tool_calls", "tool_call_id", "reasoning_content")
                if message.get(key) is not None
            }
            for message in messages
        ]
        for message in normalized:
            if isinstance(message.get("content"), list):
                assert all(part["type"] == "text" for part in message["content"])
                message["content"] = "".join(part["text"] for part in message["content"])
        return normalized

    assert [protocol(json.loads(request.content)["messages"]) for request in requests] == [
        protocol(messages) for messages in reference_model.requests
    ]
    observed = json.loads(json.loads(requests[-1].content)["messages"][-1]["content"])
    assert observed["output_head"] == output[:5000]
    assert observed["output_tail"] == output[-5000:]
    assert observed["elided_chars"] == len(output) - 10000


@pytest.mark.parametrize("read_stdin", [False, True])
async def test_failed_podman_exec_preserves_diagnostics(monkeypatch, tmp_path, read_stdin):
    podman = tmp_path / "podman"
    read = (
        "payload = json.load(sys.stdin)\nassert payload['action']['command'] == 'x' * 1048576\n"
        if read_stdin
        else "os.close(0)\n"
    )
    podman.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "assert sys.argv[1:6] == ['exec', '-i', 'test', 'python3', '-c']\n"
        + read
        + "sys.stdout.write('partial')\n"
        "sys.stderr.write('resource unavailable')\n"
        "sys.exit(255)\n"
    )
    podman.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    environment = mini_swe.PodmanEnvironment(SimpleNamespace(name="test"), {})
    result = await environment._execute({"command": "x" * 1048576})
    assert result == {
        "output": "partialresource unavailable",
        "returncode": -1,
        "exception_info": "Podman exec exited 255",
    }


@pytest.mark.parametrize("error", ["invalid json", TimeoutError("unexpected transport timeout")])
async def test_runner_protocol_errors_fail_loudly(monkeypatch, error):
    run = (
        AsyncMock(side_effect=error)
        if isinstance(error, Exception)
        else AsyncMock(return_value=subprocess.CompletedProcess([], 0, error.encode(), b""))
    )
    monkeypatch.setattr(mini_swe.anyio, "run_process", run)
    environment = mini_swe.PodmanEnvironment(SimpleNamespace(name="test"), {})
    with pytest.raises(TimeoutError if isinstance(error, Exception) else json.JSONDecodeError):
        await environment._execute({"command": "echo hi"})


async def test_exec_timeout_cancels_transport_and_becomes_an_observation(monkeypatch):
    stopped = asyncio.Event()

    async def wait(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    real_deadline = anyio.fail_after
    monkeypatch.setattr(mini_swe.anyio, "run_process", wait)
    monkeypatch.setattr(mini_swe.anyio, "fail_after", lambda seconds: real_deadline(0.01))
    environment = mini_swe.PodmanEnvironment(SimpleNamespace(name="test"), {})
    result = await environment._execute({"command": "echo hi"})
    assert stopped.is_set()
    assert result["returncode"] == -1 and result["exception_info"] == "Podman exec timed out"
