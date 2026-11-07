"""Pinned Mini-SWE-Agent with the shared API providers and an existing Podman sandbox."""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import textwrap
from concurrent.futures import CancelledError
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import yaml
from minisweagent import __version__, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import local
from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)

from .context_limits import ContextExhausted
from .sandbox import DeepSWESandbox, command
from .types import Generation, Message, Provider, Text, Tool

AGENT = {"name": "mini-swe-agent", "version": __version__}
CONFIG = yaml.safe_load((package_dir / "config/mini.yaml").read_text())
BASH = Tool(**BASH_TOOL["function"])


def call_async(function, *args):
    anyio.from_thread.check_cancelled()
    try:
        return anyio.from_thread.run(function, *args)
    except CancelledError:
        anyio.from_thread.check_cancelled()
        raise


def limit(status: str) -> LimitsExceeded:
    return LimitsExceeded(
        {"role": "exit", "content": status, "extra": {"exit_status": status, "submission": ""}}
    )


class ProviderModel:
    def __init__(self, provider: Provider, generation: Generation):
        self.provider = provider
        self.generation = generation
        self.responses: list[Message] = []
        self.tokens = 0
        self.context_exhausted = False

    def query(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        remaining = self.generation.max_tokens - self.tokens
        if remaining <= 0:
            raise limit("OutputBudgetExceeded")
        if self.context_exhausted:
            raise limit("ContextExhausted")
        history = [
            self.responses[message["extra"]["response_index"]]
            if message["role"] == "assistant"
            else Message(
                message["role"],
                (Text(message["content"]),),
                tool_call_id=message.get("tool_call_id"),
            )
            for message in messages
        ]
        try:
            response = call_async(
                partial(self.provider.generate, tools=(BASH,)),
                history,
                replace(self.generation, max_tokens=remaining),
            )
        except ContextExhausted as error:
            raise limit("ContextExhausted") from error
        if response.usage.output_tokens <= 0:
            raise RuntimeError("Mini-SWE requires positive reported output token usage")
        self.tokens += response.usage.output_tokens
        self.context_exhausted = response.stop_reason == "context_limit"
        self.responses.append(response.message)
        extra = {"response_index": len(self.responses) - 1, "response": response.raw}
        try:
            actions = parse_toolcall_actions(
                [
                    SimpleNamespace(
                        id=call.id,
                        function=SimpleNamespace(name=call.name, arguments=call.arguments),
                    )
                    for call in response.message.tool_calls
                ],
                format_error_template=CONFIG["model"]["format_error_template"],
                template_kwargs={"finish_reason": response.stop_reason},
            )
        except FormatError as error:
            error.messages[0]["extra"].update(extra)
            raise
        return {
            "role": "assistant",
            "content": response.text,
            "extra": {**extra, "actions": actions},
        }

    def format_message(self, **kwargs) -> dict[str, Any]:
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return format_toolcall_observation_messages(
            actions=message["extra"]["actions"],
            outputs=outputs,
            observation_template=CONFIG["model"]["observation_template"],
            template_vars=template_vars,
        )

    def get_template_vars(self) -> dict[str, Any]:
        return CONFIG["model"]

    def serialize(self) -> dict[str, Any]:
        return {}


def container_script() -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "import os, json, signal, subprocess, sys",
            "from types import SimpleNamespace",
            "from typing import Any",
            inspect.getsource(local._run),
            textwrap.dedent(inspect.getsource(local.LocalEnvironment.execute)),
            "payload = json.load(sys.stdin)",
            "obj = SimpleNamespace(config=SimpleNamespace(**payload['config']), "
            "_check_finished=lambda output: None)",
            "print(json.dumps(execute(obj, payload['action'])))",
        ]
    )


class PodmanEnvironment(local.LocalEnvironment):
    def __init__(self, sandbox: DeepSWESandbox, template_vars: dict[str, Any]):
        super().__init__(**CONFIG["environment"])
        self.sandbox = sandbox
        self.template_vars = template_vars
        self.script = container_script()
        self.calls = 0

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump() | self.template_vars | kwargs

    async def _execute(self, action: dict[str, Any]) -> dict[str, Any]:
        try:
            with (
                anyio.fail_after(self.config.timeout + 60) as deadline,
                tempfile.TemporaryFile() as stdin,
            ):
                stdin.write(
                    json.dumps({"config": self.config.model_dump(), "action": action}).encode()
                )
                stdin.seek(0)
                result = await anyio.run_process(
                    ["podman", "exec", "-i", self.sandbox.name, "python3", "-c", self.script],
                    stdin=stdin,
                    check=False,
                )
        except TimeoutError:
            if not deadline.cancel_called:
                raise
            return {"output": "", "returncode": -1, "exception_info": "Podman exec timed out"}
        if result.returncode:
            return {
                "output": (result.stdout + result.stderr).decode(errors="replace"),
                "returncode": -1,
                "exception_info": f"Podman exec exited {result.returncode}",
            }
        return json.loads(result.stdout)

    def execute(self, action: dict[str, Any], cwd: str = "", *, timeout=None):
        assert not cwd and timeout is None
        self.calls += 1
        output = call_async(self._execute, action)
        self._check_finished(output)
        return output


async def run_agent(
    provider: Provider,
    generation: Generation,
    task: str,
    sandbox: DeepSWESandbox,
    work_dir: Path,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    metadata = await command(
        "podman",
        "exec",
        sandbox.name,
        "python3",
        "-c",
        "import os,platform,json; "
        "print(json.dumps(dict(platform.uname()._asdict(), **os.environ)))",
        timeout=30,
    )
    if metadata.exit_code:
        raise RuntimeError(f"Cannot inspect Mini-SWE sandbox: {metadata.stderr}")
    environment = PodmanEnvironment(sandbox, json.loads(metadata.stdout))
    model = ProviderModel(provider, generation)
    config = CONFIG["agent"] | {"cost_limit": 0, "output_path": work_dir / "trajectory.json"}
    del config["mode"]
    agent = DefaultAgent(model, environment, **config)
    scope = anyio.CancelScope()

    async def solve():
        with scope, anyio.move_on_after(timeout) as deadline:
            result = await anyio.to_thread.run_sync(
                agent.run, task, limiter=anyio.CapacityLimiter(1)
            )
        if scope.cancel_called:
            return None
        if deadline.cancel_called:
            result = {"exit_status": "TimeExceeded", "submission": ""}
            agent.add_messages({"role": "exit", "content": "TimeExceeded", "extra": result})
            agent.save(config["output_path"])
        return result

    worker = asyncio.create_task(solve())
    try:
        result = await asyncio.shield(worker)
    except asyncio.CancelledError:
        scope.cancel()
        await asyncio.shield(worker)
        raise
    assert result is not None
    return result["submission"], {
        "agent": AGENT,
        "agent_exit_status": result["exit_status"],
        "stop_reason": {
            "Submitted": "stop",
            "OutputBudgetExceeded": "max_tokens",
            "ContextExhausted": "context_limit",
            "TimeExceeded": "timeout",
            "RepeatedFormatError": "format_error",
        }[result["exit_status"]],
        "turns": len(model.responses),
        "tool_calls": environment.calls,
        "generated_tokens": model.tokens,
        "token_budget_kind": "cumulative_generated_tokens",
    }
