"""Shared bash agent loop for executable coding benchmarks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

from inkling_vendor_verifier.sandbox import DockerSandbox, phase_hosts
from inkling_vendor_verifier.types import (
    Evaluation,
    Example,
    Generation,
    Message,
    Provider,
    Text,
    Tool,
)

MAX_TURNS = 4096
MAX_TOOL_CALLS = 4096


def bash_tool(hosts: list[str]) -> Tool:
    description = (
        "Executes a bash command in the SandboxInterface. Returns stdout, stderr "
        "(truncated to 16KB each), exit code, and execution time."
    )
    if not hosts:
        description += "\n\nCommands run with this tool have no internet access."
    elif hosts != ["*"]:
        description += (
            "\n\nCommands run with this tool can only reach a restricted set of allowlisted "
            "internet hosts."
        )
    return Tool(
        name="bash",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command (default: container WORKDIR)",
                },
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 300)"},
            },
            "required": ["command"],
        },
    )


class CodeBenchmark:
    zero_reward_on_limit = True

    async def evaluate(
        self,
        example: Example,
        provider: Provider,
        generation: Generation,
        work_dir: Path,
        *,
        judge: Provider | None = None,
    ) -> Evaluation:
        messages = list(example.messages)
        remaining = generation.max_tokens
        calls = 0
        turns = 0
        stop = "max_turns"
        answer = ""
        async with DockerSandbox(Path(example.metadata["task_dir"]), work_dir) as sandbox:
            tool = bash_tool(phase_hosts(sandbox.config, "agent"))
            deadline = asyncio.timeout(
                float(sandbox.config.get("agent", {}).get("timeout_sec", 28800))
            )
            try:
                async with deadline:
                    for turns in range(1, MAX_TURNS + 1):
                        response = await provider.generate(
                            messages, replace(generation, max_tokens=remaining), tools=(tool,)
                        )
                        if response.usage.output_tokens <= 0:
                            raise RuntimeError(
                                "SWE generation requires positive reported output token usage"
                            )
                        remaining -= response.usage.output_tokens
                        messages.append(response.message)
                        answer = response.text
                        if response.stop_reason == "length" or remaining <= 0:
                            stop = "max_tokens"
                            break
                        if not response.message.tool_calls:
                            stop = response.stop_reason
                            break
                        if calls + len(response.message.tool_calls) > MAX_TOOL_CALLS:
                            stop = "max_tool_calls"
                            break
                        for call in response.message.tool_calls:
                            calls += 1
                            try:
                                if call.name != "bash":
                                    raise ValueError(f"Unknown tool: {call.name}")
                                args = json.loads(call.arguments)
                                if not isinstance(args, dict) or args.keys() - {
                                    "command",
                                    "workdir",
                                    "timeout",
                                }:
                                    raise ValueError(
                                        "Expected bash command, optional workdir and timeout"
                                    )
                                if not isinstance(args.get("command"), str) or not args["command"]:
                                    raise ValueError("command must be a nonempty string")
                                if "workdir" in args and not isinstance(args["workdir"], str):
                                    raise ValueError("workdir must be a string")
                                timeout = args.get("timeout", 300)
                                if type(timeout) is not int or timeout <= 0:
                                    raise ValueError(
                                        "timeout must be a positive integer in seconds"
                                    )
                            except (ValueError, TypeError) as error:
                                result_text = json.dumps({"error": str(error)})
                            else:
                                result = await sandbox.run(
                                    args["command"], workdir=args.get("workdir"), timeout=timeout
                                )
                                result_text = json.dumps(asdict(result))
                            messages.append(
                                Message("tool", (Text(result_text),), tool_call_id=call.id)
                            )
                        if calls == MAX_TOOL_CALLS:
                            stop = "max_tool_calls"
                            break
            except TimeoutError:
                if not deadline.expired():
                    raise
                stop = "timeout"
            reward, grade = await sandbox.grade()
            return Evaluation(
                score=0.0
                if self.zero_reward_on_limit
                and stop in {"max_tokens", "max_turns", "max_tool_calls", "timeout"}
                else reward,
                answer=answer,
                details={
                    "suite": example.metadata["suite"],
                    "stop_reason": stop,
                    "turns": turns,
                    "tool_calls": calls,
                    "generated_tokens": generation.max_tokens - remaining,
                    "token_budget_kind": "cumulative_generated_tokens",
                    "verifier_reward": reward,
                    "solve_allowed_hosts": phase_hosts(sandbox.config, "agent"),
                    "verifier_allowed_hosts": phase_hosts(sandbox.config, "verifier"),
                    **grade,
                },
            )
