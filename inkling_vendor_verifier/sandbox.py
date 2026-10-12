"""Docker execution for versioned SWE task bundles."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float


async def command(
    *args: str, timeout: float = 3600, output_limit: int = 1024 * 1024
) -> CommandResult:
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    async def read(stream: asyncio.StreamReader) -> str:
        output = bytearray()
        truncated = False
        while chunk := await stream.read(65536):
            available = output_limit - len(output)
            output.extend(chunk[:available])
            truncated |= len(chunk) > available
        text = output.decode(errors="replace")
        return text + ("\n... [output truncated]" if truncated else "")

    assert process.stdout is not None and process.stderr is not None
    readers = asyncio.gather(read(process.stdout), read(process.stderr))
    try:
        async with asyncio.timeout(timeout):
            code = await process.wait()
            stdout, stderr = await readers
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        readers.cancel()
        try:
            await readers
        except asyncio.CancelledError:
            pass
        raise
    return CommandResult(stdout, stderr, code, time.monotonic() - started)


def phase_hosts(config: dict[str, Any], phase: str) -> list[str]:
    policy = config[phase]
    mode = policy["network_mode"]
    match mode:
        case "no-network":
            if policy.get("allowed_hosts"):
                raise ValueError("no-network cannot specify allowed_hosts")
            return []
        case "allowlist":
            hosts = policy["allowed_hosts"]
            if (
                not isinstance(hosts, list)
                or not hosts
                or not all(isinstance(h, str) for h in hosts)
            ):
                raise ValueError("An allowlist needs nonempty host names")
            if any(not re.fullmatch(r"[a-zA-Z0-9.-]+", h) for h in hosts):
                raise ValueError("Invalid allowlist host")
            return hosts
        case "public":
            return ["*"]
        case _:
            raise ValueError(f"Explicit network policy required for {phase}: {mode!r}")


def memory_mb(environment: dict[str, Any]) -> int:
    if "memory_mb" in environment:
        value = environment["memory_mb"]
        if not isinstance(value, int) or value <= 0:
            raise ValueError("memory_mb must be a positive integer")
        return max(8192, value)
    value = environment.get("memory", "8G")
    match = re.fullmatch(r"([0-9]+)([MG])", value)
    if not match:
        raise ValueError(f"Unsupported task memory: {value!r}")
    return max(8192, int(match[1]) * (1024 if match[2] == "G" else 1))


def task_config(task: Path) -> dict[str, Any]:
    config = tomllib.loads((task / "task.toml").read_text())
    unexpected = config.keys() - {"metadata", "environment", "agent", "verifier", "version"}
    if unexpected:
        raise ValueError(f"Unsupported task sections: {sorted(unexpected)}")
    supported = {
        "environment": {
            "build_timeout_sec",
            "cpus",
            "memory",
            "memory_mb",
            "storage",
            "storage_mb",
            "network_mode",
            "allowed_hosts",
        },
        "agent": {"network_mode", "allowed_hosts", "timeout_sec"},
        "verifier": {"network_mode", "allowed_hosts", "timeout_sec"},
    }
    for section, keys in supported.items():
        if extra := config.get(section, {}).keys() - keys:
            raise ValueError(f"Unsupported {section} fields: {sorted(extra)}")
    phase_hosts(config, "agent")
    phase_hosts(config, "verifier")
    memory_mb(config.get("environment", {}))
    return config


class DockerSandbox:
    def __init__(self, task: Path, work_dir: Path):
        self.task = task
        self.work_dir = work_dir
        self.config = task_config(task)
        self.name = "inkling-eval-" + uuid.uuid4().hex
        self.network = self.name + "-network"
        self.proxy = self.name + "-proxy"
        self._created: list[tuple[str, str]] = []
        self.base_commit: str | None = None

    async def _docker(self, *args: str, **kwargs: Any) -> CommandResult:
        result = await command("docker", *args, **kwargs)
        if result.exit_code:
            raise RuntimeError(f"Docker {' '.join(args[:3])} failed: {result.stderr}")
        return result

    async def __aenter__(self) -> DockerSandbox:
        try:
            await self.start()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def start(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        context = self.task / "environment"
        digest = hashlib.sha256()
        for path in sorted(context.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(context)).encode() + b"\0" + path.read_bytes())
        image = "inkling-verifier-task:" + digest.hexdigest()
        environment = self.config.get("environment", {})
        result = await self._docker(
            "build",
            "--tag",
            image,
            str(context),
            timeout=float(environment.get("build_timeout_sec", 1800)),
        )
        (self.work_dir / "build.log").write_text(result.stdout + result.stderr)
        await self._docker("network", "create", "--internal", self.network)
        self._created.append(("network", self.network))
        policy_dir = self.work_dir / "network"
        policy_dir.mkdir()
        (policy_dir / "allowed_hosts.json").write_text(
            json.dumps(phase_hosts(self.config, "agent"))
        )
        await self._docker(
            "run",
            "--detach",
            "--name",
            self.proxy,
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--pids-limit=128",
            "--memory=128m",
            "--cpus=1",
            "--network=bridge",
            "--mount",
            f"type=bind,src={Path(__file__).with_name('_proxy.py')},dst=/proxy.py,readonly",
            "--mount",
            f"type=bind,src={policy_dir.resolve()},dst=/policy,readonly",
            "python:3.12-alpine",
            "python",
            "-u",
            "/proxy.py",
            "/policy/allowed_hosts.json",
        )
        self._created.append(("container", self.proxy))
        await self._docker(
            "network", "connect", "--alias", "verifier-proxy", self.network, self.proxy
        )
        env = dict.fromkeys(
            ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"), "http://verifier-proxy:3128"
        )
        env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
        env["JAVA_TOOL_OPTIONS"] = (
            "-Dhttp.proxyHost=verifier-proxy -Dhttp.proxyPort=3128"
            " -Dhttps.proxyHost=verifier-proxy -Dhttps.proxyPort=3128"
            " -Dhttp.nonProxyHosts=localhost|127.*|[::1]"
        )
        flags = [flag for key, value in env.items() for flag in ("--env", f"{key}={value}")]
        await self._docker(
            "run",
            "--detach",
            "--name",
            self.name,
            "--network",
            self.network,
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=512",
            f"--memory={memory_mb(environment)}m",
            f"--cpus={environment.get('cpus', 1)}",
            *flags,
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "while :; do sleep 3600; done",
        )
        self._created.append(("container", self.name))
        baseline = await self.run("git rev-parse HEAD", timeout=60)
        if baseline.exit_code or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", baseline.stdout.strip()
        ):
            raise RuntimeError("Task environment must start in a checked-out git repository")
        self.base_commit = baseline.stdout.strip()

    async def close(self) -> None:
        failures = []
        while self._created:
            kind, name = self._created.pop()
            args = ("rm", "--force", name) if kind == "container" else ("network", "rm", name)
            result = await command("docker", *args, timeout=60)
            if result.exit_code:
                failures.append(result.stderr)
        if failures:
            raise RuntimeError("Docker cleanup failed: " + "\n".join(failures))

    async def run(
        self,
        text: str,
        *,
        workdir: str | None = None,
        timeout: float = 300,
        output_limit: int = 16 * 1024,
    ) -> CommandResult:
        if not text or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("A command and positive finite timeout are required")
        flags = []
        if workdir is not None:
            flags.extend(("--workdir", workdir))
        return await command(
            "docker",
            "exec",
            *flags,
            self.name,
            "timeout",
            "-s",
            "KILL",
            str(timeout),
            "bash",
            "-lc",
            text,
            timeout=timeout + 30,
            output_limit=output_limit,
        )

    async def grade(self) -> tuple[float, dict[str, Any]]:
        await self._docker("restart", self.name)
        assert self.base_commit is not None
        patch = await self.run(
            "git add -A && git --no-replace-objects diff --no-ext-diff --no-textconv "
            f"{self.base_commit} --binary",
            timeout=60,
            output_limit=16 * 1024 * 1024,
        )
        (self.work_dir / "candidate.patch").write_text(patch.stdout)
        (self.work_dir / "candidate-patch.stderr").write_text(patch.stderr)
        hosts = phase_hosts(self.config, "verifier")
        policy = self.work_dir / "network" / "allowed_hosts.json"
        replacement = policy.with_suffix(".new")
        replacement.write_text(json.dumps(hosts))
        replacement.replace(policy)
        await self._docker("restart", self.proxy)
        await self._docker("cp", str(self.task / "tests") + "/.", self.name + ":/tests")
        verifier = self.config.get("verifier", {})
        test = await self.run(
            "rm -rf /logs/verifier && mkdir -p /logs/verifier && "
            "env LOG_FILE=/logs/verifier/test-output.log bash /tests/test.sh",
            timeout=float(verifier.get("timeout_sec", 3600)),
            output_limit=1024 * 1024,
        )
        (self.work_dir / "verifier.stdout").write_text(test.stdout)
        (self.work_dir / "verifier.stderr").write_text(test.stderr)
        await self._docker("cp", self.name + ":/logs/verifier", str(self.work_dir / "verifier"))
        reward = parse_reward(self.work_dir / "verifier")
        return reward, {
            "verifier_exit_code": test.exit_code,
            "verifier_seconds": test.execution_time,
            "patch_capture_exit_code": patch.exit_code,
            "base_commit": self.base_commit,
        }


def parse_reward(directory: Path) -> float:
    if not (path := directory / "reward.txt").is_file():
        raise RuntimeError("Verifier did not produce reward.txt; inspect its logs")
    value = float(path.read_text().strip())
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid verifier reward: {value}")
    return value
