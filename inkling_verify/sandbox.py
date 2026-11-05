"""Container execution and grading for pinned DeepSWE tasks."""

from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from inkling_verify.deepswe_images import PLATFORM, solver_image


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


def memory_mb(environment: dict[str, Any]) -> int:
    value = environment["memory_mb"]
    if type(value) is not int or value <= 0:
        raise ValueError("memory_mb must be a positive integer")
    return max(8192, value)


class Container:
    def __init__(self, work_dir: Path, *, allow_network: bool = False):
        self.work_dir = work_dir
        self.allow_network = allow_network
        self.name = "inkling-verify-" + uuid.uuid4().hex
        self._active = False

    async def _podman(self, *args: str, **kwargs: Any) -> CommandResult:
        try:
            result = await command("podman", *args, **kwargs)
        except FileNotFoundError as error:
            raise RuntimeError("Podman not found; install and configure rootless Podman") from error
        if result.exit_code:
            raise RuntimeError(f"Podman {' '.join(args[:3])} failed: {result.stderr}")
        return result

    async def __aenter__(self) -> Self:
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            info = await self._podman("info", "--format", "{{.Host.Security.Rootless}}", timeout=30)
            if info.stdout.strip() != "true":
                raise RuntimeError("Rootless Podman is required; configure a rootless connection")
            await self.start()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def start(self) -> None:
        raise NotImplementedError

    async def _start_container(
        self, image: str, environment: dict[str, Any], *, platform: str | None = None
    ) -> None:
        self._active = True
        await self._podman(
            "run",
            *(["--platform", platform] if platform else []),
            "--detach",
            "--name",
            self.name,
            "--network",
            "bridge" if self.allow_network else "none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=512",
            f"--memory={memory_mb(environment)}m",
            f"--cpus={environment.get('cpus', 1)}",
            "--env",
            "CARGO_CACHE_AUTO_CLEAN_FREQUENCY=never",
            "--env",
            "TERM=xterm-256color",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "while :; do sleep 3600; done",
        )

    async def close(self) -> None:
        if self._active:
            removal = asyncio.create_task(
                self._podman("rm", "--force", "--ignore", "--time=0", self.name, timeout=60)
            )
            cancelled = None
            while not removal.done():
                try:
                    await asyncio.shield(removal)
                except asyncio.CancelledError as error:
                    cancelled = error
            removal.result()
            self._active = False
            if cancelled is not None:
                raise cancelled

    async def run(
        self,
        text: str,
        *,
        timeout: float = 300,
        output_limit: int = 16 * 1024,
        login_shell: bool = True,
    ) -> CommandResult:
        if not text or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("A command and positive finite timeout are required")
        return await command(
            "podman",
            "exec",
            self.name,
            "timeout",
            "-s",
            "KILL",
            str(timeout),
            "bash",
            "-lc" if login_shell else "-c",
            text,
            timeout=timeout + 30,
            output_limit=output_limit,
        )


class DeepSWESandbox(Container):
    def __init__(self, task: Path, work_dir: Path, *, allow_network: bool = False):
        super().__init__(work_dir, allow_network=allow_network)
        self.config = tomllib.loads((task / "task.toml").read_text())
        memory_mb(self.config["environment"])
        memory_mb(self.config["verifier"]["environment"])
        self.base_commit: str | None = None
        self.solver_image = solver_image(self.config["environment"]["docker_image"])
        self.verifier_files = tuple(
            task / "tests" / name for name in ("test.sh", "test.patch", "grader.py", "config.json")
        )
        instructions = [
            line.strip()
            for line in (task / "tests/Dockerfile").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        expected = [
            f"FROM {self.config['environment']['docker_image']}",
            *(f"COPY {path.name} /tests/{path.name}" for path in self.verifier_files),
            "RUN chmod +x /tests/test.sh",
        ]
        if instructions != expected:
            raise ValueError("Unexpected DeepSWE verifier Dockerfile recipe")
        for path in self.verifier_files:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"DeepSWE verifier input must be a regular file: {path}")

    async def start(self) -> None:
        host = await self._podman("info", "--format", "{{.Host.Arch}}", timeout=30)
        if host.stdout.strip() != "amd64":
            raise RuntimeError(
                "DeepSWE requires a native x86-64 Podman host; ARM emulation is not supported"
            )
        environment = self.config["environment"]
        await self._start_container(self.solver_image, environment, platform=PLATFORM)
        logs = self.work_dir / "logs"
        (logs / "artifacts").mkdir(parents=True)
        await self._podman("cp", str(logs), self.name + ":/")
        baseline = await self.run("git rev-parse HEAD", timeout=60)
        if baseline.exit_code or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", baseline.stdout.strip()
        ):
            raise RuntimeError("Task environment must start in a checked-out git repository")
        self.base_commit = baseline.stdout.strip()
        if not self.base_commit.startswith(self.config["metadata"]["base_commit_hash"]):
            raise RuntimeError("Pinned DeepSWE image does not match the task's base commit")

    async def grade(self) -> tuple[float, dict[str, Any]]:
        await self._podman("restart", self.name)
        assert self.base_commit is not None
        patch = await self._separate_verifier()
        verifier = self.config["verifier"]
        timeout = float(verifier.get("timeout_sec", 3600))
        deadline = asyncio.timeout(timeout)
        started = time.monotonic()
        test = None
        try:
            async with deadline:
                test = await self.run(
                    "rm -rf /logs/verifier && mkdir -p /logs/verifier && "
                    "env LOG_FILE=/logs/verifier/test-output.log bash /tests/test.sh",
                    timeout=timeout + 30,
                    output_limit=1024 * 1024,
                    login_shell=False,
                )
        except TimeoutError:
            if not deadline.expired():
                raise
            await self._podman("kill", "--signal=KILL", self.name)
        seconds = time.monotonic() - started
        if test is not None:
            (self.work_dir / "verifier.stdout").write_text(test.stdout)
            (self.work_dir / "verifier.stderr").write_text(test.stderr)
        await self._podman("cp", self.name + ":/logs/verifier", str(self.work_dir / "verifier"))
        reward = 0.0 if deadline.expired() else parse_reward(self.work_dir / "verifier")
        return reward, {
            "verifier_exit_code": test.exit_code if test is not None else None,
            "verifier_seconds": seconds,
            "verifier_timed_out": deadline.expired(),
            "patch_capture_exit_code": patch.exit_code,
            "base_commit": self.base_commit,
        }

    async def _separate_verifier(self) -> CommandResult:
        verifier = self.config["verifier"]
        hook = verifier["collect"][0]
        patch = await self.run(hook["command"], timeout=hook["timeout_sec"])
        (self.work_dir / "candidate-patch.stderr").write_text(patch.stderr)
        if patch.exit_code:
            raise RuntimeError(f"DeepSWE patch collection failed: {patch.stderr}")
        candidate = self.work_dir / "candidate.patch"
        await self._podman("cp", self.name + ":/logs/artifacts/model.patch", str(candidate))
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("DeepSWE patch collection must produce a regular file")
        await self.close()
        self.name += "-verifier"
        await self._start_container(self.solver_image, verifier["environment"], platform=PLATFORM)
        tests = self.work_dir / "tests"
        tests.mkdir()
        tests.chmod(0o755)
        for path in self.verifier_files:
            shutil.copy2(path, tests / path.name)
        test_script = tests / "test.sh"
        test_script.chmod(test_script.stat().st_mode | 0o111)
        await self._podman("cp", str(tests), self.name + ":/")
        await self._podman("cp", str(self.work_dir / "logs"), self.name + ":/")
        await self._podman("cp", str(candidate), self.name + ":/logs/artifacts/model.patch")
        return patch


def parse_reward(directory: Path) -> float:
    path = directory / "reward.json"
    if not path.is_file():
        raise RuntimeError("Verifier did not produce reward.json; inspect its logs")
    value = json.loads(path.read_text())["reward"]
    if type(value) not in (int, float) or value not in (0, 1):
        raise ValueError(f"Invalid DeepSWE binary reward: {value!r}")
    return value
