"""Container execution and grading for pinned DeepSWE tasks."""

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
from typing import Any, Self

from inkling_verify.deepswe_images import (
    PLATFORM,
    solver_image,
    verifier_dockerfile,
)


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
        self._created: list[str] = []

    async def _docker(self, *args: str, **kwargs: Any) -> CommandResult:
        result = await command("docker", *args, **kwargs)
        if result.exit_code:
            raise RuntimeError(f"Docker {' '.join(args[:3])} failed: {result.stderr}")
        return result

    async def __aenter__(self) -> Self:
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
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
        await self._docker(
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
        self._created.append(self.name)

    async def close(self) -> None:
        failures = []
        while self._created:
            name = self._created.pop()
            result = await command("docker", "rm", "--force", name, timeout=60)
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
        login_shell: bool = True,
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
            "-lc" if login_shell else "-c",
            text,
            timeout=timeout + 30,
            output_limit=output_limit,
        )


class DockerSandbox(Container):
    def __init__(self, task: Path, work_dir: Path, *, allow_network: bool = False):
        super().__init__(work_dir, allow_network=allow_network)
        self.task = task
        self.config = tomllib.loads((task / "task.toml").read_text())
        memory_mb(self.config["environment"])
        memory_mb(self.config["verifier"]["environment"])
        self.base_commit: str | None = None
        self.solver_image = solver_image(self.config["environment"]["docker_image"])

    async def start(self) -> None:
        environment = self.config["environment"]
        await self._start_container(self.solver_image, environment, platform=PLATFORM)
        logs = self.work_dir / "logs"
        (logs / "artifacts").mkdir(parents=True)
        await self._docker("cp", str(logs), self.name + ":/")
        baseline = await self.run("git rev-parse HEAD", timeout=60)
        if baseline.exit_code or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", baseline.stdout.strip()
        ):
            raise RuntimeError("Task environment must start in a checked-out git repository")
        self.base_commit = baseline.stdout.strip()
        if not self.base_commit.startswith(self.config["metadata"]["base_commit_hash"]):
            raise RuntimeError("Pinned DeepSWE image does not match the task's base commit")

    async def _verifier_image(self) -> str:
        context = self.task / "tests"
        digest = hashlib.sha256()
        for path in sorted(context.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(context)).encode() + b"\0" + path.read_bytes())
        recipe = verifier_dockerfile((context / "Dockerfile").read_text(), self.solver_image)
        dockerfile = self.work_dir / "tests.Dockerfile"
        dockerfile.write_text(recipe)
        digest.update(recipe.encode())
        image = "inkling-verify-task:" + digest.hexdigest()
        result = await self._docker(
            "build",
            "--platform",
            PLATFORM,
            "--network=none",
            "--tag",
            image,
            "--file",
            str(dockerfile),
            str(context),
            timeout=float(self.config["verifier"]["environment"].get("build_timeout_sec", 1800)),
        )
        (self.work_dir / "verifier-build.log").write_text(result.stdout + result.stderr)
        return image

    async def grade(self) -> tuple[float, dict[str, Any]]:
        await self._docker("restart", self.name)
        assert self.base_commit is not None
        patch = await self._separate_verifier()
        verifier = self.config["verifier"]
        test = await self.run(
            "rm -rf /logs/verifier && mkdir -p /logs/verifier && "
            "env LOG_FILE=/logs/verifier/test-output.log bash /tests/test.sh",
            timeout=float(verifier.get("timeout_sec", 3600)),
            output_limit=1024 * 1024,
            login_shell=False,
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

    async def _separate_verifier(self) -> CommandResult:
        verifier = self.config["verifier"]
        hook = verifier["collect"][0]
        patch = await self.run(hook["command"], timeout=hook["timeout_sec"])
        (self.work_dir / "candidate-patch.stderr").write_text(patch.stderr)
        if patch.exit_code:
            raise RuntimeError(f"DeepSWE patch collection failed: {patch.stderr}")
        candidate = self.work_dir / "candidate.patch"
        await self._docker("cp", self.name + ":/logs/artifacts/model.patch", str(candidate))
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("DeepSWE patch collection must produce a regular file")
        await self._docker("rm", "--force", self.name)
        self._created.remove(self.name)
        self.name += "-verifier"
        image = await self._verifier_image()
        await self._start_container(image, verifier["environment"], platform=PLATFORM)
        await self._docker("cp", str(self.work_dir / "logs"), self.name + ":/")
        await self._docker("cp", str(candidate), self.name + ":/logs/artifacts/model.patch")
        return patch


def parse_reward(directory: Path) -> float:
    path = directory / "reward.json"
    if not path.is_file():
        raise RuntimeError("Verifier did not produce reward.json; inspect its logs")
    value = json.loads(path.read_text())["reward"]
    if type(value) not in (int, float) or value not in (0, 1):
        raise ValueError(f"Invalid DeepSWE binary reward: {value!r}")
    return value
