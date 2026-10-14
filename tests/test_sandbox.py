import hashlib
import json
import sys
from pathlib import Path

import pytest

from inkling_vendor_verifier import deepswe_images
from inkling_vendor_verifier import sandbox as module
from inkling_vendor_verifier.benchmarks.task_bundles import release
from inkling_vendor_verifier.deepswe_images import (
    IMAGE_MANIFEST_SHA256,
    PLATFORM,
    solver_dockerfile,
    verifier_dockerfile,
)
from inkling_vendor_verifier.sandbox import (
    CommandResult,
    DockerSandbox,
    command,
    parse_reward,
)

UPSTREAM_IMAGE = "public.ecr.aws/d3j8x8q7/swe-bench-202605:kh7d5m4ed35zfp7gyhx7wdahed82yw72-v1.1"
PINNED_IMAGE = (
    "public.ecr.aws/d3j8x8q7/swe-bench-202605@"
    "sha256:3a4d47f5281269305343c83729836ac2f3172811aee72681e472a4196178eda1"
)


@pytest.mark.parametrize("crash_sentinel", [False, True])
def test_deepswe_missing_reward_is_an_infrastructure_failure(tmp_path, crash_sentinel):
    if crash_sentinel:
        (tmp_path / "reward.txt").write_text("-1")
    with pytest.raises(RuntimeError, match="did not produce reward.json"):
        parse_reward(tmp_path)


@pytest.mark.parametrize("reward", [0, 1, 0.5, -1, True, "1", None, float("nan")])
def test_deepswe_uses_binary_structured_reward(tmp_path, reward):
    (tmp_path / "reward.json").write_text(json.dumps({"reward": reward, "f2p_passed": 3}))
    if type(reward) is int and reward in (0, 1):
        assert parse_reward(tmp_path) == reward
    else:
        with pytest.raises(ValueError, match="binary reward"):
            parse_reward(tmp_path)


@pytest.fixture
def deepswe_task(tmp_path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment/Dockerfile").write_text(
        "FROM public.ecr.aws/x8v8d7g8/mars-base:latest\nWORKDIR /app\n"
    )
    (task / "tests").mkdir(parents=True)
    (task / "tests/Dockerfile").write_text("FROM fixture:solver\nCOPY . /tests/\n")
    (task / "task.toml").write_text(
        """schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]
[metadata]
base_commit_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
[environment]
docker_image = "fixture:solver"
os = "linux"
gpus = 0
mcp_servers = []
env = {}
cpus = 2
memory_mb = 8192
[agent]
network_mode = "no-network"
timeout_sec = 10800
[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800
env = {}
[verifier.environment]
cpus = 3
memory_mb = 12288
[[verifier.collect]]
command = "git diff --binary abc123 HEAD > /logs/artifacts/model.patch"
timeout_sec = 300
[solution]
env = {}
""".replace('docker_image = "fixture:solver"', f'docker_image = "{UPSTREAM_IMAGE}"')
    )
    return task


@pytest.mark.parametrize(
    "before,after",
    [
        ('network_mode = "no-network"', 'network_mode = "public"'),
        ('environment_mode = "separate"', 'environment_mode = "same"'),
        ('os = "linux"', 'os = "windows"'),
        ("gpus = 0", "gpus = 1"),
        ("env = {}", 'env = {SECRET = "value"}'),
        ("artifacts/model.patch", "artifacts/other.patch"),
        ("mcp_servers = []", 'mcp_servers = ["service"]'),
        ("timeout_sec = 300", "timeout_sec = -1"),
        ("[verifier.environment]", '[verifier.environment]\nuser = "root"'),
    ],
)
def test_deepswe_rejects_unsupported_execution_fields(deepswe_task, before, after):
    path = deepswe_task / "task.toml"
    path.write_text(path.read_text().replace(before, after))
    with pytest.raises(ValueError, match="Unsupported|Invalid DeepSWE"):
        module.task_config(deepswe_task)


@pytest.mark.parametrize("failure", [None, "collect", "verifier-build", "patch-symlink"])
@pytest.mark.parametrize("allow_network", [False, True])
async def test_deepswe_isolates_solver_and_verifier(
    deepswe_task, tmp_path, monkeypatch, failure, allow_network
):
    work = tmp_path / "result"
    calls = []
    hook = module.task_config(deepswe_task)["verifier"]["collect"][0]["command"]
    patch_bytes = b"binary patch\x00\xff"

    async def docker(*args, **kwargs):
        calls.append(args)
        if args[-1] == "git rev-parse HEAD":
            return CommandResult("a" * 40, "", 0, 0.1)
        if (failure == "collect" and args[-1] == hook) or (
            failure == "verifier-build"
            and args[1] == "build"
            and args[-1] == str(deepswe_task / "tests")
        ):
            return CommandResult("", "injected failure", 1, 0.1)
        if args[1] == "cp" and args[2].endswith(":/logs/artifacts/model.patch"):
            candidate = work / "candidate.patch"
            if failure == "patch-symlink":
                candidate.symlink_to(work / "missing")
            else:
                candidate.write_bytes(patch_bytes)
        if args[1] == "cp" and args[2].endswith(":/logs/verifier"):
            (work / "verifier").mkdir()
            (work / "verifier/reward.json").write_text('{"reward": 1}')
        return CommandResult("", "", 0, 0.1)

    monkeypatch.setattr(module, "command", docker)
    sandbox = DockerSandbox(deepswe_task, work, allow_network=allow_network)
    solver = sandbox.name

    async def evaluate():
        async with sandbox:
            assert not any(a[1] in {"cp", "network"} for a in calls)
            builds = [a for a in calls if a[1] == "build"]
            assert len(builds) == 1
            assert builds[0][-1] == str(deepswe_task / "environment")
            await sandbox.run("printf x", workdir="/app")
            assert ("--workdir", "/app") == calls[-1][2:4]
            assert calls[-1][-3:] == ("bash", "-lc", "printf x")
            result = await sandbox.grade()
            assert result[0] == 1

    if failure:
        with pytest.raises(RuntimeError, match="injected failure|regular file"):
            await evaluate()
    else:
        await evaluate()
        assert (work / "candidate.patch").read_bytes() == patch_bytes
        verifier = solver + "-verifier"
        runs = [a for a in calls if a[1] == "run"]
        assert len(runs) == 2
        assert sandbox.solver_image in runs[0]
        assert "fixture:solver" not in runs[0]
        assert (work / "environment.Dockerfile").read_text() == (
            f"FROM {PINNED_IMAGE}\nRUN mkdir -p /logs/artifacts\n"
        )
        assert (work / "tests.Dockerfile").read_text() == (
            f"FROM {sandbox.solver_image}\nCOPY . /tests/\n"
        )
        assert (deepswe_task / "tests/Dockerfile").read_text().startswith("FROM fixture:solver\n")
        assert "--cpus=3" in runs[1] and "--memory=12288m" in runs[1]
        remove = calls.index(("docker", "rm", "--force", solver))
        assert remove < calls.index(runs[1])
        assert any(a[1] == "build" and a[-1] == str(deepswe_task / "tests") for a in calls)
        copies = [a[2:] for a in calls if a[1] == "cp"]
        assert copies == [
            (solver + ":/logs/artifacts/model.patch", str(work / "candidate.patch")),
            (str(work / "candidate.patch"), verifier + ":/logs/artifacts/model.patch"),
            (verifier + ":/logs/verifier", str(work / "verifier")),
        ]
    assert any(a[-1] == hook for a in calls)
    assert not any(a[1] == "network" or any("PROXY=" in flag for flag in a) for a in calls)
    for args in (a for a in calls if a[1] in {"build", "run"}):
        assert args[args.index("--platform") + 1] == "linux/amd64"
        if args[1] == "build":
            assert "--network=none" in args
    for args in (a for a in calls if a[1] == "run"):
        assert args[args.index("--network") + 1] == ("bridge" if allow_network else "none")
        assert "--cap-drop=ALL" in args and "--security-opt=no-new-privileges" in args
        assert "CARGO_CACHE_AUTO_CLEAN_FREQUENCY=never" in args
        assert "TERM=xterm-256color" in args
        assert not any(flag.startswith(("--mount", "--volume", "--privileged")) for flag in args)
        name = args[args.index("--name") + 1]
        assert ("docker", "rm", "--force", name) in calls
    if not failure:
        grading = next(a for a in calls if a[-1].endswith("bash /tests/test.sh"))
        assert grading[-3:-1] == ("bash", "-c")
    assert not sandbox._created


async def test_verifier_shell_preserves_image_virtualenv_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv/bin"
    bin_dir.mkdir(parents=True)
    pytest_executable = bin_dir / "pytest"
    pytest_executable.write_text("#!/bin/sh\nprintf 'image virtualenv pytest\\n'\n")
    pytest_executable.chmod(0o755)
    image_path = f"{bin_dir}:/usr/bin:/bin"
    monkeypatch.setenv("PATH", image_path)
    monkeypatch.delenv("BASH_ENV", raising=False)
    container = module.Container(tmp_path)

    async def local_exec(*args, **kwargs):
        assert args[:3] == ("docker", "exec", container.name)
        return await command(*args[3:], **kwargs)

    monkeypatch.setattr(module, "command", local_exec)
    result = await container.run('pytest && printf "%s\\n" "$PATH"', login_shell=False)
    assert result.exit_code == 0
    assert result.stdout == f"image virtualenv pytest\n{image_path}\n"


@pytest.mark.parametrize("source", ["RUN true\n", "FROM first\nFROM second\n"])
def test_deepswe_rejects_unexpected_verifier_image_stages(source):
    with pytest.raises(ValueError, match="one DeepSWE Dockerfile stage"):
        verifier_dockerfile(source, "inkling-verifier-task:123")


def test_deepswe_image_pins_cover_the_release_and_have_a_resume_identity():
    raw = Path(deepswe_images.__file__).with_suffix(".json").read_bytes()
    manifest = json.loads(raw)
    assert manifest["task_manifest_sha256"] == release("deepswe")["manifest_sha256"]
    assert len(manifest["images"]) == 113
    assert IMAGE_MANIFEST_SHA256 == hashlib.sha256(raw).hexdigest()
    assert PLATFORM == "linux/amd64"
    assert solver_dockerfile(UPSTREAM_IMAGE) == (
        f"FROM {PINNED_IMAGE}\nRUN mkdir -p /logs/artifacts\n"
    )


async def test_deepswe_rejects_unpinned_images_without_building(deepswe_task, monkeypatch):
    path = deepswe_task / "task.toml"
    path.write_text(path.read_text().replace(UPSTREAM_IMAGE, "unlisted:image"))

    async def unexpected_docker(*args, **kwargs):
        pytest.fail("Unpinned images must not reach the container engine")

    monkeypatch.setattr(module, "command", unexpected_docker)
    with pytest.raises(ValueError, match="No pinned DeepSWE image"):
        async with DockerSandbox(deepswe_task, deepswe_task / "result"):
            pytest.fail("unexpected unpinned image")


async def test_deepswe_checks_pinned_image_base_commit(deepswe_task, monkeypatch):
    async def docker(*args, **kwargs):
        return CommandResult("b" * 40 if args[-1] == "git rev-parse HEAD" else "", "", 0, 0)

    monkeypatch.setattr(module, "command", docker)
    sandbox = DockerSandbox(deepswe_task, deepswe_task / "result")
    with pytest.raises(RuntimeError, match="does not match the task's base commit"):
        async with sandbox:
            pytest.fail("unexpected mismatched image")
    assert not sandbox._created


async def test_image_tools_container_uses_native_platform(tmp_path, monkeypatch):
    calls = []

    async def docker(*args, **kwargs):
        calls.append(args)
        return CommandResult("", "", 0, 0)

    monkeypatch.setattr(module, "command", docker)
    sandbox = module.Container(tmp_path)
    await sandbox._start_container("python-tools:image", {"memory_mb": 8192})
    assert "--platform" not in calls[0]


async def test_deepswe_verifier_cache_includes_solver_image(deepswe_task, tmp_path, monkeypatch):
    calls = []

    async def docker(*args, **kwargs):
        calls.append(args)
        return CommandResult("", "", 0, 0.1)

    monkeypatch.setattr(module, "command", docker)
    sandbox = DockerSandbox(deepswe_task, tmp_path)
    first = await sandbox._image(deepswe_task / "tests", {}, "solver:first")
    second = await sandbox._image(deepswe_task / "tests", {}, "solver:second")
    assert first != second
    assert len(calls) == 2


@pytest.mark.parametrize("section", ["agent", "verifier"])
@pytest.mark.parametrize("mode", ["public", "allowlist", "host", "bridge"])
async def test_deepswe_rejects_network_access_before_start(
    deepswe_task, monkeypatch, section, mode
):
    path = deepswe_task / "task.toml"
    source = path.read_text()
    source = source.replace(
        f'[{section}]\nnetwork_mode = "no-network"',
        f'[{section}]\nnetwork_mode = "{mode}"',
    )
    path.write_text(source)

    async def unexpected_docker(*args, **kwargs):
        pytest.fail("An invalid task must not reach the container engine")

    monkeypatch.setattr(module, "command", unexpected_docker)
    with pytest.raises(ValueError, match="Unsupported DeepSWE execution"):
        async with DockerSandbox(deepswe_task, deepswe_task / "result"):
            pytest.fail("unexpected network-enabled sandbox")


@pytest.mark.parametrize("section", ["agent", "verifier", "environment"])
def test_deepswe_rejects_allowlists(deepswe_task, section):
    path = deepswe_task / "task.toml"
    path.write_text(path.read_text().replace(f"[{section}]", f"[{section}]\nallowed_hosts = []"))
    with pytest.raises(ValueError, match="Unsupported .* fields"):
        module.task_config(deepswe_task)


def test_deepswe_rejects_legacy_task_schema(deepswe_task):
    path = deepswe_task / "task.toml"
    path.write_text(path.read_text().replace('schema_version = "1.3"', 'version = "1.0"'))
    with pytest.raises(ValueError, match="Unsupported DeepSWE task schema"):
        module.task_config(deepswe_task)


async def test_command_bounds_output_while_draining_both_pipes():
    result = await command(
        sys.executable,
        "-c",
        "import sys; print('x'*200000); print('y'*200000,file=sys.stderr)",
        output_limit=100,
    )
    assert result.exit_code == 0
    assert result.stdout == "x" * 100 + "\n... [output truncated]"
    assert result.stderr == "y" * 100 + "\n... [output truncated]"


async def test_command_timeout_terminates_local_client():
    with pytest.raises(TimeoutError):
        await command(sys.executable, "-c", "import time; time.sleep(60)", timeout=0.05)


async def test_failed_build_cannot_leak_a_running_container(deepswe_task, tmp_path, monkeypatch):
    calls = []

    async def docker(*args, **kwargs):
        calls.append(args)
        return CommandResult("", "build failed", 1, 0)

    monkeypatch.setattr(module, "command", docker)
    with pytest.raises(RuntimeError, match="build failed"):
        async with DockerSandbox(deepswe_task, tmp_path / "result"):
            pytest.fail("unexpected successful build")
    assert len(calls) == 1
