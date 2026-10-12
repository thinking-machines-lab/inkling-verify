import json
import socket
import sys
import threading
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inkling_vendor_verifier import _proxy
from inkling_vendor_verifier import sandbox as module
from inkling_vendor_verifier.sandbox import (
    CommandResult,
    DockerSandbox,
    command,
    parse_reward,
    phase_hosts,
)


@pytest.mark.parametrize("content", ["nan", "inf", "-1", "1.1", "not a score"])
def test_invalid_rewards_fail(tmp_path, content):
    (tmp_path / "reward.txt").write_text(content)
    with pytest.raises(ValueError):
        parse_reward(tmp_path)


def test_zero_is_valid_and_missing_is_failure(tmp_path):
    (tmp_path / "reward.json").write_text('{"reward": 0}')
    with pytest.raises(RuntimeError, match="did not produce"):
        parse_reward(tmp_path)
    (tmp_path / "reward.txt").write_text("0")
    assert parse_reward(tmp_path) == 0


@pytest.mark.parametrize(
    "config",
    [
        '[environment]\nenv = {FOO = "bar"}',
        '[environment]\nworkdir = "/repo"',
        '[agent]\nuser = "root"',
        '[verifier]\nenv = {FOO = "bar"}',
        '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["*.example.com"]',
    ],
)
def test_unsupported_task_configuration_fails(tmp_path, config):
    (tmp_path / "task.toml").write_text(config)
    with pytest.raises(ValueError, match="Unsupported|Invalid allowlist"):
        module.task_config(tmp_path)


def test_phase_policy_replaces_environment_policy():
    config = {
        "environment": {"network_mode": "allowlist", "allowed_hosts": ["example.com"]},
        "agent": {"network_mode": "no-network"},
        "verifier": {"network_mode": "public"},
    }
    assert phase_hosts(config, "agent") == []
    assert phase_hosts(config, "verifier") == ["*"]
    del config["agent"]["network_mode"]
    with pytest.raises(KeyError, match="network_mode"):
        phase_hosts(config, "agent")


def test_proxy_matches_domains_and_rejects_private_destinations(monkeypatch):
    assert _proxy.permitted_host("EXAMPLE.com.", ["example.com"])
    assert _proxy.permitted_host("packages.example.com", ["*"])
    assert not _proxy.permitted_host("example.com.attacker.com", ["example.com"])
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))],
    )
    with pytest.raises(PermissionError, match="public"):
        _proxy.connect("example.com", 443, ["example.com"])
    with pytest.raises(PermissionError, match="not allowed"):
        _proxy.connect("elsewhere.com", 443, ["example.com"])


def test_proxy_forwards_allowed_http_and_blocks_other_hosts(tmp_path, monkeypatch):
    class Origin(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.path.encode())

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    origin_thread = threading.Thread(target=origin.serve_forever, daemon=True)
    origin_thread.start()

    def local_connect(host, port, hosts):
        if not _proxy.permitted_host(host, hosts):
            raise PermissionError("Destination is not allowed")
        return socket.create_connection(origin.server_address)

    monkeypatch.setattr(_proxy, "connect", local_connect)
    policy = tmp_path / "hosts.json"
    policy.write_text('["allowed.example"]')
    monkeypatch.setattr(_proxy.Proxy, "policy_file", policy, raising=False)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), _proxy.Proxy)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        client = HTTPConnection(*proxy.server_address, timeout=5)
        client.request("GET", "http://allowed.example/path?q=1")
        response = client.getresponse()
        assert response.status == 200 and response.read() == b"/path?q=1"
        client.close()
        client = HTTPConnection(*proxy.server_address, timeout=5)
        client.request("GET", "http://blocked.example/path")
        response = client.getresponse()
        assert response.status == 403
        response.read()
        client.close()
    finally:
        proxy.shutdown()
        origin.shutdown()
        proxy.server_close()
        origin.server_close()
        proxy_thread.join()
        origin_thread.join()


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


async def test_docker_full_lifecycle_keeps_tests_out_until_grading(tmp_path, monkeypatch):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment/Dockerfile").write_text("FROM fixture\nWORKDIR /repo\n")
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt")
    (task / "task.toml").write_text(
        '[agent]\nnetwork_mode="no-network"\n'
        '[verifier]\nnetwork_mode="allowlist"\nallowed_hosts=["pypi.org"]\n'
    )
    calls = []
    work = tmp_path / "results"

    async def docker(*args, **kwargs):
        calls.append(args)
        if args[-1] == "git rev-parse HEAD":
            return CommandResult("a" * 40 + "\n", "", 0, 0.1)
        if args[:2] == ("docker", "cp") and args[2].endswith(":/logs/verifier"):
            destination = work / "verifier"
            destination.mkdir()
            (destination / "reward.txt").write_text("1")
        return CommandResult("", "", 0, 0.1)

    monkeypatch.setattr(module, "command", docker)
    async with DockerSandbox(task, work) as sandbox:
        policy = work / "network/allowed_hosts.json"
        assert json.loads(policy.read_text()) == []
        assert not any(args[1] == "cp" for args in calls)
        agent_run = next(a for a in calls if a[1] == "run" and sandbox.name in a)
        assert sandbox.network in agent_run and "--cap-drop=ALL" in agent_run
        assert any(a[1:4] == ("network", "create", "--internal") for a in calls)
        assert "HTTP_PROXY=http://verifier-proxy:3128" in agent_run
        assert any(flag.startswith("JAVA_TOOL_OPTIONS=-Dhttp.proxyHost=") for flag in agent_run)
        await sandbox.run("printf x", workdir="/repo")
        assert ("--workdir", "/repo") == calls[-1][2:4]
        reward, _ = await sandbox.grade()
        assert reward == 1
        assert any("a" * 40 in args[-1] and "git add -A" in args[-1] for args in calls)
        assert json.loads(policy.read_text()) == ["pypi.org"]
        assert any(a[1] == "cp" and a[2] == str(task / "tests") + "/." for a in calls)
    assert calls[-1][1:3] == ("network", "rm")
    assert (work / "candidate.patch").is_file()


async def test_failed_build_cannot_leak_a_running_container(tmp_path, monkeypatch):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment/Dockerfile").write_text("FROM fixture")
    (task / "task.toml").write_text(
        '[agent]\nnetwork_mode="no-network"\n[verifier]\nnetwork_mode="no-network"\n'
    )
    calls = []

    async def docker(*args, **kwargs):
        calls.append(args)
        return CommandResult("", "build failed", 1, 0)

    monkeypatch.setattr(module, "command", docker)
    with pytest.raises(RuntimeError, match="build failed"):
        async with DockerSandbox(task, tmp_path / "result"):
            pytest.fail("unexpected successful build")
    assert len(calls) == 1
