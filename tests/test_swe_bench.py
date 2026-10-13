import asyncio
import hashlib
import io
import json
import tarfile

import pytest

from inkling_vendor_verifier.benchmarks import swe_bench as swe
from inkling_vendor_verifier.sandbox import CommandResult
from inkling_vendor_verifier.types import (
    Example,
    Generation,
    Message,
    Response,
    Text,
    ToolCall,
    Usage,
)


class Sandbox:
    config = {"agent": {"network_mode": "no-network"}, "verifier": {"network_mode": "no-network"}}

    def __init__(self, *args):
        self.commands = []
        self.graded = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, command, **kwargs):
        self.commands.append(command)
        return CommandResult("ok", "", 0, 0.01)

    async def grade(self):
        self.graded = True
        return 1, {"verifier_exit_code": 0}


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
def task(monkeypatch):
    sandbox = Sandbox()
    monkeypatch.setattr(swe, "DockerSandbox", lambda *args: sandbox)
    example = Example(
        "task",
        (Message("user", (Text("fix the bug"),)),),
        "",
        {"task_dir": "/task", "suite": "verified"},
    )
    return example, sandbox


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
@pytest.mark.parametrize("subset", ["signal", "deepswe"])
async def test_limits_preserve_each_benchmarks_grading_contract(
    task, tmp_path, monkeypatch, limit, tool_count, executed, subset
):
    example, sandbox = task
    monkeypatch.setattr(swe, "MAX_TURNS", 1 if limit == "max_turns" else 2)
    monkeypatch.setattr(swe, "MAX_TOOL_CALLS", 1 if limit == "max_tool_calls" else 2)
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
        sandbox.config = sandbox.config | {"agent": sandbox.config["agent"] | {"timeout_sec": 0}}
        generate = provider.generate

        async def wait_for_deadline(*args, **kwargs):
            await generate(*args, **kwargs)
            await asyncio.sleep(0)
            pytest.fail("Agent deadline did not cancel generation")

        monkeypatch.setattr(provider, "generate", wait_for_deadline)
    result = await swe.SWEBench(subset).evaluate(example, provider, Generation(30), tmp_path)
    assert result.score == (1 if subset == "deepswe" else 0)
    assert result.details["verifier_reward"] == 1
    assert result.details["stop_reason"] == ("max_tokens" if limit == "length" else limit)
    assert result.details["turns"] == len(provider.calls) == 1
    assert result.details["tool_calls"] == executed
    assert sandbox.commands == ["echo ok"] * executed
    assert sandbox.graded


async def test_bad_tool_arguments_are_model_feedback_not_host_commands(task, tmp_path):
    example, sandbox = task
    provider = Provider(
        [
            reply(calls=(ToolCall("call", "bash", '{"command":"bad","timeout":true}'),)),
            reply("done"),
        ]
    )
    await swe.SWEBench().evaluate(example, provider, Generation(30), tmp_path)
    assert not sandbox.commands
    assert "positive integer" in provider.calls[1][0][-1].text


async def test_provider_failure_is_not_a_zero_score(task, tmp_path):
    example, sandbox = task

    class BrokenProvider:
        async def generate(self, *args, **kwargs):
            raise TimeoutError("API failed")

    with pytest.raises(TimeoutError, match="API failed"):
        await swe.SWEBench().evaluate(example, BrokenProvider(), Generation(30), tmp_path)
    assert not sandbox.graded


@pytest.mark.parametrize(
    "name,kind,error",
    [
        ("../escape", "file", tarfile.OutsideDestinationError),
        ("/absolute", "file", ValueError),
        ("tasks/task/solution/solve.sh", "file", ValueError),
        ("tasks/task/instruction.md", "symlink", ValueError),
        ("tasks/task/instruction.md", "hardlink", ValueError),
    ],
)
def test_prepare_rejects_unsafe_or_solution_members(tmp_path, monkeypatch, name, kind, error):
    monkeypatch.setattr(swe, "_validate", lambda path: None)
    archive = tmp_path / "bundle.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(name)
        if kind != "file":
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "/etc/passwd"
            tar.addfile(member)
        else:
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
    monkeypatch.setattr(
        swe,
        "release",
        lambda: {
            "bundle_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "files_sha256": swe._manifest_digest({}),
        },
    )
    with pytest.raises(error):
        swe.prepare(tmp_path / "data", archive)
    assert not list((tmp_path / "data").iterdir())


def test_prepare_rejects_archive_before_extraction(tmp_path):
    archive = tmp_path / "untrusted.tar"
    archive.write_bytes(b"not the pinned release")
    with pytest.raises(ValueError, match="pinned release bundle"):
        swe.prepare(tmp_path / "data", archive)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "mutate, error",
    [(False, "differs from the pinned release"), (True, "files have changed")],
)
@pytest.mark.parametrize("name", ["instruction.md", "task.toml"])
def test_load_rejects_noncanonical_or_changed_assets(tmp_path, monkeypatch, mutate, error, name):
    monkeypatch.setattr(swe, "_validate", lambda path: None)
    directory = tmp_path / "swe-bench-signal"
    directory.mkdir()
    asset = directory / name
    asset.write_text("original")
    (directory / "manifest.json").write_text(json.dumps({"files": swe._file_hashes(directory)}))
    if mutate:
        asset.write_text("changed")
    with pytest.raises(ValueError, match=error):
        swe.SWEBench().load(tmp_path)


def test_selection_matches_pinned_suite_counts():
    from collections import Counter

    rows = swe.selection()
    assert swe._manifest_digest(rows) == (
        "74c822e4d80df81118e52f99ff8e2b0df4c282a475d8bf5bbf26d36b15abdc4a"
    )
    assert len(rows) == len({row["id"] for row in rows}) == 250
    assert Counter(row["suite"] for row in rows) == {
        "swebench-verified": 63,
        "swebench-pro": 121,
        "swebench-multilingual": 66,
    }
