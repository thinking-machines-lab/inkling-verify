import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request

import pytest

from inkling_vendor_verifier.benchmarks import task_bundles as bundles
from inkling_vendor_verifier.benchmarks.deepswe import DeepSWE


def pack(source, archive, pin):
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(source.rglob("*")):
            output.add(path, arcname=path.relative_to(source), recursive=False)
    pin["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    pin["manifest_sha256"] = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    benchmark = DeepSWE()
    name = benchmark.name
    source = tmp_path / "source"
    tasks, hashes = [], {}
    for task_id in ("z_task", "a_task"):
        tasks.append({"id": task_id, "metadata": {"suite": name, "bundle_sha256": "original"}})
        for path, content in {
            "instruction.md": f"Fix {task_id}.",
            "task.toml": '[agent]\nnetwork_mode = "no-network"\n',
            "tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
        }.items():
            relative = f"tasks/{task_id}/{path}"
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            hashes[relative] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {"benchmark": name, "tasks": tasks, "files": hashes}
    (source / "manifest.json").write_text(json.dumps(manifest))
    archive, pin = (
        tmp_path / "tasks.tar.gz",
        {"url": "https://api.github.com/repos/example/repo/releases/assets/42"},
    )
    pack(source, archive, pin)
    monkeypatch.setattr(bundles, "release", lambda name: pin)
    monkeypatch.setattr(
        bundles, "_download", lambda url, destination: shutil.copyfile(archive, destination)
    )
    return benchmark, source, archive, manifest, pin


def test_prepare_preserves_task_order_assets_and_metadata(tmp_path, bundle):
    benchmark, source, archive, manifest, _ = bundle
    data = tmp_path / "data"
    installed = benchmark.prepare(data)
    assert installed == data / benchmark.name
    examples = benchmark.load(data)
    assert [example.id for example in examples] == ["z_task", "a_task"]
    for row, example in zip(manifest["tasks"], examples, strict=True):
        assert example.messages[0].text == f"Fix {example.id}."
        assert example.target == ""
        assert example.metadata == row["metadata"] | {
            "task_dir": str(installed / "tasks" / example.id)
        }
    for path in source.rglob("*"):
        if path.is_file():
            assert (installed / path.relative_to(source)).read_bytes() == path.read_bytes()
    assert not list(installed.rglob("solution"))
    with pytest.raises(FileExistsError):
        benchmark.prepare(data)
    assert list(data.iterdir()) == [installed]


@pytest.mark.parametrize("asset", ["instruction.md", "task.toml", "tests/test.sh", "extra.txt"])
def test_load_rejects_changed_or_extra_assets(tmp_path, bundle, asset):
    benchmark, _, archive, _, _ = bundle
    installed = benchmark.prepare(tmp_path / "data")
    (installed / "tasks/z_task" / asset).write_text("modified")
    with pytest.raises(ValueError, match="files have changed"):
        benchmark.load(tmp_path / "data")


@pytest.mark.parametrize("field", ["benchmark", "tasks", "files"])
def test_load_rejects_modified_manifest(tmp_path, bundle, field):
    benchmark, _, archive, manifest, _ = bundle
    installed = benchmark.prepare(tmp_path / "data")
    (installed / "manifest.json").write_text(json.dumps(manifest | {field: "modified"}))
    with pytest.raises(ValueError, match="manifest differs from the pinned release"):
        benchmark.load(tmp_path / "data")


@pytest.mark.parametrize("kind", ["symlink", "fifo", "missing"])
def test_load_rejects_nonregular_or_missing_assets(tmp_path, bundle, kind):
    benchmark, _, archive, _, _ = bundle
    installed = benchmark.prepare(tmp_path / "data")
    asset = installed / "tasks/z_task/instruction.md"
    asset.unlink()
    if kind == "symlink":
        asset.symlink_to(installed / "tasks/a_task/instruction.md")
    elif kind == "fifo":
        os.mkfifo(asset)
    with pytest.raises(ValueError, match="links or special files|files have changed"):
        benchmark.load(tmp_path / "data")


def test_prepare_rejects_unpinned_archive_before_extraction(tmp_path, bundle):
    benchmark, _, archive, _, _ = bundle
    archive.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="pinned release bundle"):
        benchmark.prepare(tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []


@pytest.mark.parametrize("failure", ["manifest", "assets", "benchmark"])
def test_failed_preparation_is_atomic(tmp_path, bundle, failure):
    benchmark, source, archive, manifest, pin = bundle
    if failure == "benchmark":
        manifest["benchmark"] = "another-benchmark"
        (source / "manifest.json").write_text(json.dumps(manifest))
    elif failure == "assets":
        (source / "tasks/z_task/tests/test.sh").write_text("changed grader")
    pack(source, archive, pin)
    if failure == "manifest":
        pin["manifest_sha256"] = "incorrect"
    with pytest.raises(ValueError):
        benchmark.prepare(tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("tasks/task/solution/solve.sh", tarfile.REGTYPE),
        ("tasks/task/instruction.md", tarfile.SYMTYPE),
        ("tasks/task/instruction.md", tarfile.LNKTYPE),
        ("tasks/task/instruction.md", tarfile.CHRTYPE),
    ],
)
def test_prepare_rejects_unsafe_members_and_solutions(tmp_path, bundle, name, kind):
    benchmark, _, archive, _, pin = bundle
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo(name)
        member.type, member.linkname = kind, "/etc/passwd"
        member.size = 1 if kind == tarfile.REGTYPE else 0
        output.addfile(member, io.BytesIO(b"x"))
    pin["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Unsafe task archive|reference solutions"):
        benchmark.prepare(tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []


@pytest.mark.parametrize("auth", ["GH_TOKEN", "GITHUB_TOKEN", "gh", "anonymous", "logged-out"])
def test_download_auth_and_redirects(tmp_path, monkeypatch, auth):
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    if auth in {"GH_TOKEN", "GITHUB_TOKEN"}:
        monkeypatch.setenv(auth, "test-token")
        if auth == "GH_TOKEN":
            monkeypatch.setenv("GITHUB_TOKEN", "lower-priority")
    monkeypatch.setattr(shutil, "which", lambda name: None if auth == "anonymous" else "/bin/gh")

    def login(command, **kwargs):
        assert auth in {"gh", "logged-out"}
        assert command == ["gh", "auth", "token", "--hostname", "github.com"]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 30}
        return subprocess.CompletedProcess(command, int(auth == "logged-out"), "test-token\n")

    def download(request, *, timeout):
        assert request.full_url == "https://api.github.com/repos/example/repo/releases/assets/42"
        assert timeout == 120
        assert request.get_header("Accept") == "application/octet-stream"
        expected = None if auth in {"anonymous", "logged-out"} else "Bearer test-token"
        assert request.get_header("Authorization") == expected
        redirected = urllib.request.HTTPRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "https://release-assets.githubusercontent.com/asset"
        )
        assert redirected.get_header("Authorization") is None
        return io.BytesIO(b"archive bytes")

    monkeypatch.setattr(subprocess, "run", login)
    monkeypatch.setattr(urllib.request, "urlopen", download)
    destination = tmp_path / "archive"
    bundles._download("https://api.github.com/repos/example/repo/releases/assets/42", destination)
    assert destination.read_bytes() == b"archive bytes"


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_download_http_errors_are_actionable(tmp_path, monkeypatch, status):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def fail(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, status, "Failed", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    expected = urllib.error.HTTPError if status == 500 else RuntimeError
    with pytest.raises(expected, match="500" if status == 500 else "GH_TOKEN"):
        bundles._download("https://api.github.com/asset", tmp_path / "archive")
    assert not list(tmp_path.iterdir())


def test_interrupted_download_leaves_no_partial_install(tmp_path, monkeypatch, bundle):
    benchmark, *_ = bundle

    def interrupt(url, destination):
        destination.write_bytes(b"partial archive")
        raise TimeoutError("Connection interrupted")

    monkeypatch.setattr(bundles, "_download", interrupt)
    with pytest.raises(TimeoutError):
        benchmark.prepare(tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []
