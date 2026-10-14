import hashlib
import io
import json
import os
import tarfile
import urllib.request

import pytest

from inkling_verify.benchmarks import deepswe_data as data
from inkling_verify.benchmarks.deepswe import DeepSWE


@pytest.fixture
def archive(tmp_path, monkeypatch):
    files = {"LICENSE": b"License text\n", "PROVENANCE.md": b"Public task provenance\n"}
    for task_id in ("z_task", "a_task"):
        for path, content in {
            "instruction.md": f"Fix {task_id}.\n".encode(),
            "task.toml": b'[agent]\nnetwork_mode = "no-network"\n',
            "tests/test.sh": b"echo 1 > /logs/verifier/reward.txt\n",
            "environment/fixture.bin": b"\x00\xff\r\n",
        }.items():
            files[f"tasks/{task_id}/{path}"] = content
    pin = data.MANIFEST | {
        "revision": "a" * 40,
        "count": 2,
        "files_sha256": hashlib.sha256(
            json.dumps(
                {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    skipped = {
        "README.md": b"Not evaluation data\n",
        "manifest.json": b"Not the installed manifest\n",
        "tasks/metadata.json": b"{}\n",
        "tasks/z_task/solution/solve.sh": b"reference solution\n",
        "tasks/a_task/tests/solution/reference.txt": b"nested reference solution\n",
    }
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        for name, content in (files | skipped).items():
            member = tarfile.TarInfo(f"deep-swe-{pin['revision']}/{name}")
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
        for name, target in (
            ("tasks/README.md", "../README.md"),
            ("tasks/z_task/solution/link", "../instruction.md"),
        ):
            member = tarfile.TarInfo(f"deep-swe-{pin['revision']}/{name}")
            member.type, member.linkname = tarfile.SYMTYPE, target
            output.addfile(member)
    pin["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []

    def download(url, *, timeout):
        calls.append((url, timeout))
        return path.open("rb")

    monkeypatch.setattr(data, "MANIFEST", pin)
    monkeypatch.setattr(urllib.request, "urlopen", download)
    return path, pin, files, calls


def test_prepare_anonymously_preserves_task_bytes_and_sorts_examples(
    tmp_path, monkeypatch, archive
):
    _, pin, files, calls = archive
    monkeypatch.setenv("GH_TOKEN", "must-not-send-gh-token")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-send-github-token")
    benchmark = DeepSWE()
    directory = tmp_path / "data"
    installed = benchmark.prepare(directory)
    assert installed == directory / "deepswe"
    assert calls == [
        (f"https://codeload.github.com/datacurve-ai/deep-swe/tar.gz/{pin['revision']}", 120)
    ]
    assert json.loads((installed / "manifest.json").read_text()) == pin
    assert data._file_hashes(installed) == {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
    assert data._files_digest(installed) == pin["files_sha256"]
    for name, content in files.items():
        assert (installed / name).read_bytes() == content
    assert not os.path.lexists(installed / "tasks/README.md")
    assert not list(installed.rglob("solution"))
    examples = benchmark.load(directory)
    assert [example.id for example in examples] == ["a_task", "z_task"]
    for example in examples:
        assert example.messages[0].text == f"Fix {example.id}.\n"
        assert example.target == ""
        assert example.metadata == {
            "suite": "deepswe",
            "source_repository": pin["repository"],
            "source_revision": pin["revision"],
            "task_dir": str(installed / "tasks" / example.id),
        }
    with pytest.raises(FileExistsError):
        benchmark.prepare(directory)
    assert len(calls) == 1
    assert list(directory.iterdir()) == [installed]


@pytest.mark.parametrize("change", ["modified", "extra", "missing", "symlink", "fifo"])
def test_load_rejects_changed_files(tmp_path, archive, change):
    installed = data.prepare(tmp_path / "data")
    path = installed / "tasks/z_task/instruction.md"
    if change == "modified":
        path.write_bytes(b"changed instruction")
    elif change == "extra":
        (installed / "extra.txt").write_text("extra")
    else:
        path.unlink()
        if change == "symlink":
            path.symlink_to(installed / "tasks/a_task/instruction.md")
        elif change == "fifo":
            os.mkfifo(path)
    expected = "links or special files" if change in {"symlink", "fifo"} else "files have changed"
    with pytest.raises(ValueError, match=expected):
        data.load(tmp_path / "data")


@pytest.mark.parametrize(
    "field", ["repository", "revision", "archive_sha256", "files_sha256", "count"]
)
def test_load_rejects_changed_manifest(tmp_path, archive, field):
    _, pin, _, _ = archive
    installed = data.prepare(tmp_path / "data")
    (installed / "manifest.json").write_text(json.dumps(pin | {field: "changed"}))
    with pytest.raises(ValueError, match="manifest differs from the pinned upstream revision"):
        data.load(tmp_path / "data")


def test_load_requires_manifest(tmp_path, archive):
    installed = data.prepare(tmp_path / "data")
    (installed / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        data.load(tmp_path / "data")


@pytest.mark.parametrize(
    "failure,error",
    [("archive", "archive differs"), ("files", "files have changed"), ("count", "Expected 3")],
)
def test_failed_preparation_is_atomic(tmp_path, archive, failure, error):
    path, pin, _, _ = archive
    if failure == "archive":
        path.write_bytes(b"not the pinned archive")
    elif failure == "files":
        pin["files_sha256"] = "incorrect"
    else:
        pin["count"] += 1
    with pytest.raises(ValueError, match=error):
        data.prepare(tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("tasks/z_task/../../escape", tarfile.REGTYPE),
        ("tasks/z_task/instruction.md", tarfile.SYMTYPE),
        ("tasks/z_task/instruction.md", tarfile.LNKTYPE),
        ("tasks/z_task/instruction.md", tarfile.CHRTYPE),
        ("LICENSE", tarfile.SYMTYPE),
    ],
)
def test_prepare_rejects_unsafe_paths_and_selected_links_or_special_files(
    tmp_path, archive, name, kind
):
    path, pin, _, _ = archive
    with tarfile.open(path, "w:gz") as output:
        full_name = name if name.startswith("/") else f"deep-swe-{pin['revision']}/{name}"
        member = tarfile.TarInfo(full_name)
        member.type, member.linkname = kind, "/etc/passwd"
        member.size = 1 if kind == tarfile.REGTYPE else 0
        output.addfile(member, io.BytesIO(b"x"))
    pin["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Unsafe task archive member"):
        data.prepare(tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []
    assert not (tmp_path / "escape").exists()


def test_interrupted_download_leaves_no_partial_install(tmp_path, monkeypatch, archive):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise TimeoutError("Connection interrupted")
            return super().read(size)

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: Interrupted(b"partial archive")
    )
    with pytest.raises(TimeoutError, match="Connection interrupted"):
        data.prepare(tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []
