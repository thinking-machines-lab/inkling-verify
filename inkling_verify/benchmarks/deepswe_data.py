"""Pinned public DeepSWE tasks, excluding reference solutions."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from inkling_verify.types import Example, Message, Text

MANIFEST = {
    "repository": "https://github.com/datacurve-ai/deep-swe",
    "revision": "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea",
    "archive_sha256": "2c3178146ae5d5e4dce988691e90e79c608fcc9d2ea0c9f20ac96e137f73b94b",
    "files_sha256": "cb4d72ff54a419ae628069f1f3e472456df53ca782b8f80565639badf842d534",
    "count": 113,
}


def _file_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    if directory.is_symlink():
        raise ValueError("Task data cannot contain links or special files")
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Task data cannot contain links or special files")
        if path.is_file() and relative != Path("manifest.json"):
            hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _files_digest(directory: Path) -> str:
    return hashlib.sha256(json.dumps(_file_hashes(directory), sort_keys=True).encode()).hexdigest()


def prepare(data_dir: Path) -> Path:
    destination = data_dir / "deepswe"
    if destination.exists():
        raise FileExistsError(destination)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".deepswe-prepare-", dir=data_dir) as temporary:
        archive_path = Path(temporary) / "upstream.tar.gz"
        print("Downloading DeepSWE task data...", flush=True)
        url = f"https://codeload.github.com/datacurve-ai/deep-swe/tar.gz/{MANIFEST['revision']}"
        with urllib.request.urlopen(url, timeout=120) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        with archive_path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != MANIFEST["archive_sha256"]:
                raise ValueError("Task archive differs from the pinned upstream revision")
        staged = Path(temporary) / "deepswe"
        staged.mkdir()
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe task archive member: {member.name}")
                relative = path.relative_to(f"deep-swe-{MANIFEST['revision']}")
                if str(relative) not in {"LICENSE", "PROVENANCE.md"} and not (
                    len(relative.parts) >= 3
                    and relative.parts[0] == "tasks"
                    and "solution" not in relative.parts
                ):
                    continue
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsafe task archive member: {member.name}")
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
        (staged / "manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
        load(Path(temporary))
        staged.rename(destination)
    return destination


def load(data_dir: Path) -> list[Example]:
    directory = data_dir / "deepswe"
    if json.loads((directory / "manifest.json").read_text()) != MANIFEST:
        raise ValueError("Task manifest differs from the pinned upstream revision; prepare again")
    if _files_digest(directory) != MANIFEST["files_sha256"]:
        raise ValueError("Task files have changed; prepare again")
    tasks = sorted((directory / "tasks").iterdir())
    if len(tasks) != MANIFEST["count"]:
        raise ValueError(f"Expected {MANIFEST['count']} DeepSWE tasks")
    return [
        Example(
            id=task.name,
            messages=(Message("user", (Text((task / "instruction.md").read_text()),)),),
            target="",
            metadata={
                "suite": "deepswe",
                "source_repository": MANIFEST["repository"],
                "source_revision": MANIFEST["revision"],
                "task_dir": str(task),
            },
        )
        for task in tasks
    ]
