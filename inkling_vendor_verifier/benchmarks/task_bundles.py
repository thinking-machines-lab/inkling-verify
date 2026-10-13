"""Install and load pinned coding tasks without reference solutions."""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from importlib.resources import files
from pathlib import Path, PurePosixPath

from inkling_vendor_verifier.types import Example, Message, Text


def release(benchmark: str) -> dict[str, str]:
    return json.loads(files(__package__).joinpath("task_bundles.json").read_text())[benchmark]


def _file_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    if directory.is_symlink():
        raise ValueError("Task bundles cannot contain links or special files")
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Task bundles cannot contain links or special files")
        if "solution" in relative.parts:
            raise ValueError("Task bundles cannot contain reference solutions")
        if path.is_file() and relative != Path("manifest.json"):
            hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _manifest(directory: Path, benchmark: str) -> dict:
    hashes = _file_hashes(directory)
    raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != release(benchmark)["manifest_sha256"]:
        raise ValueError("Task manifest differs from the pinned release")
    manifest = json.loads(raw)
    if manifest["benchmark"] != benchmark:
        raise ValueError("Task bundle has the wrong benchmark")
    if manifest["files"] != hashes:
        raise ValueError("Task files have changed; prepare the bundle again")
    return manifest


def prepare(data_dir: Path, benchmark: str, bundle: Path) -> Path:
    with bundle.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    if checksum != release(benchmark)["sha256"]:
        raise ValueError("Task archive differs from the pinned release bundle")
    destination = data_dir / benchmark
    if destination.exists():
        raise FileExistsError(destination)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tasks-prepare-", dir=data_dir) as temporary:
        staged = Path(temporary) / benchmark
        staged.mkdir()
        with tarfile.open(bundle) as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"Unsafe task archive member: {member.name}")
            archive.extractall(staged, filter="data")
        _manifest(staged, benchmark)
        staged.rename(destination)
    return destination


def load(data_dir: Path, benchmark: str) -> list[Example]:
    directory = data_dir / benchmark
    manifest = _manifest(directory, benchmark)
    return [
        Example(
            id=row["id"],
            messages=(
                Message(
                    "user",
                    (Text((directory / "tasks" / row["id"] / "instruction.md").read_text()),),
                ),
            ),
            target="",
            metadata=row["metadata"] | {"task_dir": str(directory / "tasks" / row["id"])},
        )
        for row in manifest["tasks"]
    ]
