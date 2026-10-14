"""Install and load pinned coding tasks without reference solutions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path, PurePosixPath

from inkling_verify.types import Example, Message, Text


def release() -> dict[str, str]:
    return json.loads(files(__package__).joinpath("task_bundles.json").read_text())["deepswe"]


def _download(url: str, destination: Path) -> None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token and shutil.which("gh"):
        auth = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if auth.returncode == 0:
            token = auth.stdout.strip()
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            destination.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as error:
        if error.code not in {401, 403, 404}:
            raise
        raise RuntimeError(
            f"Task download failed (HTTP {error.code}). For private release access, set GH_TOKEN "
            "or GITHUB_TOKEN, or use gh auth login with an account that can read this repository."
        ) from error


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


def _manifest(directory: Path) -> dict:
    hashes = _file_hashes(directory)
    raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != release()["manifest_sha256"]:
        raise ValueError("Task manifest differs from the pinned release")
    manifest = json.loads(raw)
    if manifest["benchmark"] != "deepswe":
        raise ValueError("Task bundle has the wrong benchmark")
    if manifest["files"] != hashes:
        raise ValueError("Task files have changed; prepare the bundle again")
    return manifest


def prepare(data_dir: Path) -> Path:
    destination = data_dir / "deepswe"
    if destination.exists():
        raise FileExistsError(destination)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tasks-prepare-", dir=data_dir) as temporary:
        pin = release()
        bundle = Path(temporary) / "tasks.tar.gz"
        print("Downloading DeepSWE task data...", flush=True)
        _download(pin["url"], bundle)
        with bundle.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != pin["sha256"]:
            raise ValueError("Task archive differs from the pinned release bundle")
        staged = Path(temporary) / "deepswe"
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
        _manifest(staged)
        staged.rename(destination)
    return destination


def load(data_dir: Path) -> list[Example]:
    directory = data_dir / "deepswe"
    manifest = _manifest(directory)
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
