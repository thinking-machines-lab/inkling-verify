"""Install pinned public Harbor SWE-bench task suites without reference solutions."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import tomllib
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from inkling_vendor_verifier.sandbox import task_config
from inkling_vendor_verifier.types import Example, Message, Text

REGISTRY_REVISION = "5fcf07f5af37eba19db389c7d8916743e4591ff5"
REGISTRY_URL = (
    f"https://raw.githubusercontent.com/harbor-framework/harbor/{REGISTRY_REVISION}/registry.json"
)
REGISTRY_SHA256 = "da1446bce05eabbd72a25eb9eef5a2f5db94645ce88c28e2497581433b3d2e60"
SOURCE_TREES = {
    "86723674f04e4209ac479d0fb75d9d9f44b4377e": "a7a1bf388872b81824b35554c27b5e8a1fc273e9",
    "c8e8f3fac7097accaacf261d74c3d6f441de45b1": "b3131e104b5e6c47386680d932463163b85e9b99",
    "61366953049fd63a7dff54f67193bf39f5d1cde6": "3460931640b0efd1af798a626de8a0beed18ceef",
    "19fb29e2b4d1e24615f620541b1a91461c00f5c5": "e31d13cf52166a2819d6295b54c32751cb904270",
}
SUITES = {
    "verified": (
        "swebench-verified",
        500,
        "b568ba7ee1209d477059a30bbb96920614ae5cfe0576963b106586dcf3bf8394",
    ),
    "pro": (
        "swebenchpro",
        731,
        "ddd9f5bb27e2299ff1ecb641023749cfd9a6e9d0654b0b2883a9f0a82955e8a2",
    ),
    "multilingual": (
        "swebench_multilingual",
        300,
        "92fa8885194cf47f0e5f2a395a74493c00f5ddbf7dd98dc3f375369f1890e9ce",
    ),
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_selection(subset: str, tasks: list[dict[str, str]]) -> None:
    _, count, checksum = SUITES[subset]
    if len(tasks) != count or _digest(tasks) != checksum:
        raise ValueError(f"Task selection differs from the pinned {subset} source suite")


def source_selection(subset: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(REGISTRY_URL, timeout=120) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != REGISTRY_SHA256:
        raise ValueError("Harbor registry differs from its pinned checksum")
    matches = [item for item in json.loads(raw) if item["name"] == SUITES[subset][0]]
    if len(matches) != 1 or matches[0]["version"] != "1.0":
        raise ValueError("Expected exactly one pinned version 1.0 source suite")
    tasks = matches[0]["tasks"]
    _validate_selection(subset, tasks)
    return tasks


def _download_tasks(destination: Path, tasks: list[dict[str, str]], revision: str) -> None:
    tree_sha = SOURCE_TREES[revision]
    tree_url = f"https://api.github.com/repos/laude-institute/harbor-datasets/git/trees/{tree_sha}?recursive=1"
    with urllib.request.urlopen(tree_url, timeout=120) as response:
        tree = json.load(response)
    if tree["truncated"] or tree["sha"] != tree_sha:
        raise ValueError("Expected the complete pinned SWE source tree")
    selected = {task["name"]: task["path"] for task in tasks}
    assets = []
    seen: set[str] = set()
    for entry in tree["tree"]:
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"Unsafe source tree path: {entry['path']}")
        if len(path.parts) < 2 or path.parts[0] not in selected:
            continue
        if path.parts[1] not in {"instruction.md", "task.toml", "environment", "tests"}:
            continue
        if entry["type"] == "tree":
            continue
        if entry["type"] != "blob" or entry["mode"] not in {"100644", "100755"}:
            raise ValueError(f"Source links and special assets are forbidden: {path}")
        if path.as_posix() in seen:
            raise ValueError(f"Duplicate source asset: {path}")
        seen.add(path.as_posix())
        assets.append(entry)

    def download(entry: dict) -> None:
        path = PurePosixPath(entry["path"])
        source_path = selected[path.parts[0]] + "/" + PurePosixPath(*path.parts[1:]).as_posix()
        url = (
            "https://raw.githubusercontent.com/laude-institute/harbor-datasets/"
            f"{revision}/{source_path}"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if len(raw) != entry["size"] or blob != entry["sha"]:
            raise ValueError(f"Source asset differs from its pinned git blob: {source_path}")
        target = destination / "tasks" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(int(entry["mode"], 8) & 0o777)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(download, assets):
            pass


def _normalize_network(task: Path) -> None:
    path = task / "task.toml"
    original = path.read_text()
    config = tomllib.loads(original)
    if any(
        key in config.get(section, {})
        for section in ("environment", "agent", "verifier")
        for key in ("network_mode", "allowed_hosts", "allow_internet")
    ):
        raise ValueError("Expected the pinned upstream Harbor default public network policy")
    for section in ("agent", "verifier"):
        heading = f"[{section}]"
        if section in config:
            original, count = re.subn(
                rf"(?m)^\[{section}\][ \t]*$", heading + '\nnetwork_mode = "public"', original
            )
            if count != 1:
                raise ValueError(f"Expected exactly one {section} heading in {path}")
        else:
            original += f'\n{heading}\nnetwork_mode = "public"\n'
    path.write_text(original)


def _file_hashes(directory: Path) -> dict[str, str]:
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Task bundles cannot contain symbolic links")
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file() and path != directory / "manifest.json"
    }


def _validate_tasks(directory: Path, names: list[str]) -> None:
    expected = set(names)
    actual = {path.name for path in (directory / "tasks").iterdir()}
    if actual != expected:
        raise ValueError(
            f"Wrong SWE selection: missing={expected - actual}, extra={actual - expected}"
        )
    for name in names:
        task = directory / "tasks" / name
        for required in ("instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"):
            if not (task / required).is_file():
                raise ValueError(f"Missing task asset: {name}/{required}")
        task_config(task)


def prepare_source(data_dir: Path, subset: str) -> Path:
    destination = data_dir / f"swe-bench-{subset}"
    if destination.exists():
        raise FileExistsError(destination)
    tasks = source_selection(subset)
    revisions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        revisions[task["git_commit_id"]].append(task)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swe-source-", dir=data_dir) as temporary:
        staged = Path(temporary) / destination.name
        (staged / "tasks").mkdir(parents=True)
        for revision, rows in revisions.items():
            _download_tasks(staged, rows, revision)
        for task in tasks:
            _normalize_network(staged / "tasks" / task["name"])
        _validate_tasks(staged, [task["name"] for task in tasks])
        manifest = {
            "benchmark": destination.name,
            "subset": subset,
            "registry_revision": REGISTRY_REVISION,
            "registry_sha256": REGISTRY_SHA256,
            "tasks": tasks,
            "network_policy": "upstream-default-public",
            "files": _file_hashes(staged),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(destination)
    return destination


def load_source(data_dir: Path, subset: str) -> list[Example]:
    directory = data_dir / f"swe-bench-{subset}"
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {
        "benchmark": directory.name,
        "subset": subset,
        "registry_revision": REGISTRY_REVISION,
        "registry_sha256": REGISTRY_SHA256,
        "network_policy": "upstream-default-public",
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("Installed SWE source manifest has the wrong benchmark or source identity")
    tasks = manifest["tasks"]
    _validate_selection(subset, tasks)
    _validate_tasks(directory, [task["name"] for task in tasks])
    hashes = _file_hashes(directory)
    if manifest["files"] != hashes:
        raise ValueError("Installed SWE source task files have changed; prepare the source again")
    bundle_digest = _digest(hashes)
    return [
        Example(
            id=row["name"],
            messages=(
                Message(
                    "user",
                    (Text((directory / "tasks" / row["name"] / "instruction.md").read_text()),),
                ),
            ),
            target="",
            metadata={
                "task_dir": str(directory / "tasks" / row["name"]),
                "suite": SUITES[subset][0],
                "subset": subset,
                "bundle_sha256": bundle_digest,
                "source_repository": row["git_url"],
                "source_revision": row["git_commit_id"],
            },
        )
        for row in tasks
    ]
