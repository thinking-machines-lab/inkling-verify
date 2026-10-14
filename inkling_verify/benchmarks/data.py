"""Local, integrity-checked benchmark downloads; dataset contents stay out of git."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_cache(
    directory: Path, rows: list[dict[str, Any]], *, repository: str, revision: str
) -> None:
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (directory / "examples.jsonl").write_bytes(encoded)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "repository": repository,
                "revision": revision,
                "count": len(rows),
                "examples_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def read_cache(
    directory: Path, *, repository: str, revision: str, count: int
) -> list[dict[str, Any]]:
    if not (directory / "manifest.json").is_file():
        raise ValueError(f"Dataset is not prepared: {directory}; run prepare first")
    manifest = json.loads((directory / "manifest.json").read_text())
    for key, expected in (("repository", repository), ("revision", revision), ("count", count)):
        if manifest[key] != expected:
            raise ValueError(f"Dataset manifest {key} differs: {manifest[key]!r} != {expected!r}")
    encoded = (directory / "examples.jsonl").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != manifest["examples_sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {directory / 'examples.jsonl'}")
    rows = [json.loads(line) for line in encoded.splitlines()]
    if len(rows) != count or len({row["id"] for row in rows}) != count:
        raise ValueError(f"Expected {count} distinct examples in {directory}")
    return rows


def required_text(row: dict[str, Any], key: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset {key} must be a nonempty string")
    return value
