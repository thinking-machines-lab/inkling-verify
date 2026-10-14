"""Use the original DeepSWE environments identically on x86 and emulated ARM."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_manifest_bytes = Path(__file__).with_suffix(".json").read_bytes()
_manifest = json.loads(_manifest_bytes)
IMAGE_MANIFEST_SHA256 = hashlib.sha256(_manifest_bytes).hexdigest()
PLATFORM = _manifest["platform"]


def solver_dockerfile(configured_image: str) -> str:
    try:
        digest = _manifest["images"][configured_image]
    except KeyError:
        raise ValueError(f"No pinned DeepSWE image for {configured_image!r}") from None
    image = configured_image.rsplit(":", 1)[0] + "@" + digest
    return f"FROM {image}\nRUN mkdir -p /logs/artifacts\n"


def verifier_dockerfile(source: str, solver_image: str) -> str:
    source, count = re.subn(r"^FROM [^\n]+$", f"FROM {solver_image}", source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected one DeepSWE Dockerfile stage")
    return source
