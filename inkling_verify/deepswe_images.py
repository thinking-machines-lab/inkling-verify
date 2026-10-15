"""Pinned DeepSWE environments for native x86-64 hosts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_manifest_bytes = Path(__file__).with_suffix(".json").read_bytes()
_manifest = json.loads(_manifest_bytes)
IMAGE_MANIFEST_SHA256 = hashlib.sha256(_manifest_bytes).hexdigest()
PLATFORM = _manifest["platform"]


def solver_image(configured_image: str) -> str:
    try:
        digest = _manifest["images"][configured_image]
    except KeyError:
        raise ValueError(f"No pinned DeepSWE image for {configured_image!r}") from None
    return configured_image.rsplit(":", 1)[0] + "@" + digest
