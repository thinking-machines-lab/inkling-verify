"""Build the pinned DeepSWE recipes natively on ARM64 and x86-64."""

from __future__ import annotations

import re

BASE_IMAGE = (
    "public.ecr.aws/x8v8d7g8/mars-base@"
    "sha256:91db850db926024eed328c4bf519d54986bc10aad75302cbb074f8e9d79b4c46"
)


def dockerfile(source: str, solver_image: str | None = None) -> str:
    if solver_image is None:
        base = "public.ecr.aws/x8v8d7g8/mars-base:latest"
        if not source.startswith(f"FROM {base}\n"):
            raise ValueError("Unexpected DeepSWE base image")
        image = BASE_IMAGE
        source = source.replace(
            "deno-x86_64-unknown-linux-gnu.zip", "deno-$(uname -m)-unknown-linux-gnu.zip"
        ).replace(
            '"https://get.nexte.st/${NEXTEST_VERSION}/linux" | tar zxf -',
            '"https://github.com/nextest-rs/nextest/releases/download/'
            "cargo-nextest-${NEXTEST_VERSION}/"
            'cargo-nextest-${NEXTEST_VERSION}-$(uname -m)-unknown-linux-gnu.tar.gz"'
            " | tar --no-same-owner -zxf -",
        )
        source += "\nRUN mkdir -p /logs/artifacts\n"
    else:
        image = solver_image
    source, count = re.subn(r"^FROM [^\n]+$", f"FROM {image}", source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected one DeepSWE Dockerfile stage")
    return source
