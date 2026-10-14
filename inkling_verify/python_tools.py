"""Persistent, offline Python image tools for multimodal benchmark questions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from PIL import Image as PILImage

from inkling_verify._python_kernel import (
    MAX_IMAGE_DIM,
    MAX_IMAGES_PER_CELL,
    MAX_TOTAL_IMAGE_BYTES,
    _truncate_text,
)
from inkling_verify.sandbox import Container
from inkling_verify.types import Image, Message, Text, Tool

_IMAGE_BUILD_LOCK = asyncio.Lock()
_BUILT_IMAGES: set[str] = set()

SYSTEM_PROMPT = (
    "You are an AI assistant with native vision for perceiving and reasoning over images "
    "users send. You have access to an iPython tool for selecting appropriate actions to "
    "process the images as needed:\n"
    "- `ipython(code)`: run Python 3 in a persistent IPython kernel. Use it to crop, zoom, "
    "measure pixels/colors, count, or compute when needed. To load an image at `path` "
    "(PNG/JPEG) into context, `display` it, e.g. "
    "`from PIL import Image; display(Image.open(path))`; matplotlib figures and "
    "`display(...)` output are returned to you as inline images. Variables and imports "
    "persist across calls, so do NOT re-import modules or re-open an image you already "
    "loaded — build on what earlier calls defined, and split large work into several "
    "smaller calls.\n\n"
    "When the user question requires precise reading of the given image — use the tool "
    "proactively to ground your answer rather than guessing. When a value must be read "
    "precisely (axis labels, table cells, small text, exact positions or colors), use the "
    "`ipython` tool to measure it instead of eyeballing it."
    "\n\nApply these rules when solving:\n"
    "1. Ground every visual claim in what is actually rendered: read fine details, printed "
    "labels, captions, annotations, axes, and symbols exactly rather than guessing from "
    "overall shape.\n"
    "2. If the question text or answer options are rendered inside the image, transcribe "
    "them precisely (cropping and upscaling as needed) before reasoning.\n"
    "3. When reading a numeric value off a chart, report the value at the nearest "
    "explicitly-written axis tick or printed label, not an interpolated approximation "
    "(answer '60', not 'about 58'; '0.8', not '0.77'). Chart answers are graded against "
    "text written in the figure.\n"
    "4. In multi-series charts, bind every series to its color/line-style via the legend "
    "before reading it; do not assume a default color order.\n"
    "5. For threshold conditions, treat the boundary strictly ('below 0.5' excludes exactly "
    "0.50; 'over 80' excludes exactly 80).\n"
    "6. If fine detail is unclear (small text, dense ticks, subscripts, thin curves, "
    "similar colors, small/overlapping structures), use the ipython tool (PIL/numpy) to "
    "crop, upscale, and measure pixels, then display() the crop before deciding.\n"
    "7. Reason step by step, then map your result back to the exact requested "
    "parts/values/full text — or exactly one offered answer option — before committing. "
    "Never output a placeholder.\n"
)

TOOL = Tool(
    name="ipython",
    description=(
        "Executes Python 3 code in an IPython-like kernel running in the sandbox. "
        "Available libraries: PIL/numpy/matplotlib/scipy/cv2/skimage/sympy/sklearn/shapely/"
        "pandas/pytesseract. "
        "State IS preserved across calls: variables, imports, and loaded data from "
        "previous calls remain available, so you do NOT need to re-import or reload "
        "files each time. Each call returns the execution output as plain text (stdout "
        "plus the value of the last expression; use print() for anything else), or the "
        "traceback on error.  Output text is truncated to 16KB per stream. "
        "Each call displays at most 8 images. Trying to display more raises a ValueError. "
        "Images larger than 8192 pixels on either side raise a ValueError; "
        "crop or resize them before displaying. "
        "Each call can return at most 32 MiB of PNG data in total; "
        "exceeding this limit raises a ValueError.\n\n"
        "Commands run with this tool have no internet access."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Do not wrap in Markdown fences.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default and maximum: 60)",
                "default": 60,
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
)


class PythonSandbox(Container):
    def __init__(self, work_dir: Path):
        super().__init__(work_dir)
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._image_count = 0

    async def start(self) -> None:
        directory = Path(__file__).parent
        recipe = directory / "python-tools.Dockerfile"
        kernel = directory / "_python_kernel.py"
        digest = hashlib.sha256(recipe.read_bytes() + kernel.read_bytes()).hexdigest()
        image = f"inkling-verify-python:{digest}"
        async with _IMAGE_BUILD_LOCK:
            if image not in _BUILT_IMAGES:
                cached = await self._podman("image", "ls", "--quiet", image)
                if not cached.stdout.strip():
                    result = await self._podman(
                        "build", "--tag", image, "--file", str(recipe), str(directory)
                    )
                    (self.work_dir / "python-build.log").write_text(result.stdout + result.stderr)
                _BUILT_IMAGES.add(image)
        await self._start_container(image, {"memory_mb": 8192, "cpus": 2})
        await self._start_kernel()

    async def _start_kernel(self) -> None:
        with (self.work_dir / "python-kernel.log").open("ab") as log:
            self._process = await asyncio.create_subprocess_exec(
                "podman",
                "exec",
                "-i",
                self.name,
                "python",
                "-u",
                "/opt/python_kernel.py",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                limit=64 * 1024 * 1024,
            )
        assert self._process.stdout is not None
        ready = await asyncio.wait_for(self._process.stdout.readline(), timeout=30)
        if not ready or json.loads(ready) != {"ready": True}:
            raise RuntimeError("Python kernel did not become ready")

    async def _reap_kernel(self) -> None:
        if self._process is not None:
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
            self._process = None

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self._reap_kernel()

    async def stage(self, messages: Sequence[Message]) -> list[Message]:
        directory = self.work_dir / "images"
        directory.mkdir(exist_ok=True)
        prepared = []
        for message in messages:
            content = []
            for part in message.content:
                content.append(part)
                if not isinstance(part, Image):
                    continue
                with PILImage.open(io.BytesIO(part.data)) as image:
                    suffix = "jpg" if image.format in {"JPEG", "JPG"} else "png"
                name = f"image_{self._image_count}.{suffix}"
                self._image_count += 1
                source = directory / name
                source.write_bytes(part.data)
                destination = f"/sandbox/images/{name}"
                await self._podman("cp", str(source), f"{self.name}:{destination}", timeout=60)
                content.append(Text(f"[Image: source: {destination}]"))
            prepared.append(replace(message, content=tuple(content)))
        return prepared

    async def execute(self, arguments: str) -> tuple[Text | Image, ...]:
        try:
            args = json.loads(arguments)
            if not isinstance(args, dict) or args.keys() - {"code", "timeout"}:
                raise ValueError("Expected code and optional timeout")
            if not isinstance(args.get("code"), str) or not args["code"]:
                raise ValueError("'code' argument is required")
            timeout = args.get("timeout", 60)
            if type(timeout) is not int or timeout <= 0:
                raise ValueError("timeout must be a positive integer")
        except (ValueError, TypeError) as error:
            return (Text(f"Error: {error}"),)
        timeout = min(timeout, 60)
        async with self._lock:
            process = self._process
            assert process is not None and process.stdin is not None and process.stdout is not None
            request = {"code": args["code"], "timeout": float(timeout), "max_output_bytes": 16384}
            try:
                async with asyncio.timeout(timeout + 10):
                    process.stdin.write((json.dumps(request) + "\n").encode())
                    await process.stdin.drain()
                    raw = await process.stdout.readline()
            except TimeoutError:
                await self._podman("restart", self.name)
                await self._reap_kernel()
                await self._start_kernel()
                return (
                    Text(
                        f"Python execution timed out after {timeout} seconds. "
                        "The kernel was restarted, so in-memory state (variables, imports) "
                        "was lost; re-run setup before retrying."
                    ),
                )
            if not raw:
                code = await asyncio.wait_for(process.wait(), timeout=10)
                if code != 139:
                    raise RuntimeError(
                        f"Python kernel exited without a reply (exit {code}); "
                        "inspect python-kernel.log"
                    )
                with (self.work_dir / "python-kernel-crashes.jsonl").open("a") as log:
                    log.write(json.dumps({"exit_code": code, "request": request}) + "\n")
                await self._reap_kernel()
                await self._start_kernel()
                return (
                    Text(
                        "Python execution failed: the kernel exited with SIGSEGV (exit 139). "
                        "The kernel was restarted, so in-memory state (variables, imports) "
                        "was lost; re-run setup before retrying."
                    ),
                )
        result = json.loads(raw)
        parts = [_truncate_text(result[key], 16384).rstrip("\n") for key in ("stdout", "stderr")]
        text = "\n".join(part for part in parts if part)
        if result["exit_code"] == 0 and result["result_repr"] is not None:
            text = text + ("\n" if text else "") + _truncate_text(result["result_repr"], 16384)
        content: list[Text | Image] = [Text(text)] if text else []
        if len(result["images"]) > MAX_IMAGES_PER_CELL:
            raise ValueError("Python kernel exceeded its image count limit")
        total = 0
        for encoded in result["images"]:
            data = base64.b64decode(encoded, validate=True)
            total += len(data)
            with PILImage.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or max(image.size) > MAX_IMAGE_DIM:
                    raise ValueError("Python kernel returned an invalid display image")
            if total > MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("Python kernel exceeded its image byte limit")
            content.append(Image(data, "image/png"))
        return tuple(content)
