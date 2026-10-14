import asyncio
import base64
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image as PILImage

from inkling_verify import python_tools
from inkling_verify.python_tools import SYSTEM_PROMPT, PythonSandbox
from inkling_verify.sandbox import CommandResult, Container
from inkling_verify.types import Image, Message, Text


def encoded_image(format="PNG"):
    output = io.BytesIO()
    PILImage.new("RGB", (3, 2), (7, 11, 13)).save(output, format=format)
    return output.getvalue()


def kernel_reply(sandbox, **fields):
    result = {
        "stdout": "",
        "stderr": "",
        "result_repr": None,
        "exit_code": 0,
        "images": [],
    } | fields
    output = asyncio.StreamReader()
    output.feed_data((json.dumps(result) + "\n").encode())
    sandbox._process = SimpleNamespace(
        stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()), stdout=output
    )


def test_expert_prompt_matches_reference_bytes():
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == (
        "75ea9b06c186ed187c3beb3feb4cac0b1419de9a52a0a36f6d818ad4a86cb4cf"
    )


def test_kernel_protocol_preserves_state_images_and_recovers_from_cell_failures(tmp_path):
    cells = [
        "value = 41\nprint('hello')\nvalue + 1",
        "from PIL import Image\ndisplay(Image.new('RGB', (3, 2), (7, 11, 13)))",
        "raise ValueError('cell failed')",
        "raise SystemExit(7)",
        "import time\ntime.sleep(2)",
        "print('x' * 20000)",
        "value + 1",
    ]
    requests = "".join(
        json.dumps({"code": code, "timeout": 0.05 if index == 4 else 5, "max_output_bytes": 16384})
        + "\n"
        for index, code in enumerate(cells)
    )
    kernel = tmp_path / "python_kernel.py"
    kernel.write_bytes(Path(python_tools.__file__).with_name("_python_kernel.py").read_bytes())
    process = subprocess.run(
        [sys.executable, str(kernel)],
        input=requests,
        text=True,
        capture_output=True,
        timeout=10,
        cwd=tmp_path,
        check=True,
    )
    ready, *results = map(json.loads, process.stdout.splitlines())
    assert ready == {"ready": True}
    assert [result["exec_count"] for result in results] == list(range(1, 8))
    assert results[0]["stdout"] == "hello\n"
    assert results[0]["result_repr"] == results[6]["result_repr"] == "42"
    assert len(results[1]["images"]) == 1
    with PILImage.open(io.BytesIO(base64.b64decode(results[1]["images"][0]))) as image:
        assert image.size == (3, 2)
        assert image.tobytes() == bytes((7, 11, 13)) * 6
    assert results[2]["exit_code"] == results[3]["exit_code"] == 1
    assert "ValueError: cell failed" in results[2]["stderr"]
    assert "SystemExit: 7" in results[3]["stderr"]
    assert results[4]["exit_code"] == 124
    assert "TimeoutError" in results[4]["stderr"]
    assert results[5]["stdout"] == "x" * 16384 + "\n... [output truncated]"
    assert results[6]["exit_code"] == 0


@pytest.mark.parametrize("format,suffix", [("PNG", "png"), ("JPEG", "jpg")])
async def test_stage_preserves_original_bytes_and_interleaves_path_hints(tmp_path, format, suffix):
    sandbox = PythonSandbox(tmp_path)
    sandbox._docker = AsyncMock()
    image = Image(encoded_image(format), f"image/{suffix}")
    messages = [Message("user", (Text("before"), image, Text("after"), image))]
    staged = await sandbox.stage(messages)
    assert staged == [
        Message(
            "user",
            (
                Text("before"),
                image,
                Text(f"[Image: source: /sandbox/images/image_0.{suffix}]"),
                Text("after"),
                image,
                Text(f"[Image: source: /sandbox/images/image_1.{suffix}]"),
            ),
        )
    ]
    assert messages[0].content == (Text("before"), image, Text("after"), image)
    assert not sandbox.allow_network
    for index, call in enumerate(sandbox._docker.await_args_list):
        local = tmp_path / "images" / f"image_{index}.{suffix}"
        assert local.read_bytes() == image.data
        assert call.args == ("cp", str(local), f"{sandbox.name}:/sandbox/images/{local.name}")


@pytest.mark.parametrize(
    "arguments",
    [
        "{",
        "[]",
        "null",
        "{}",
        '{"code": ""}',
        '{"code": 1}',
        '{"code":"1", "timeout": true}',
        '{"code":"1", "timeout": 0}',
        '{"code":"1", "timeout": 1.5}',
        '{"code":"1", "unknown": 1}',
    ],
)
async def test_bad_arguments_return_model_feedback_without_starting_execution(tmp_path, arguments):
    result = await PythonSandbox(tmp_path).execute(arguments)
    assert len(result) == 1 and result[0].text.startswith("Error: ")


@pytest.mark.parametrize(
    "exit_code,expected",
    [
        (0, "  printed\nwarning\n42"),
        (1, "  printed\nwarning"),
    ],
)
async def test_execution_preserves_rendering_and_uses_stdin_as_data(tmp_path, exit_code, expected):
    sandbox = PythonSandbox(tmp_path)
    image = encoded_image()
    kernel_reply(
        sandbox,
        stdout="  printed\n",
        stderr="warning\n",
        result_repr="42",
        exit_code=exit_code,
        images=[base64.b64encode(image).decode()],
    )
    code = 'print("literal $(touch /host-file)")\n42'
    result = await sandbox.execute(json.dumps({"code": code, "timeout": 90}))
    assert result == (Text(expected), Image(image, "image/png"))
    request = json.loads(sandbox._process.stdin.write.call_args.args[0])
    assert request == {"code": code, "timeout": 60.0, "max_output_bytes": 16384}


@pytest.mark.parametrize("with_image", [False, True])
async def test_empty_cells_and_image_only_cells_omit_empty_text(tmp_path, with_image):
    sandbox = PythonSandbox(tmp_path)
    image = encoded_image()
    kernel_reply(sandbox, images=[base64.b64encode(image).decode()] if with_image else [])
    assert await sandbox.execute('{"code":"display(image)"}') == (
        (Image(image, "image/png"),) if with_image else ()
    )


async def test_concurrent_questions_build_the_python_image_once(tmp_path, monkeypatch):
    monkeypatch.setattr(python_tools, "_BUILT_IMAGES", set())
    monkeypatch.setattr(python_tools, "_IMAGE_BUILD_LOCK", asyncio.Lock())
    calls = []

    async def docker(self, *args):
        calls.append(args)
        await asyncio.sleep(0)
        return CommandResult("", "", 0, 0)

    monkeypatch.setattr(PythonSandbox, "_docker", docker)
    monkeypatch.setattr(PythonSandbox, "_start_container", AsyncMock())
    monkeypatch.setattr(PythonSandbox, "_start_kernel", AsyncMock())
    await asyncio.gather(PythonSandbox(tmp_path).start(), PythonSandbox(tmp_path).start())
    assert [call[0] for call in calls] == ["image", "build"]


@pytest.mark.parametrize("exit_code", [0, 1, 125, 137, -11])
async def test_unconfirmed_kernel_crash_fails_the_request(tmp_path, exit_code):
    sandbox = PythonSandbox(tmp_path)
    kernel_reply(sandbox)
    sandbox._process.stdout.readline = AsyncMock(return_value=b"")
    sandbox._process.wait = AsyncMock(return_value=exit_code)
    sandbox._start_kernel = AsyncMock()
    with pytest.raises(RuntimeError, match="without a reply"):
        await sandbox.execute('{"code":"1"}')
    sandbox._start_kernel.assert_not_awaited()


async def test_sigsegv_restarts_kernel_reports_lost_state_and_allows_next_cell(tmp_path):
    sandbox = PythonSandbox(tmp_path)
    kernel_reply(sandbox)
    sandbox._process.stdout.readline = AsyncMock(return_value=b"")
    sandbox._process.wait = AsyncMock(return_value=139)
    sandbox._reap_kernel = AsyncMock()
    sandbox._docker = AsyncMock()

    async def start_kernel():
        assert sandbox._lock.locked()
        kernel_reply(sandbox, result_repr="42")

    sandbox._start_kernel = AsyncMock(side_effect=start_kernel)
    code = (
        "import cv2, numpy as np\n"
        "cv2.connectedComponentsWithStats(np.zeros((0, 90), dtype=np.uint8), 8)"
    )
    result = await sandbox.execute(json.dumps({"code": code}))
    assert "SIGSEGV (exit 139)" in result[0].text
    assert "in-memory state (variables, imports) was lost" in result[0].text
    assert json.loads((tmp_path / "python-kernel-crashes.jsonl").read_text()) == {
        "exit_code": 139,
        "request": {"code": code, "timeout": 60.0, "max_output_bytes": 16384},
    }
    sandbox._reap_kernel.assert_awaited_once_with()
    sandbox._start_kernel.assert_awaited_once_with()
    sandbox._docker.assert_not_awaited()
    assert await sandbox.execute('{"code":"6 * 7"}') == (Text("42"),)


async def test_sigsegv_restart_failure_remains_fatal(tmp_path):
    sandbox = PythonSandbox(tmp_path)
    kernel_reply(sandbox)
    sandbox._process.stdout.readline = AsyncMock(return_value=b"")
    sandbox._process.wait = AsyncMock(return_value=139)
    sandbox._reap_kernel = AsyncMock()
    sandbox._start_kernel = AsyncMock(side_effect=RuntimeError("restart failed"))
    with pytest.raises(RuntimeError, match="restart failed"):
        await sandbox.execute('{"code":"native_crash()"}')


async def test_output_limits_are_enforced_at_the_container_boundary(tmp_path, monkeypatch):
    sandbox = PythonSandbox(tmp_path)
    kernel_reply(sandbox, stdout="x" * 20000)
    assert await sandbox.execute('{"code":"1"}') == (
        Text("x" * 16384 + "\n... [output truncated]"),
    )
    image = encoded_image()
    kernel_reply(sandbox, images=[base64.b64encode(image).decode()] * 9)
    with pytest.raises(ValueError, match="image count"):
        await sandbox.execute('{"code":"1"}')
    kernel_reply(sandbox, images=[base64.b64encode(image).decode()])
    monkeypatch.setattr(python_tools, "MAX_TOTAL_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="image byte"):
        await sandbox.execute('{"code":"1"}')


async def test_wedged_kernel_restarts_container_and_reports_lost_state(tmp_path):
    sandbox = PythonSandbox(tmp_path)
    kernel_reply(sandbox)
    sandbox._process.stdout.readline = AsyncMock(side_effect=TimeoutError)
    sandbox._docker = AsyncMock()
    sandbox._reap_kernel = AsyncMock()
    sandbox._start_kernel = AsyncMock()
    result = await sandbox.execute('{"code":"blocking_native_call()", "timeout": 1}')
    assert "in-memory state (variables, imports) was lost" in result[0].text
    sandbox._docker.assert_awaited_once_with("restart", sandbox.name)
    sandbox._reap_kernel.assert_awaited_once_with()
    sandbox._start_kernel.assert_awaited_once_with()


async def test_cleanup_reaps_kernel_when_container_cleanup_fails(tmp_path, monkeypatch):
    sandbox = PythonSandbox(tmp_path)
    monkeypatch.setattr(Container, "close", AsyncMock(side_effect=RuntimeError("docker failed")))
    sandbox._reap_kernel = AsyncMock()
    with pytest.raises(RuntimeError, match="docker failed"):
        await sandbox.close()
    sandbox._reap_kernel.assert_awaited_once_with()
