"""Persistent Python cells executed only inside the image-tool container."""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import signal
import sys
import traceback
import warnings
from contextlib import suppress
from types import FrameType
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class CellResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int
    exec_count: int
    result_repr: str | None
    images: list[str]


NS: dict[str, object] = {"__name__": "__main__"}
_exec_count = 0
MAX_IMAGES_PER_CELL = 8
MAX_IMAGE_DIM = 8192
MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
MAX_TEXT_FIELD_CHARS = 4 * 1024 * 1024
_CAPTURED: list[str] = []
_captured_bytes = 0


def _truncate_text(text: str, limit: int) -> str:
    cap = MAX_TEXT_FIELD_CHARS if not limit or limit <= 0 else min(limit, MAX_TEXT_FIELD_CHARS)
    if len(text) > cap:
        return text[:cap] + "\n... [output truncated]"
    return text


class _CellTimeout(Exception):
    pass


def _on_alarm(signum: int, frame: FrameType | None) -> None:
    raise _CellTimeout()


def validate_image_size(width: int, height: int) -> None:
    if max(width, height) > MAX_IMAGE_DIM:
        raise ValueError(
            f"Cannot display image of size {width}x{height}: the maximum is {MAX_IMAGE_DIM} "
            "pixels on either side. Crop or resize the image before displaying it."
        )


def _add_png(png_bytes: bytes) -> None:
    from PIL import Image

    global _captured_bytes
    if len(_CAPTURED) >= MAX_IMAGES_PER_CELL:
        raise ValueError(f"Cannot display more than {MAX_IMAGES_PER_CELL} images per cell.")
    with Image.open(io.BytesIO(png_bytes)) as im:
        validate_image_size(*im.size)
    if _captured_bytes + len(png_bytes) > MAX_TOTAL_IMAGE_BYTES:
        raise ValueError(
            f"Cannot display images totaling more than {MAX_TOTAL_IMAGE_BYTES} bytes of PNG data "
            "per cell. Resize the images or display fewer images per call."
        )
    _captured_bytes += len(png_bytes)
    _CAPTURED.append(base64.b64encode(png_bytes).decode("ascii"))


def _fig_to_png(fig: Figure) -> bytes:
    from matplotlib import rcParams

    dpi = rcParams["savefig.dpi"]
    if dpi == "figure":
        dpi = fig._original_dpi
    validate_image_size(*(int(size * dpi) for size in fig.get_size_inches()))
    original_dpi = fig.dpi
    layout_engine = fig.get_layout_engine()
    try:
        fig.set_dpi(dpi)
        fig.draw_without_rendering()
        bbox = fig.get_tightbbox().padded(rcParams["savefig.pad_inches"])
        validate_image_size(int(bbox.width * dpi), int(bbox.height * dpi))
        if layout_engine is not None:
            fig.set_layout_engine("none")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox)
        return buf.getvalue()
    finally:
        if layout_engine is not None:
            fig.set_layout_engine(layout_engine)
        fig.set_dpi(original_dpi)


def _capture_displayable(obj: object) -> bool:
    mod = type(obj).__module__ or ""
    if mod.startswith("matplotlib"):
        import matplotlib.pyplot as plt
        from matplotlib.artist import Artist
        from matplotlib.figure import Figure

        if isinstance(obj, Figure):
            fig = obj
        elif isinstance(obj, Artist):
            fig = obj.get_figure()
        else:
            return False
        if isinstance(fig, Figure):
            try:
                _add_png(_fig_to_png(fig))
            finally:
                plt.close(fig)
            return True
    elif mod.startswith("PIL"):
        from PIL import Image

        if not isinstance(obj, Image.Image):
            return False
        validate_image_size(*obj.size)
        im = obj.convert("LA") if obj.mode == "La" else obj
        mode = "RGBA" if im.has_transparency_data else "RGB"
        if im.mode != mode:
            im = im.convert(mode)
        buf = io.BytesIO()
        im.save(buf, format="PNG", exif=b"")
        _add_png(buf.getvalue())
        return True
    return False


def display(*objs: object) -> None:
    for obj in objs:
        if not _capture_displayable(obj):
            with suppress(Exception):
                print(repr(obj))


NS["display"] = display


def _sweep_matplotlib() -> bool:
    if sys.modules.get("matplotlib") is None:
        return False
    import matplotlib.pyplot as plt

    failed = False
    for num in plt.get_fignums():
        fig = plt.figure(num)
        try:
            _add_png(_fig_to_png(fig))
        except Exception:
            traceback.print_exc()
            failed = True
        finally:
            plt.close(fig)
    return failed


def run_cell(code: str, timeout: float, max_output_bytes: int = 0) -> CellResult:
    global _exec_count, _captured_bytes
    _exec_count += 1
    _CAPTURED.clear()
    _captured_bytes = 0
    out, err = (io.StringIO(), io.StringIO())
    exit_code = 0
    result_repr = None
    real_out, real_err = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = (out, err)
    if timeout and timeout > 0:
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        tree = ast.parse(code, "<cell>", "exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body[-1].value)
            tree.body.pop()
        mod = compile(tree, "<cell>", "exec")
        exec(mod, NS, NS)
        if last_expr is not None:
            value = eval(compile(last_expr, "<cell>", "eval"), NS, NS)
            if value is not None and (not _capture_displayable(value)):
                result_repr = repr(value)
    except _CellTimeout:
        exit_code = 124
        err.write(f"TimeoutError: cell exceeded {timeout} seconds\n")
    except SystemExit:
        exit_code = 1
        err.write(traceback.format_exc())
    except BaseException:
        exit_code = 1
        err.write(traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if _sweep_matplotlib() and exit_code == 0:
                exit_code = 1
        finally:
            sys.stdout, sys.stderr = (real_out, real_err)
    return {
        "stdout": _truncate_text(out.getvalue(), max_output_bytes),
        "stderr": _truncate_text(err.getvalue(), max_output_bytes),
        "exit_code": exit_code,
        "exec_count": _exec_count,
        "result_repr": _truncate_text(result_repr, max_output_bytes)
        if result_repr is not None
        else None,
        "images": list(_CAPTURED),
    }


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    warnings.filterwarnings("ignore", message=".*non-interactive.*")
    signal.signal(signal.SIGALRM, _on_alarm)
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    wire.write('{"ready": true}\n')
    for line in sys.stdin:
        request = json.loads(line)
        result = run_cell(request["code"], request["timeout"], request["max_output_bytes"])
        wire.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
