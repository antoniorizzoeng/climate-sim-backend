import hashlib
import io
import json
import subprocess
from contextlib import suppress
from dataclasses import asdict, dataclass

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .models import Colormap
from .netcdf import read_frame


@dataclass(frozen=True)
class Style:
    colormap: Colormap = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    max_side: int | None = None


def display_range(path, boundaries):
    plane = read_frame(path, 0)
    finite = plane[np.isfinite(plane)]
    if not finite.size:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    if any(
        boundaries.get(edge, "dirichlet") == "dirichlet"
        for edge in ("left", "right", "top", "bottom")
    ):
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    return [lo, hi]


def render(values, style, limits):
    lo, hi = (
        (limits or (None, None)) if style.vmin is None else (style.vmin, style.vmax)
    )
    mask = ~np.isfinite(values)
    # Uniform planes use the midpoint by default, including a nonzero constant
    # whose run range has been widened to include a Dirichlet boundary.
    uniform = style.vmin is None and not mask.any() and values.min() == values.max()
    if lo is None or lo == hi or uniform:
        scaled = np.full(values.shape, 0.5)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            if np.isfinite(hi - lo):
                scaled = (values - lo) / (hi - lo)
            else:
                scaled = (values * 0.5 - lo * 0.5) / (hi * 0.5 - lo * 0.5)
    rgba = colormaps[style.colormap](np.ma.array(scaled, mask=mask), bytes=True)
    image = Image.fromarray(np.flipud(rgba))
    if style.max_side and max(image.size) > style.max_side:
        scale = style.max_side / max(image.size)
        image = image.resize(
            tuple(max(1, round(s * scale)) for s in image.size), Image.Resampling.BOX
        )
    return image


def png(values, style, limits):
    buf = io.BytesIO()
    render(values, style, limits).save(buf, format="PNG")
    return buf.getvalue()


def export_name(fmt, fps, style):
    key = json.dumps([fps, asdict(style)], sort_keys=True).encode()
    return f"export-{hashlib.sha256(key).hexdigest()}.{fmt}"


def encode(directory, target, fps, style, run, cancelled, progress):
    tmp = target.with_suffix(".partial" + target.suffix)
    process = None
    try:
        for n in range(run["frames"]):
            if cancelled.is_set():
                return
            plane = read_frame(directory / "result.nc", n)
            frame = render(plane, style, run["display_range"]).convert("RGB")
            if process is None:
                if target.suffix == ".mp4":
                    codec = [
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                else:
                    codec = [
                        "-filter_complex_threads",
                        "1",
                        "-filter_complex",
                        "split[a][b];[a]palettegen[p];[b][p]paletteuse",
                    ]
                command = [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{frame.width}x{frame.height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    *codec,
                    str(tmp),
                ]
                process = subprocess.Popen(command, stdin=subprocess.PIPE)
            process.stdin.write(frame.tobytes())
            progress[0] = (n + 1) / run["frames"]
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode animation")
        tmp.replace(target)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            with suppress(BrokenPipeError):
                process.stdin.close()
        tmp.unlink(missing_ok=True)
