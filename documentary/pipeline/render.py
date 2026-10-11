"""Parallel frame renderer -> x264 chunks -> lossless concat."""
import json
import math
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from . import config as C
from .gfx import Ctx

_CTX = None
_CAP = None


def _init(W, H, fps, feats_path, events_path, caps):
    global _CTX, _CAP
    feats = json.loads(Path(feats_path).read_text())
    events = json.loads(Path(events_path).read_text())
    _CTX = Ctx(W, H, fps, feats, events)
    if caps is not None:
        from .captions import burner
        _CAP = burner(caps)


def ffmpeg_x264(out, W, H, fps, crf, preset, threads=2):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-tune", "grain",
            "-profile:v", "high", "-threads", str(threads),
            "-x264-params", f"keyint={fps * 2}:min-keyint={fps}:colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            str(out)]


def _chunk(job):
    from .scenes import render_frame
    idx, f0, f1, out, W, H, fps, crf, preset, t_off = job
    p = subprocess.Popen(ffmpeg_x264(out, W, H, fps, crf, preset), stdin=subprocess.PIPE)
    for f in range(f0, f1):
        img = render_frame(_CTX, t_off + f / fps, _CAP)
        if img.size != (W, H) or img.mode != "RGB":
            img = img.convert("RGB").resize((W, H))
        p.stdin.write(img.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on chunk {idx}")
    return idx, f1 - f0


def render(out, W, H, t0, t1, feats_path, events_path, crf, preset, workers, captions=None, log=print,
           fps=C.FPS, chunk_s=6.0):
    out = Path(out)
    tmp = out.parent / (out.stem + "_chunks")
    tmp.mkdir(parents=True, exist_ok=True)
    nf = int(round((t1 - t0) * fps))
    per = max(1, int(chunk_s * fps))
    jobs = []
    for i, f0 in enumerate(range(0, nf, per)):
        jobs.append((i, f0, min(nf, f0 + per), str(tmp / f"c{i:04d}.mp4"), W, H, fps, crf, preset, t0))
    st = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(W, H, fps, str(feats_path), str(events_path), captions)) as pool:
        for idx, k in pool.imap_unordered(_chunk, jobs):
            done += k
            el = time.time() - st
            if idx % max(1, len(jobs) // 10) == 0 or done == nf:
                log(f"    {out.name}: {done}/{nf} frames  {done / el:5.1f} fps  eta {el / done * (nf - done):5.0f}s")
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{Path(j[3]).name}'\n" for j in jobs))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    for j in jobs:
        Path(j[3]).unlink()
    lst.unlink()
    tmp.rmdir()
    return out


def still(t, W, H, feats_path, events_path, path, portrait_caps=None):
    _init(W, H, C.FPS, feats_path, events_path, portrait_caps)
    from .scenes import render_frame
    img = render_frame(_CTX, t, _CAP)
    img.save(path)
    return path
