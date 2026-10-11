"""Automated delivery QC. Every check returns (name, passed, detail). Used by smoke tests and release gate."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config as C
from . import mix

FRAME = 1.0 / C.FPS


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def _ff_filter_log(path, vf=None, af=None, extra=()):
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path)]
    if vf:
        cmd += ["-vf", vf, "-an"]
    if af:
        cmd += ["-af", af, "-vn"]
    cmd += list(extra) + ["-f", "null", "-"]
    return subprocess.run(cmd, capture_output=True, text=True).stderr


def check_video(path, W, H, dur, fps=C.FPS, audio_tracks=1, subs=False, black_max=1.0, freeze_max=2.5,
                allow_black_at=((0.0, 1.65), (34.4, 35.05))):
    # 0-1.6 s: brief's 'black screen, pulsing VU meter' cold open; 34.45-35 s: the withheld drop
    res = []
    p = probe(path)
    vs = [s for s in p["streams"] if s["codec_type"] == "video"]
    as_ = [s for s in p["streams"] if s["codec_type"] == "audio"]
    ss = [s for s in p["streams"] if s["codec_type"] == "subtitle"]
    v = vs[0] if vs else {}
    res.append(("video stream present", len(vs) == 1, f"{len(vs)} video streams"))
    res.append(("codec h264 high", v.get("codec_name") == "h264" and v.get("profile") == "High", f"{v.get('codec_name')} {v.get('profile')}"))
    res.append(("resolution", (v.get("width"), v.get("height")) == (W, H), f"{v.get('width')}x{v.get('height')} (want {W}x{H})"))
    res.append(("frame rate 24p", v.get("r_frame_rate") == f"{fps}/1", v.get("r_frame_rate")))
    res.append(("pixel format yuv420p", v.get("pix_fmt") == "yuv420p", v.get("pix_fmt")))
    res.append(("bt709 tagged", v.get("color_primaries") == "bt709" and v.get("color_space") == "bt709",
                f"{v.get('color_primaries')}/{v.get('color_transfer')}/{v.get('color_space')}"))
    vdur = float(v.get("duration", p["format"]["duration"]))
    res.append(("video duration", abs(vdur - dur) <= FRAME + 1e-6, f"{vdur:.3f}s (want {dur:.3f})"))
    nb = int(v.get("nb_frames", 0) or 0)
    res.append(("frame count", nb == int(round(dur * fps)), f"{nb} frames (want {int(round(dur * fps))})"))
    res.append(("audio tracks", len(as_) == audio_tracks, f"{len(as_)} (want {audio_tracks})"))
    for a in as_:
        ad = float(a.get("duration", 0))
        res.append((f"a/v sync track {a['index']} ({a.get('channel_layout')})", abs(ad - vdur) <= 0.05, f"audio {ad:.3f}s vs video {vdur:.3f}s"))
        res.append((f"audio 48 kHz track {a['index']}", a.get("sample_rate") == "48000", a.get("sample_rate")))
    if subs:
        res.append(("subtitle track", len(ss) == 1, f"{len(ss)} subtitle streams"))
    # black segments
    log = _ff_filter_log(path, vf="blackdetect=d=0.25:pix_th=0.06")
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]
    bad = [(a, b) for a, b in blacks if (b - a) > black_max and not any(lo <= a and b <= hi for lo, hi in allow_black_at)]
    res.append(("no unintended black", not bad, f"black segments: {[(round(a, 2), round(b, 2)) for a, b in blacks]}"))
    # frozen picture
    log = _ff_filter_log(path, vf=f"freezedetect=n=0.0008:d={freeze_max}")
    fr = re.findall(r"freeze_start: ([\d.]+)", log)
    res.append(("no frozen picture", not fr, f"freezes at {fr}"))
    return res


def check_audio_file(path, target, tp_max, channels, weights=None, tol=1.0):
    x, sr = sf.read(path, always_2d=True)
    res = [(f"{Path(path).name}: 48 kHz", sr == C.SR, str(sr)),
           (f"{Path(path).name}: channels", x.shape[1] == channels, f"{x.shape[1]} (want {channels})"),
           (f"{Path(path).name}: no NaN/inf", bool(np.isfinite(x).all()), "")]
    l = mix.lufs(x, weights)
    tp = mix.true_peak_db(x)
    res.append((f"{Path(path).name}: loudness {target} LUFS ±{tol}", abs(l - target) <= tol, f"{l:.2f} LUFS"))
    res.append((f"{Path(path).name}: true peak ≤ {tp_max} dBTP", tp <= tp_max + 0.05, f"{tp:.2f} dBTP"))
    return res, x


def check_silence(x, max_gap=1.0, allow=((34.4, 35.05), (269.0, 270.0))):
    e = np.abs(x).max(axis=1)
    hop = C.SR // 100
    fr = e[: len(e) // hop * hop].reshape(-1, hop).max(axis=1) < 10 ** (-60 / 20)
    gaps, i = [], 0
    while i < len(fr):
        if fr[i]:
            j = i
            while j < len(fr) and fr[j]:
                j += 1
            a, b = i / 100, j / 100
            if (b - a) > 0.3 and not any(lo <= a and b <= hi for lo, hi in allow):
                gaps.append((a, b))
            i = j
        else:
            i += 1
    bad = [g for g in gaps if g[1] - g[0] > max_gap]
    return [("no unintended dropouts (> -60 dBFS gaps)", not bad, f"gaps: {gaps}")]


def check_vo(vo_items):
    res = []
    prev = -1
    for it in vo_items:
        ok = it["end"] <= it["cue"].hard_end and it["start"] >= prev
        res.append((f"VO {it['cue'].id} fits window", ok, f"{it['start']:.2f}-{it['end']:.2f} (limit {it['cue'].hard_end})"))
        prev = it["end"]
    return res


def check_dialogue_clarity(stereo, vo_stem, vo_items):
    """Dialogue-to-bed ratio during VO lines (stereo master minus scaled VO estimate)."""
    res = []
    for it in vo_items:
        a, b = int(it["start"] * C.SR), int(it["end"] * C.SR)
        mix_seg = stereo[a:b].mean(axis=1)
        vo_seg = vo_stem[a:b]
        g = np.dot(mix_seg, vo_seg) / (np.dot(vo_seg, vo_seg) + 1e-9)
        bed = mix_seg - g * vo_seg
        ratio = 10 * np.log10((np.mean((g * vo_seg) ** 2) + 1e-12) / (np.mean(bed ** 2) + 1e-12))
        res.append((f"dialogue clarity {it['cue'].id} (VO/bed ≥ 6 dB)", ratio >= 6.0, f"{ratio:.1f} dB"))
    return res


def check_subs(path, n_expected):
    txt = Path(path).read_text(encoding="utf-8")
    blocks = [b for b in txt.strip().split("\n\n") if b.strip()]
    times = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)", txt)
    ts = [(int(a) * 3600 + int(b) * 60 + int(c) + int(d) / 1000, int(e) * 3600 + int(f) * 60 + int(g) + int(h) / 1000)
          for a, b, c, d, e, f, g, h in times]
    mono = all(s < e for s, e in ts) and all(ts[i][1] <= ts[i + 1][0] + 1e-3 for i in range(len(ts) - 1))
    cps = max(len(b.split("\n", 2)[2]) / (e - s) for b, (s, e) in zip(blocks, ts)) if ts else 0
    return [("SRT well-formed", len(blocks) == n_expected == len(ts), f"{len(blocks)} cues (want {n_expected})"),
            ("SRT monotonic, non-overlapping", mono, ""),
            ("SRT reading speed ≤ 25 cps", cps <= 25, f"max {cps:.1f} chars/s")]


def check_stills(paths):
    from PIL import Image
    res = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        res.append((f"still {Path(p).stem} has content", float(a.std()) > 6.0, f"luma std {a.std():.1f}, mean {a.mean():.1f}"))
    return res


def report(results, path_json, path_md, title):
    ok = all(r[1] for r in results)
    Path(path_json).write_text(json.dumps([dict(check=n, passed=bool(p), detail=d) for n, p, d in results], indent=1))
    lines = [f"# {title}", "", f"**Overall: {'PASS' if ok else 'FAIL'}** — {sum(r[1] for r in results)}/{len(results)} checks passed", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if p else '**FAIL**'} | {str(d).replace('|', '/')} |" for n, p, d in results]
    Path(path_md).write_text("\n".join(lines) + "\n")
    return ok
