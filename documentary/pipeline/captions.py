"""Caption chunking (SRT/VTT sidecars + burned-in social captions)."""
import re

from PIL import ImageDraw

from .gfx import wrap, BLACK, WHITE


def chunks(vo_items, max_chars=64):
    """Split each VO line into readable caption events timed by character share."""
    out = []
    for it in vo_items:
        text = it["cue"].display
        parts = re.split(r"(?<=[.?!:;—])\s+", text)
        segs = []
        for p in parts:
            words, cur = p.split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > max_chars and cur:
                    segs.append(cur)
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                segs.append(cur)
        total = sum(len(s) for s in segs)
        t, dur = it["start"], it["end"] - it["start"]
        for s in segs:
            d = dur * len(s) / total
            out.append((round(t, 3), round(t + d, 3), s))
            t += d
    return out


def _ts(t, sep=","):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(round((s % 1) * 1000)) % 1000:03d}"


def write_srt(caps, path, offset=0.0):
    lines = []
    for i, (a, b, s) in enumerate(caps, 1):
        lines += [str(i), f"{_ts(a - offset)} --> {_ts(b - offset)}", s, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(caps, path):
    lines = ["WEBVTT", ""]
    for a, b, s in caps:
        lines += [f"{_ts(a, '.')} --> {_ts(b, '.')}", s, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def burner(caps):
    """Returns a caption overlay fn(ctx, img, t) for vertical social cuts (sound-off viewing)."""
    def draw(ctx, img, t):
        cur = [s for a, b, s in caps if a <= t < b]
        if not cur:
            return img
        u = ctx.u
        fnt = ctx.f("black", 54)
        lines = wrap(fnt, cur[0], ctx.W * 0.84)
        lh = 66 * u
        y0 = ctx.H * 0.70
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            w = fnt.getlength(ln)
            y = y0 + i * lh
            d.rounded_rectangle([ctx.W / 2 - w / 2 - 18 * u, y - lh / 2, ctx.W / 2 + w / 2 + 18 * u, y + lh / 2],
                                radius=12 * u, fill=BLACK)
            d.text((ctx.W / 2, y), ln, font=fnt, fill=WHITE, anchor="mm")
        return img
    return draw
