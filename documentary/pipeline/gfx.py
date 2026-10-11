"""Resolution-independent 2D graphics toolkit (PIL + numpy) tuned for 4K throughput.

All layout is authored in 'units' where 1u = 1px at 1080 lines (short side),
so the same scene code renders 640x360 smoke frames, 3840x2160 masters and
1080x1920 vertical social cuts.
"""
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ------------------------------------------------------------------ palette
BLACK = (6, 6, 8)
INK = (12, 12, 16)
WHITE = (245, 242, 235)
PAPER = (236, 228, 210)
AMBER = (255, 176, 64)
RED = (232, 65, 44)
SEPIA = (196, 150, 98)
CYAN = (45, 226, 230)
MAGENTA = (255, 46, 136)
VIOLET = (122, 92, 255)
LIME = (190, 255, 80)
GOLD = (255, 205, 110)
GREY = (120, 122, 130)

FONT_DIRS = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/freefont"]
FONT_FILES = {
    "black": "InterDisplay-Black.otf",
    "xbold": "InterDisplay-ExtraBold.otf",
    "bold": "InterDisplay-Bold.otf",
    "semi": "InterDisplay-SemiBold.otf",
    "med": "InterDisplay-Medium.otf",
    "light": "InterDisplay-Light.otf",
    "thin": "InterDisplay-Thin.otf",
    "italic": "InterDisplay-BoldItalic.otf",
    "mono": "DejaVuSansMono-Bold.ttf",
    "mono_r": "DejaVuSansMono.ttf",
    "type": "LiberationMono-Bold.ttf",
}
FALLBACK = "DejaVuSans-Bold.ttf"


def _find(name):
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return str(p)
    return None


@lru_cache(maxsize=512)
def font(kind, px):
    path = _find(FONT_FILES.get(kind, kind)) or _find(FALLBACK)
    return ImageFont.truetype(path, max(6, int(px)))


def fonts_ok():
    return {k: bool(_find(v)) for k, v in FONT_FILES.items()}


# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp(x)
    return x ** 3


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def back_out(x, s=1.7):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def window(t, a, b, fin=0.4, fout=0.4):
    """0..1 visibility of an element shown between a and b with fades."""
    if t < a or t > b:
        return 0.0
    return min(1.0, (t - a) / fin if fin else 1.0, (b - t) / fout if fout else 1.0)


def lerp(a, b, k):
    return a + (b - a) * k


def mixc(c1, c2, k):
    return tuple(int(lerp(a, b, clamp(k))) for a, b in zip(c1, c2))


def scale_c(c, k):
    return tuple(int(clamp(v * k, 0, 255)) for v in c)


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, W, H, fps, feats, events, seed=0):
        self.W, self.H, self.fps = W, H, fps
        self.portrait = H > W
        self.u = min(W, H) / 1080.0
        self.feats = feats
        self.events = events
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self._grain = None
        self._vig = {}

    def U(self, v):
        return v * self.u

    def f(self, kind, size):
        return font(kind, size * self.u)

    def feat(self, key, t):
        arr = self.feats[key]
        i = int(t * self.fps)
        return arr[min(max(i, 0), len(arr) - 1)]

    def env_at(self, t):
        """100 Hz envelope of the bed (for drawing waveforms)."""
        e = self.feats["env"]
        i = int(t * 100)
        return e[min(max(i, 0), len(e) - 1)]

    def ev(self, kind):
        return self.events.get(kind, [])

    def since(self, kind, t, tag=None, horizon=10.0):
        """Seconds since the most recent event of kind at/before t (inf if none)."""
        best = math.inf
        for e in self.events.get(kind, []):
            if tag and not str(e.get("tag", "")).startswith(tag):
                continue
            d = t - e["t"]
            if 0 <= d < best:
                best = d
        return best if best < horizon else math.inf

    def pulse(self, kind, t, tau=0.12, tag=None):
        d = self.since(kind, t, tag)
        return math.exp(-d / tau) if d != math.inf else 0.0

    # grain bank: 16 frames of luminance noise, tiled at full res
    def grain(self, i):
        if self._grain is None:
            rng = np.random.default_rng(16)
            gw, gh = max(64, self.W // 2), max(64, self.H // 2)
            bank = []
            for _ in range(12):
                g = rng.normal(128, 30, (gh, gw)).clip(0, 255).astype(np.uint8)
                im = Image.fromarray(g, "L").resize((self.W, self.H), Image.BILINEAR)
                bank.append(Image.merge("RGB", (im, im, im)))
            self._grain = bank
        return self._grain[i % len(self._grain)]

    def vignette(self, strength=0.55, w=None, h=None):
        w, h = w or self.W, h or self.H
        key = (strength, w, h)
        if key not in self._vig:
            sw, sh = max(32, w // 8), max(32, h // 8)
            y, x = np.mgrid[0:sh, 0:sw]
            d = np.sqrt(((x - sw / 2) / (sw / 2)) ** 2 + ((y - sh / 2) / (sh / 2)) ** 2) / 1.414
            v = (1 - strength * np.clip(d, 0, 1) ** 2.2) * 255
            im = Image.fromarray(v.astype(np.uint8), "L").resize((w, h), Image.BILINEAR)
            self._vig[key] = Image.merge("RGB", (im, im, im))
        return self._vig[key]


# ------------------------------------------------------------------ canvases
def canvas(ctx, color=BLACK, w=None, h=None):
    return Image.new("RGB", (w or ctx.W, h or ctx.H), color)


def layer(ctx, w=None, h=None):
    return Image.new("RGBA", (w or ctx.W, h or ctx.H), (0, 0, 0, 0))


def vgrad(w, h, top, bottom):
    a = np.linspace(0, 1, max(h, 2))[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - a) + np.array(bottom)[None, None, :] * a)
    col = Image.fromarray(arr.astype(np.uint8).repeat(1, axis=1), "RGB")
    return col.resize((w, h), Image.BILINEAR)


def radial(w, h, cx, cy, r, color, power=2.0):
    """RGB radial glow (black outside) — computed at 1/4 res."""
    sw, sh = max(8, w // 4), max(8, h // 4)
    y, x = np.mgrid[0:sh, 0:sw]
    d = np.sqrt((x - cx * sw / w) ** 2 + (y - cy * sh / h) ** 2) / max(r * sw / w, 1e-3)
    k = np.clip(1 - d, 0, 1) ** power
    arr = (k[..., None] * np.array(color)[None, None, :]).astype(np.uint8)
    return Image.fromarray(arr, "RGB").resize((w, h), Image.BILINEAR)


def add(base, top, k=1.0):
    if k != 1.0:
        top = Image.eval(top, lambda v: int(v * k))
    return ImageChops.add(base, top)


def screen(base, top):
    return ImageChops.screen(base, top)


def glow(img, radius_u, ctx, k=1.0, thresh=0):
    """Bloom: blur a 1/6-res copy and screen it back."""
    w, h = img.size
    s = 6
    small = img.resize((max(1, w // s), max(1, h // s)), Image.BILINEAR)
    if thresh:
        small = small.point(lambda v: 0 if v < thresh else int((v - thresh) * 255 / (255 - thresh)))
    small = small.filter(ImageFilter.GaussianBlur(max(1, radius_u * ctx.u / s)))
    if k != 1.0:
        small = small.point(lambda v: min(255, int(v * k)))
    return ImageChops.screen(img, small.resize((w, h), Image.BILINEAR))


def paste_rgba(base, lay, xy=(0, 0)):
    base.paste(lay, xy, lay)
    return base


def alpha_mul(lay, k):
    if k >= 0.999:
        return lay
    r, g, b, a = lay.split()
    a = a.point(lambda v: int(v * k))
    return Image.merge("RGBA", (r, g, b, a))


# ------------------------------------------------------------------ looks
def film16(ctx, img, t, strength=1.0, warm=True, frame_i=0, scratches=True):
    """16mm emulation: grade, gate weave, flicker, grain, vignette, dust & scratches."""
    rng = np.random.default_rng(int(t * 1000) + 7)
    # grade: lifted blacks, warm mids, rolled highlights
    if warm:
        lr = [int(clamp(18 + v * 0.93 * 1.05, 0, 255)) for v in range(256)]
        lg = [int(clamp(14 + v * 0.90, 0, 255)) for v in range(256)]
        lb = [int(clamp(16 + v * 0.74, 0, 255)) for v in range(256)]
    else:
        lr = lg = lb = [int(clamp(12 + v * 0.92, 0, 255)) for v in range(256)]
    flick = 1 + 0.045 * strength * math.sin(t * 37.0) * math.sin(t * 13.3) + rng.normal(0, 0.012 * strength)
    lut = [int(clamp(v * flick, 0, 255)) for v in lr + lg + lb]
    img = img.point(lut)
    # gate weave
    dx = int(round(math.sin(t * 2.1) * 2.0 * ctx.u * strength + rng.normal(0, 0.6 * ctx.u * strength)))
    dy = int(round(math.sin(t * 1.3 + 1) * 2.6 * ctx.u * strength))
    if dx or dy:
        img = ImageChops.offset(img, dx, dy)
    # grain
    g = ctx.grain(frame_i + int(t * 24))
    img = ImageChops.overlay(img, Image.blend(Image.new("RGB", img.size, (128, 128, 128)), g, 0.9 * strength)) \
        if img.size == g.size else img
    # dust & scratches
    if scratches:
        d = ImageDraw.Draw(img)
        for _ in range(rng.poisson(3 * strength)):
            x, y = rng.uniform(0, ctx.W), rng.uniform(0, ctx.H)
            r = rng.uniform(1, 4) * ctx.u
            c = int(rng.choice([20, 230]))
            d.ellipse([x - r, y - r, x + r, y + r * rng.uniform(0.5, 2)], fill=(c, c, c))
        if rng.uniform() < 0.25 * strength:
            x = rng.uniform(0.1, 0.9) * ctx.W
            d.line([(x, 0), (x + rng.normal(0, 6) * ctx.u, ctx.H)], fill=(200, 196, 180), width=max(1, int(ctx.u)))
    img = ImageChops.multiply(img, ctx.vignette(0.65))
    return img


def digital(ctx, img, bloom=0.9, vig=0.35, frame_i=0, grain=0.12):
    img = glow(img, 26, ctx, k=bloom, thresh=40)
    if grain:
        g = ctx.grain(frame_i)
        img = ImageChops.overlay(img, Image.blend(Image.new("RGB", img.size, (128, 128, 128)), g, grain))
    return ImageChops.multiply(img, ctx.vignette(vig))


def pillarbox43(ctx, content, bg=BLACK, x0=None, w=None, h=None, rounded=True):
    """Fit content into a 4:3 'gate' within the frame (or a given region)."""
    W, H = w or ctx.W, h or ctx.H
    gh = H
    gw = int(gh * 4 / 3)
    if gw > W:
        gw = W
        gh = int(gw * 3 / 4)
    out = Image.new("RGB", (W, H), bg)
    c = content.resize((gw, gh), Image.BILINEAR) if content.size != (gw, gh) else content
    mask = Image.new("L", (gw, gh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, gw - 1, gh - 1], radius=int(gh * 0.04) if rounded else 0, fill=255)
    out.paste(c, ((W - gw) // 2, (H - gh) // 2), mask)
    return out


# ------------------------------------------------------------------ text
def text_size(fnt, s, tracking=0):
    l, t, r, b = fnt.getbbox(s)
    return r - l + tracking * max(0, len(s) - 1), b - t


def wrap(fnt, s, maxw):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_size(fnt, trial)[0] <= maxw or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_text(d, xy, s, fnt, fill, anchor="mm", tracking=0, stroke=0, stroke_fill=None):
    if not tracking:
        d.text(xy, s, font=fnt, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=stroke_fill)
        return
    w, _ = text_size(fnt, s, tracking)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    for ch in s:
        d.text((x, y), ch, font=fnt, fill=fill, anchor="l" + anchor[1], stroke_width=stroke, stroke_fill=stroke_fill)
        x += text_size(fnt, ch)[0] + tracking if ch != " " else fnt.getlength(" ") + tracking


def kinetic(ctx, img, s, t, a, b, cx, cy, size, kind="black", color=WHITE, style="words", maxw=None,
            align="center", line_h=1.08, stroke=0, accent=None, shadow=True):
    """Kinetic typography. Styles: words (staggered rise), slam (scale-punch), type (typewriter), track."""
    vis = window(t, a, b, 0.001, 0.35)
    if vis <= 0:
        return img
    lt = t - a
    fnt = ctx.f(kind, size)
    maxw = maxw or ctx.W * 0.84
    lines = wrap(fnt, s, maxw)
    lh = size * ctx.u * line_h
    y0 = cy - lh * (len(lines) - 1) / 2
    lay = layer(ctx)
    d = ImageDraw.Draw(lay)
    wi = 0
    for li, line in enumerate(lines):
        y = y0 + li * lh
        words = line.split(" ")
        lw = text_size(fnt, line)[0]
        x = cx - lw / 2 if align == "center" else cx
        for w in words:
            ww = fnt.getlength(w)
            if style == "words":
                k = ease_out((lt - wi * 0.07) / 0.45)
                dy = (1 - k) * size * 0.55 * ctx.u
                alpha = int(255 * k * vis)
            elif style == "type":
                k = 1.0 if lt > wi * 0.06 else 0.0
                dy, alpha = 0, int(255 * k * vis)
            else:
                k, dy, alpha = 1.0, 0, int(255 * vis)
            if alpha > 0:
                col = accent if (accent and w.strip(".,!?—:'\"").upper() in accent[1]) else None
                fill = (col[0] if col else color) + (alpha,)
                if shadow:
                    d.text((x + 3 * ctx.u, y + dy + 4 * ctx.u), w, font=fnt, fill=(0, 0, 0, int(alpha * 0.55)), anchor="lm")
                d.text((x, y + dy), w, font=fnt, fill=fill, anchor="lm", stroke_width=stroke,
                       stroke_fill=(0, 0, 0, alpha) if stroke else None)
            x += ww + fnt.getlength(" ")
            wi += 1
    if style == "slam":
        k = back_out(lt / 0.35)
        sc = 1.35 - 0.35 * k
        if abs(sc - 1) > 0.01:
            w2, h2 = int(ctx.W * sc), int(ctx.H * sc)
            big = lay.resize((w2, h2), Image.BILINEAR)
            ox, oy = int(cx * sc - cx), int(cy * sc - cy)
            lay = big.crop((ox, oy, ox + ctx.W, oy + ctx.H))
    img.paste(lay, (0, 0), lay)
    return img


def label(d, ctx, x, y, s, size=22, kind="mono", fill=WHITE, anchor="lm", tracking=0):
    draw_text(d, (x, y), s, ctx.f(kind, size), fill, anchor=anchor, tracking=tracking * ctx.u)


def chip(d, ctx, x, y, s, size=20, fg=BLACK, bg=AMBER, kind="mono", pad=10, anchor="lm"):
    fnt = ctx.f(kind, size)
    w, h = text_size(fnt, s)
    p = pad * ctx.u
    if anchor[0] == "m":
        x -= (w + 2 * p) / 2
    elif anchor[0] == "r":
        x -= w + 2 * p
    d.rounded_rectangle([x, y - h / 2 - p * 0.8, x + w + 2 * p, y + h / 2 + p * 0.8], radius=p * 0.5, fill=bg)
    d.text((x + p, y), s, font=fnt, fill=fg, anchor="lm")
    return w + 2 * p


# ------------------------------------------------------------------ widgets
def vu_meter(d, ctx, x, y, w, h, level, peak=None, segs=24, horizontal=True, palette=(LIME, AMBER, RED)):
    gap = 0.18
    lit = int(clamp(level) * segs)
    for i in range(segs):
        fr = i / segs
        c = palette[0] if fr < 0.65 else palette[1] if fr < 0.85 else palette[2]
        on = i < lit or (peak is not None and i == int(clamp(peak) * segs) - 1)
        col = c if on else scale_c(c, 0.12)
        if horizontal:
            sw = w / segs
            d.rectangle([x + i * sw, y, x + (i + 1 - gap) * sw, y + h], fill=col)
        else:
            sh = h / segs
            d.rectangle([x, y + h - (i + 1) * sh, x + w, y + h - (i + gap) * sh], fill=col)


def waveform(d, ctx, x, y, w, h, t0, t1, color, playhead=None, width=None, dim=None, mute=None, sym=True):
    """Draw a waveform of the bed envelope between t0..t1 into box (x,y,w,h)."""
    cols = max(8, int(w / (3 * ctx.u)))
    width = width or max(1, int(1.8 * ctx.u))
    for i in range(cols):
        tt = t0 + (t1 - t0) * i / cols
        a = ctx.env_at(tt) * (0.25 + 0.75 * abs(math.sin(tt * 37.1) * math.cos(tt * 11.3)))
        if mute and mute(tt):
            a *= 0.08
        hh = clamp(a) * h / 2
        xx = x + w * i / cols
        c = color if (playhead is None or tt <= playhead or dim is None) else dim
        if sym:
            d.line([(xx, y + h / 2 - hh), (xx, y + h / 2 + hh)], fill=c, width=width)
        else:
            d.line([(xx, y + h), (xx, y + h - hh * 2)], fill=c, width=width)


def head_silhouette(d, x, y, s, fill, hands=0.0):
    """Generic crowd member (head + shoulders), size s = head radius."""
    d.ellipse([x - s, y - s * 1.15, x + s, y + s * 1.05], fill=fill)
    d.rounded_rectangle([x - s * 2.3, y + s * 0.95, x + s * 2.3, y + s * 4.2], radius=s * 1.4, fill=fill)
    if hands > 0.05:
        hy = y - s * (1.4 + 2.6 * hands)
        for sgn in (-1, 1):
            hx = x + sgn * s * 1.9
            d.line([(x + sgn * s * 1.7, y + s * 1.6), (hx, hy)], fill=fill, width=max(1, int(s * 0.7)))
            d.ellipse([hx - s * 0.45, hy - s * 0.55, hx + s * 0.45, hy + s * 0.45], fill=fill)


def dj_silhouette(d, x, y, s, fill, pose=0.0, arms=(0.0, 0.0), head_bob=0.0, phones=False):
    """Generic DJ figure behind a booth line at y (no likeness of any real person)."""
    hb = head_bob * s * 0.25
    d.rectangle([x - s * 0.22, y - s * 2.1 + hb, x + s * 0.22, y - s * 1.8 + hb], fill=fill)            # neck
    d.ellipse([x - s * 0.5, y - s * 3.15 + hb, x + s * 0.5, y - s * 1.95 + hb], fill=fill)          # head
    if phones:
        d.arc([x - s * 0.62, y - s * 3.3 + hb, x + s * 0.62, y - s * 2.1 + hb], 180, 360, fill=fill, width=max(2, int(s * 0.14)))
        for sg in (-1, 1):
            d.rounded_rectangle([x + sg * s * 0.62 - s * 0.16, y - s * 2.75 + hb, x + sg * s * 0.62 + s * 0.16, y - s * 2.3 + hb],
                                radius=s * 0.08, fill=fill)
    d.polygon([(x - s * 1.15, y - s * 1.55), (x + s * 1.15, y - s * 1.55), (x + s * 0.95, y + s * 0.4), (x - s * 0.95, y + s * 0.4)], fill=fill)
    d.ellipse([x - s * 1.25, y - s * 1.85, x - s * 0.55, y - s * 1.2], fill=fill)                     # shoulders
    d.ellipse([x + s * 0.55, y - s * 1.85, x + s * 1.25, y - s * 1.2], fill=fill)
    for sgn, a in ((-1, arms[0]), (1, arms[1])):
        sx, sy = x + sgn * s * 0.95, y - s * 1.55
        ex = sx + sgn * s * (0.6 + 0.5 * math.cos(a))
        ey = sy + s * (0.9 - 1.6 * max(0, math.sin(a)))
        hx = ex - sgn * s * 0.2 + sgn * s * 0.6 * math.cos(a * 1.7)
        hy = ey + s * (0.7 - 1.4 * max(0, math.sin(a)))
        d.line([(sx, sy), (ex, ey), (hx, hy)], fill=fill, width=max(2, int(s * 0.42)), joint="curve")
        d.ellipse([hx - s * 0.22, hy - s * 0.22, hx + s * 0.22, hy + s * 0.22], fill=fill)


def record(ctx, R, label_col=RED, label_txt="BREAKS VOL. 3", seed=3):
    """Pre-rendered vinyl record RGBA sprite of radius R px."""
    key = ("record", int(R), label_col, label_txt)
    if key in ctx.cache:
        return ctx.cache[key]
    S = int(R * 2)
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=(14, 14, 16, 255))
    rng = np.random.default_rng(seed)
    for r in np.linspace(R * 0.36, R * 0.97, int(R / 3)):
        g = int(20 + rng.uniform(0, 16))
        d.ellipse([R - r, R - r, R + r, R + r], outline=(g, g, g + 2, 255), width=1)
    for r in (R * 0.55, R * 0.72, R * 0.86):   # track gaps
        d.ellipse([R - r, R - r, R + r, R + r], outline=(36, 36, 40, 255), width=max(1, int(R / 120)))
    lr = R * 0.33
    d.ellipse([R - lr, R - lr, R + lr, R + lr], fill=label_col + (255,))
    f1 = font("black", R * 0.075)
    f2 = font("mono", R * 0.04)
    d.text((R, R - lr * 0.45), label_txt, font=f1, fill=(250, 240, 220, 255), anchor="mm")
    d.text((R, R + lr * 0.45), "33⅓ RPM  ·  SIDE A", font=f2, fill=(250, 240, 220, 255), anchor="mm")
    d.ellipse([R - R * 0.02, R - R * 0.02, R + R * 0.02, R + R * 0.02], fill=(5, 5, 5, 255))
    ctx.cache[key] = im
    return im


def sheen(ctx, R, angle=35):
    key = ("sheen", int(R))
    if key not in ctx.cache:
        S = int(R * 2)
        y, x = np.mgrid[0:S, 0:S] - R
        a = np.arctan2(y, x)
        k = (np.cos(2 * (a - math.radians(angle))) ** 16) * (np.hypot(x, y) < R * 0.97) * (np.hypot(x, y) > R * 0.36)
        arr = np.zeros((S, S, 4), np.uint8)
        arr[..., :3] = 255
        arr[..., 3] = (k * 70).astype(np.uint8)
        ctx.cache[key] = Image.fromarray(arr, "RGBA")
    return ctx.cache[key]


def flash(img, k, color=(255, 255, 255)):
    if k <= 0.01:
        return img
    return Image.blend(img, Image.new("RGB", img.size, color), clamp(k))


def glitch(ctx, img, k, t):
    """RGB-split + slice displacement transition."""
    if k <= 0.02:
        return img
    rng = np.random.default_rng(int(t * 997))
    r, g, b = img.split()
    off = int(18 * k * ctx.u)
    img = Image.merge("RGB", (ImageChops.offset(r, off, 0), g, ImageChops.offset(b, -off, 0)))
    for _ in range(int(6 * k)):
        y0 = int(rng.uniform(0, ctx.H))
        hh = int(rng.uniform(8, 60) * ctx.u)
        strip = img.crop((0, y0, ctx.W, min(ctx.H, y0 + hh)))
        img.paste(ImageChops.offset(strip, int(rng.normal(0, 60) * k * ctx.u), 0), (0, y0))
    return img
