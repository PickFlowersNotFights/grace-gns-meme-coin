"""Scene renderers for REMIX. Each returns a full-frame RGB PIL image for time t (master timeline).

Archival-style plates (DJ booths, crowds, Brighton Beach) are original, generic
reconstructions — no real footage and no likeness of any real person.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

from . import config as C
from .gfx import (AMBER, BLACK, CYAN, GOLD, GREY, INK, LIME, MAGENTA, PAPER, RED, SEPIA, VIOLET, WHITE,
                  add, back_out, canvas, chip, clamp, digital, dj_silhouette, draw_text, ease_in, ease_io,
                  ease_out, film16, flash, glitch, glow, head_silhouette, kinetic, label, layer, lerp, mixc,
                  paste_rgba, pillarbox43, radial, record, scale_c, sheen, text_size, vgrad, vu_meter,
                  waveform, window, wrap, alpha_mul)


def fi(ctx, t):
    return int(round(t * ctx.fps))


# ====================================================================== shared plates
def crowd_rows(ctx, d, t, W, H, horizon, rows=9, color=(8, 6, 6), bob=1.0, hands=0.0, seed=1, density=1.0,
               x_off=0.0, scale=1.0, slow=1.0):
    """Perspective field of generic crowd silhouettes from horizon (y) to bottom of a W x H region."""
    rng = np.random.default_rng(seed)
    low = ctx.feat("low", t)
    kick = ctx.pulse("kick", t, 0.18)
    for r in range(rows):
        k = (r + 1) / rows                     # 0 far .. 1 near
        y = horizon + (H - horizon) * (k ** 1.6) * 0.95
        s = (3 + 34 * k ** 1.8) * ctx.u * scale
        count = int((14 + 60 * (1 - k)) * density)
        xs = rng.uniform(-0.1, 1.1, count)
        ph = rng.uniform(0, 6.28, count)
        hn = rng.uniform(0, 1, count)
        shade = mixc(color, (0, 0, 0), k * 0.6)
        for x, p, h in zip(xs, ph, hn):
            bb = (math.sin(t * slow * 8.04 + p) * 0.5 + 0.5) * (0.4 + low) * bob
            yy = y - bb * s * 0.5 - kick * s * 0.25 * bob
            hh = hands * clamp((h - (1 - hands)) * 3) * (0.7 + 0.3 * math.sin(t * slow * 4 + p))
            head_silhouette(d, (x + x_off * (0.3 + k)) * W, yy, s, shade, hh)


def lasers(ctx, lay, t, ox, oy, n_, color, spread=1.4, length=None, phase=0.0, width=2.0, intensity=1.0):
    d = ImageDraw.Draw(lay)
    length = length or ctx.W * 1.2
    for i in range(n_):
        a = -math.pi / 2 + spread * ((i / max(1, n_ - 1)) - 0.5) + 0.35 * math.sin(t * 1.7 + phase + i * 0.3)
        x2, y2 = ox + math.cos(a) * length, oy + math.sin(a) * length
        c = scale_c(color, intensity)
        d.line([(ox, oy), (x2, y2)], fill=c + (int(220 * intensity),), width=max(1, int(width * ctx.u)))


def led_wall(ctx, t, w, h, mode="festival", seed=0):
    """Animated LED wall: computed on a coarse grid then nearest-upscaled (pixel look)."""
    gw, gh = 96, 36
    y, x = np.mgrid[0:gh, 0:gw] * 0.5
    low = ctx.feat("low", t)
    kick = ctx.pulse("kick", t, 0.15)
    if mode == "festival":
        v = 0.5 + 0.5 * np.sin(x * 0.35 + t * 3.0 + np.sin(y * 0.4 + t))
        v = v * (0.35 + 0.65 * low) + kick * 0.5 * (np.abs(x - gw / 2) < gw * 0.5 * kick + 2)
        c1, c2 = np.array(CYAN), np.array(MAGENTA)
    elif mode == "pristine":
        v = 0.55 + 0.45 * np.sin((x / gw) * 6.28 + t * 0.6) * np.cos((y / gh) * 3.14 + t * 0.4)
        c1, c2 = np.array(VIOLET), np.array(CYAN)
    else:  # corporate
        v = (np.sin(x * 0.2 - t * 2) > 0.6) * 0.8 + 0.1
        c1, c2 = np.array(GOLD), np.array(WHITE)
    mixk = (np.sin(y / gh * 3 + t * 0.5) * 0.5 + 0.5)[..., None]
    rgb = (c1 * (1 - mixk) + c2 * mixk) * np.clip(v, 0, 1.2)[..., None]
    w, h = max(1, int(w)), max(1, int(h))
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").resize((w, h), Image.NEAREST)
    img = ImageChops.multiply(img, _led_mask(ctx, w, h, gw, gh))
    # panel seams
    d = ImageDraw.Draw(img)
    for i in range(1, 8):
        d.line([(w * i / 8, 0), (w * i / 8, h)], fill=(0, 0, 0), width=max(1, int(2 * ctx.u)))
    return img


def _led_mask(ctx, w, h, gw, gh):
    key = ("ledmask", w, h, gw, gh)
    if key not in ctx.cache:
        xs = (np.arange(w) * gw / w) % 1.0
        ys = (np.arange(h) * gh / h) % 1.0
        fx = np.clip(1 - np.abs(xs - 0.5) * 2, 0, 1) ** 0.35
        fy = np.clip(1 - np.abs(ys - 0.5) * 2, 0, 1) ** 0.35
        m = (np.outer(fy, fx) * 255).astype(np.uint8)
        im = Image.fromarray(m, "L")
        ctx.cache[key] = Image.merge("RGB", (im, im, im))
    return ctx.cache[key]


def booth(d, ctx, x, y, w, h, col=(14, 12, 12), top=(40, 34, 30)):
    d.rectangle([x, y, x + w, y + h], fill=col)
    d.rectangle([x, y, x + w, y + max(2, 6 * ctx.u)], fill=top)


def title_block(ctx, img, t, a, b, kicker, title, x=None, y=None, color=WHITE, kc=AMBER, size=64, anchor_left=True):
    vis = window(t, a, b, 0.5, 0.6)
    if vis <= 0:
        return img
    lay = layer(ctx)
    d = ImageDraw.Draw(lay)
    u = ctx.u
    x = x if x is not None else 90 * u
    y = y if y is not None else 110 * u
    k = ease_out((t - a) / 0.8)
    al = int(255 * vis)
    d.rectangle([x, y - 34 * u, x + 70 * u * k, y - 28 * u], fill=kc + (al,))
    d.text((x, y - 6 * u), kicker, font=ctx.f("mono", 22), fill=kc + (al,), anchor="ls")
    for i, line in enumerate(wrap(ctx.f("black", size), title, ctx.W * 0.8)):
        d.text((x, y + (i + 0.85) * size * u * 1.02 + (1 - k) * 20 * u), line, font=ctx.f("black", size),
               fill=color + (al,), anchor="ls")
    img.paste(lay, (0, 0), lay)
    return img


def scene_tag(ctx, img, t, s):
    """Small broadcast-style scene slug (top right)."""
    d = ImageDraw.Draw(img)
    u = ctx.u
    lt = t - s.start
    vis = window(lt, 0.3, 4.5, 0.4, 0.6)
    if vis <= 0 or ctx.portrait:
        return img
    txt = f"{s.act}  ·  SCENE {s.id[1]}  ·  {s.title.upper()}"
    col = scale_c(WHITE, 0.75 * vis)
    d.text((ctx.W - 40 * u, 28 * u), txt, font=ctx.f("mono", 15), fill=col, anchor="rm")
    return img


# ====================================================================== SCENE 1
def s1(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    if t < 1.6:
        img = canvas(ctx, BLACK)
        d = ImageDraw.Draw(img)
        lvl = clamp(0.2 + ctx.feat("rms", t) * 2.5 + 0.08 * math.sin(t * 31))
        bw, bh = (520 if not ctx.portrait else 640) * u, 16 * u
        for i, (ch, lv) in enumerate((("L", lvl), ("R", clamp(lvl * 0.92 + 0.04 * math.sin(t * 40))))):
            y = H / 2 + (i - 0.5) * 40 * u
            label(d, ctx, W / 2 - bw / 2 - 30 * u, y + bh / 2, ch, 18, "mono", GREY)
            vu_meter(d, ctx, W / 2 - bw / 2, y, bw, bh, lv, segs=32)
        a = window(t, 0.15, 1.55, 0.3, 0.2)
        d.text((W / 2, H / 2 - 110 * u), "ACT I", font=ctx.f("mono", 22), fill=scale_c(AMBER, a), anchor="mm")
        d.text((W / 2, H / 2 - 78 * u), "THE CRATE & THE ALGORITHM", font=ctx.f("semi", 30), fill=scale_c(WHITE, a), anchor="mm")
        return img
    if t < 4.0:
        img = _vinyl_shot(ctx, t)
        return film16(ctx, img, t, 0.9, frame_i=fi(ctx, t))
    if t < 6.6:
        return _basement_1998(ctx, t)
    if t < 9.2:
        return _led_2026(ctx, t)
    # 9.2-10: tape rewind — alternating eras, glitching
    f = fi(ctx, t)
    img = _basement_1998(ctx, 6.6 - (t - 9.2) * 3) if (f // 2) % 2 else _led_2026(ctx, 9.2 - (t - 9.2) * 3)
    return glitch(ctx, img, 0.6 + 0.4 * (t - 9.2) / 0.8, t)


def _vinyl_shot(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    img = canvas(ctx, (22, 16, 12))
    img = add(img, radial(W, H, W * 0.38, H * 0.2, max(W, H) * 0.9, (120, 84, 46), 1.6))
    R = min(W, H) * 0.62
    rec = record(ctx, R, RED, "BIG BEAT BREAKS")
    spin = 0 if t < 2.0 else (t - 2.0) * 200       # 33⅓ rpm ≈ 200°/s
    rr = rec.rotate(-spin, resample=Image.BILINEAR)
    cx, cy = W * 0.44, H * 0.58 + (1 - ease_out((t - 1.6) / 1.2)) * 30 * u
    img.paste(rr, (int(cx - R), int(cy - R)), rr)
    sh = sheen(ctx, R)
    img.paste(sh, (int(cx - R), int(cy - R)), sh)
    d = ImageDraw.Draw(img)
    # tonearm: pivot top-right, swings in and drops
    px, py = cx + R * 1.12, cy - R * 0.92
    ang = math.radians(lerp(52, 28, ease_io((t - 1.6) / 0.35)))
    L = R * 1.25
    ex, ey = px - math.sin(ang) * L * 0.95, py + math.cos(ang) * L
    d.line([(px, py), (ex, ey)], fill=(170, 170, 176), width=int(10 * u))
    d.ellipse([px - 40 * u, py - 40 * u, px + 40 * u, py + 40 * u], fill=(60, 60, 64))
    d.rectangle([ex - 22 * u, ey - 12 * u, ex + 22 * u, ey + 30 * u], fill=(200, 200, 205))
    # dust motes in a light shaft
    rng = np.random.default_rng(11)
    for i in range(70):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        x = (x0 + t * 14 * u * rng.uniform(0.3, 1)) % W
        y = (y0 - t * 9 * u * rng.uniform(0.3, 1)) % H
        r = rng.uniform(1, 3.2) * u
        b = int(120 + 120 * rng.uniform())
        d.ellipse([x - r, y - r, x + r, y + r], fill=(b, int(b * 0.85), int(b * 0.6)))
    return img


def _basement_1998(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    gw, gh = (int(H * 4 / 3), H) if not ctx.portrait else (W, int(W * 3 / 4))
    sub = Image.new("RGB", (gw, gh), (16, 6, 6))
    strobe = ctx.pulse("snare", t, 0.07)
    kick = ctx.pulse("kick", t, 0.12)
    sub = add(sub, radial(gw, gh, gw * 0.5, gh * 0.1, gw * 0.8, scale_c(RED, 0.6 + 0.6 * kick), 1.3))
    d = ImageDraw.Draw(sub)
    # low ceiling pipes
    for i in range(5):
        y = gh * (0.05 + i * 0.03)
        d.line([(0, y), (gw, y + gh * 0.02)], fill=(30, 12, 10), width=int(6 * u))
    # bare bulb
    bx = gw * 0.5
    d.line([(bx, 0), (bx, gh * 0.18)], fill=(20, 10, 8), width=int(3 * u))
    sub = add(sub, radial(gw, gh, bx, gh * 0.2, gw * 0.25, scale_c(AMBER, 0.8 + 0.5 * strobe), 2.0))
    d = ImageDraw.Draw(sub)
    crowd_rows(ctx, d, t, gw, gh, gh * 0.42, rows=6, color=(26, 10, 8), bob=1.4, hands=0.25, seed=4, density=0.7)
    sub = flash(sub, strobe * 0.5, (255, 225, 200))
    img = pillarbox43(ctx, sub)
    img = film16(ctx, img, t, 1.1, frame_i=fi(ctx, t))
    d = ImageDraw.Draw(img)
    chip(d, ctx, (W - gw) / 2 + 40 * u, 60 * u, "1998 · BASEMENT, BRIGHTON", 20, BLACK, AMBER)
    return img


def _led_2026(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    img = canvas(ctx, (2, 2, 8))
    lw, lh = W * 0.9, H * 0.42
    wall = led_wall(ctx, t, lw, lh, "festival")
    img.paste(wall, (int((W - lw) / 2), int(H * 0.08)))
    lay = layer(ctx)
    k = ctx.pulse("kick", t, 0.2)
    lasers(ctx, lay, t, W * 0.3, H * 0.52, 7, MAGENTA, 1.6, intensity=0.6 + 0.4 * k)
    lasers(ctx, lay, t, W * 0.7, H * 0.52, 7, CYAN, 1.6, phase=2, intensity=0.6 + 0.4 * k)
    img.paste(lay, (0, 0), lay)
    img = add(img, radial(W, H, W / 2, H * 0.5, W * 0.7, scale_c((90, 60, 140), 0.7 + 0.5 * k), 1.2))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H * 0.5, W, H * 0.56], fill=(10, 10, 16))
    crowd_rows(ctx, d, t, W, H, H * 0.56, rows=8, color=(4, 4, 10), bob=1.0, hands=0.6, seed=8)
    img = digital(ctx, img, 1.1, frame_i=fi(ctx, t))
    d = ImageDraw.Draw(img)
    chip(d, ctx, 60 * u, 60 * u, "2026 · MAINSTAGE", 20, BLACK, CYAN)
    return img


# ====================================================================== SCENE 2
def _panel_1998(ctx, t, w, h):
    u = ctx.u
    img = Image.new("RGB", (w, h), (24, 10, 6))
    k = ctx.pulse("kick", t, 0.12, "s2-98")
    sn = ctx.pulse("snare", t, 0.09, "s2-98")
    img = add(img, radial(w, h, w * 0.5, h * 0.25, w * 0.9, scale_c(AMBER, 0.45 + 0.4 * k), 1.4))
    img = add(img, radial(w, h, w * 0.15, h * 0.1, w * 0.4, scale_c(RED, 0.3 + 0.8 * sn), 1.6))
    d = ImageDraw.Draw(img)
    # record crates wall
    for i in range(14):
        x = w * i / 14
        d.rectangle([x + 2 * u, h * 0.1, x + w / 14 - 2 * u, h * 0.34], fill=mixc((40, 22, 14), (90, 50, 30), (i * 37 % 10) / 10))
    by = h * 0.66
    # frantic DJ: arms driven by hits
    a1 = 0.8 + 1.4 * sn + 0.5 * math.sin(t * 9.1)
    a2 = 0.4 + 1.6 * k + 0.6 * math.sin(t * 7.3 + 1)
    dj_silhouette(d, w * 0.5, by, 62 * u * w / (ctx.W / 2 if not ctx.portrait else ctx.W), (8, 4, 3),
                  arms=(a1, a2), head_bob=math.sin(t * 2 * math.pi * 112 / 60))
    # sweat droplets
    rng = np.random.default_rng(int(t * 6))
    for _ in range(int(6 * (k + sn))):
        x = w * 0.5 + rng.normal(0, 40) * u
        y = by - 170 * u + rng.uniform(-20, 30) * u
        d.ellipse([x - 3 * u, y - 4 * u, x + 3 * u, y + 4 * u], fill=(255, 220, 170))
    booth(d, ctx, 0, by, w, h - by, (20, 12, 8), (70, 44, 26))
    # two turntables + sampler pads
    for sx in (0.2, 0.8):
        R = h * 0.09
        cx, cy = w * sx, by + h * 0.13
        d.ellipse([cx - R, cy - R * 0.45, cx + R, cy + R * 0.45], fill=(12, 10, 10), outline=(80, 60, 40), width=max(1, int(2 * u)))
        a = t * 3.49 + sx * 10
        d.line([(cx, cy), (cx + math.cos(a) * R * 0.9, cy + math.sin(a) * R * 0.4)], fill=(200, 60, 40), width=max(1, int(3 * u)))
    pad = h * 0.035
    rng = np.random.default_rng(int(t * 8))
    lit = set(rng.choice(16, 3 + int(5 * sn), replace=False))
    for i in range(16):
        r, c = divmod(i, 4)
        x = w * 0.5 - 2 * pad * 1.25 + c * pad * 1.25
        y = by + h * 0.06 + r * pad * 1.25
        col = RED if i in lit else (60, 30, 22)
        d.rounded_rectangle([x, y, x + pad, y + pad], radius=pad * 0.2, fill=col)
    crowd_rows(ctx, d, t, w, h, h * 0.92, rows=2, color=(6, 3, 2), bob=1.5, hands=0.3, seed=21, density=0.45, scale=1.2)
    return img


def _panel_2026(ctx, t, w, h):
    u = ctx.u
    img = Image.new("RGB", (w, h), (2, 3, 10))
    edm = t >= 23.5
    lw, lh = w * 0.92, h * 0.48
    wall = led_wall(ctx, t, lw, lh, "festival" if edm else "pristine")
    if not edm:
        wall = Image.eval(wall, lambda v: int(v * 0.55))
    img.paste(wall, (int((w - lw) / 2), int(h * 0.06)))
    img = add(img, radial(w, h, w * 0.5, h * 0.62, w * 0.8, (40, 50, 110), 1.3))
    d = ImageDraw.Draw(img)
    by = h * 0.66
    dj_silhouette(d, w * 0.5, by, 62 * u * w / (ctx.W / 2 if not ctx.portrait else ctx.W), (3, 3, 8),
                  arms=(-0.3, 1.3 if ctx.since("co2", t) < 0.6 else 0.1), head_bob=0.0)
    booth(d, ctx, 0, by, w, h - by, (8, 10, 18), (40, 60, 90))
    # the single cue button
    bx, byy, br = w * 0.5, by + h * 0.12, h * 0.035
    press = ctx.pulse("cue_button", t, 0.5) + ctx.pulse("co2", t, 0.3)
    glowc = mixc((20, 60, 70), CYAN, 0.35 + 0.65 * clamp(press + (0.3 if edm else 0)))
    d.ellipse([bx - br, byy - br, bx + br, byy + br], fill=glowc, outline=WHITE, width=max(1, int(2 * u)))
    d.text((bx, byy + br * 1.9), "CUE", font=ctx.f("mono", 16), fill=CYAN, anchor="mm")
    # CO2 cannons
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    for e in ctx.ev("co2"):
        dt = t - e["t"]
        if 0 <= dt < 1.6:
            for cxr in (0.12, 0.88):
                rng = np.random.default_rng(int(e["t"] * 10 + cxr * 100))
                for _ in range(90):
                    sp = rng.uniform(0.6, 1.0)
                    hgt = h * 0.55 * ease_out(dt / 0.5) * sp
                    x = w * cxr + rng.normal(0, 1) * hgt * 0.18
                    y = by - hgt * rng.uniform(0.2, 1)
                    r = (10 + 50 * dt) * u * rng.uniform(0.4, 1)
                    a = int(70 * clamp(1 - dt / 1.6) * rng.uniform(0.4, 1))
                    ld.ellipse([x - r, y - r, x + r, y + r], fill=(235, 245, 255, a))
    lay = lay.filter(ImageFilter.GaussianBlur(4 * u))
    img.paste(lay, (0, 0), lay)
    d = ImageDraw.Draw(img)
    crowd_rows(ctx, d, t, w, h, h * 0.92, rows=2, color=(2, 2, 6), bob=0.6, hands=0.5, seed=31, density=0.45, scale=1.2)
    return img


def s2(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    img = canvas(ctx, INK)
    left_on = t < 22.0
    if not ctx.portrait:
        pw, ph = W // 2, H
        gh = int(pw * 3 / 4)
        p1 = _panel_1998(ctx, t, pw, gh)
        p1 = pillarbox43(ctx, p1, bg=(10, 8, 6), w=pw, h=gh, rounded=True)
        p1 = film16(ctx, _fit(p1, W, H, pw, gh, 0, (H - gh) // 2), t, 1.0, frame_i=fi(ctx, t))
        p2 = digital(ctx, _fit(_panel_2026(ctx, t, pw, ph), W, H, pw, ph, pw, 0), 1.0, frame_i=fi(ctx, t))
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rectangle([0, 0, pw - 1, H], fill=255)
        img = Image.composite(p1, p2, mask)
        regions = [(0, 0, pw, H), (pw, 0, W, H)]
    else:
        pw, ph = W, H // 2
        p1 = film16(ctx, _fit(_panel_1998(ctx, t, pw, ph), W, H, pw, ph, 0, 0), t, 1.0, frame_i=fi(ctx, t))
        p2 = digital(ctx, _fit(_panel_2026(ctx, t, pw, ph), W, H, pw, ph, 0, ph), 1.0, frame_i=fi(ctx, t))
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rectangle([0, 0, W, ph - 1], fill=255)
        img = Image.composite(p1, p2, mask)
        regions = [(0, 0, W, ph), (0, ph, W, H)]
    # dim the inactive era
    dimk = 0.5
    for i, (x0, y0, x1, y1) in enumerate(regions):
        active = left_on if i == 0 else not left_on
        if not active:
            sub = img.crop((x0, y0, x1, y1)).point(lambda v: int(v * dimk))
            img.paste(sub, (x0, y0))
    d = ImageDraw.Draw(img)
    # headers + dashboards
    lt = t - 10.0
    hdr = [("1998", "RAW BREAKBEATS & CRATE-DIGGING", AMBER, "112 BPM · SWING 60% · TIMING ±6 ms · 4 DECKS + SAMPLER"),
           ("2026", "ALGORITHMIC DROPS & PRE-RECORDED CUES", CYAN, "128.00 BPM · SWING 0% · QUANTIZED 100% · 1 CUE")]
    for i, ((x0, y0, x1, y1), (yr, txt, col, dash)) in enumerate(zip(regions, hdr)):
        k = ease_out((lt - 0.3 - i * 0.25) / 0.6)
        cx = (x0 + x1) / 2
        yy = y0 + (120 if not ctx.portrait else 70) * u
        d.text((cx, yy - (1 - k) * 30 * u), yr, font=ctx.f("black", 78), fill=scale_c(col, k), anchor="mm")
        d.text((cx, yy + 58 * u), txt, font=ctx.f("bold", 26), fill=scale_c(WHITE, k), anchor="mm")
        d.text((cx, y1 - 34 * u), dash, font=ctx.f("mono", 16), fill=scale_c(col, 0.85 * k), anchor="mm")
        lvl = ctx.feat("rms", t) * (1 if ((i == 0) == left_on) else 0.15)
        vu_meter(d, ctx, cx - 160 * u, y1 - 70 * u, 320 * u, 10 * u, clamp(lvl), segs=28)
    # divider + VS
    if not ctx.portrait:
        d.rectangle([W / 2 - 2 * u, 0, W / 2 + 2 * u, H], fill=WHITE)
        cx, cy = W / 2, H / 2
    else:
        d.rectangle([0, H / 2 - 2 * u, W, H / 2 + 2 * u], fill=WHITE)
        cx, cy = W / 2, H / 2
    r = (40 + 8 * ctx.pulse("kick", t, 0.15)) * u * (back_out((t - 12.9) / 0.4) if not ctx.portrait else 1.0)
    if r > 1:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE)
        d.text((cx, cy), "VS", font=ctx.f("black", 34 * r / (40 * u)), fill=BLACK, anchor="mm")
    # contrarian pull-quote (master only — social gets burned captions)
    if not ctx.portrait:
        img = kinetic(ctx, img, "Why are thirty-year-old breakbeats still outshining contemporary headliners?",
                      t, 27.6, 34.4, W / 2, H * 0.83, 40, "xbold", WHITE, "words", maxw=W * 0.7,
                      accent=(AMBER, {"THIRTY-YEAR-OLD", "BREAKBEATS"}))
    return img


def _fit(p, W, H, pw, ph, x, y):
    out = Image.new("RGB", (W, H), BLACK)
    out.paste(p if p.size == (pw, ph) else p.resize((pw, ph)), (x, y))
    return out


# ====================================================================== SCENE 3
LAYERS = [("60s FUNK DRUMS", "1960s · BREAK", AMBER, 41.0), ("70s ROCK BASSLINE", "1970s · RIFF", RED, None),
          ("GOSPEL ACAPELLA", "GOSPEL · VOCAL", VIOLET, None)]


def s3(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    ai_t = (ctx.ev("ai_cut") or [{"t": 52.43}])[0]["t"]
    if t < 41.0:
        img = _crate_dolly(ctx, t)
        img = film16(ctx, img, t, 0.7, frame_i=fi(ctx, t))
        img = title_block(ctx, img, t, 35.6, 40.8, "WHAT THEY MISSED", "THE LOST ART OF SAMPLING", size=78)
        return img
    if t < ai_t:
        return _spectral(ctx, t)
    return _daw(ctx, t, ai_t)


def _crate_dolly(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    lt = t - 35.0
    img = vgrad(W, H, (34, 24, 18), (10, 8, 6))
    rng = np.random.default_rng(42)
    for depth, (blur, sc, spd, dark) in enumerate([(5, 0.7, 40, 0.45), (0, 1.0, 90, 1.0), (9, 1.5, 170, 0.7)]):
        lay = layer(ctx)
        d = ImageDraw.Draw(lay)
        sw = 26 * u * sc
        y0 = H * (0.22 + depth * 0.1)
        hh = H * (0.5 + 0.12 * sc)
        off = (lt * spd * u) % (sw * 40)
        for i in range(-2, int(W / sw) + 42):
            x = i * sw - off
            if x > W + sw or x < -sw:
                continue
            seed = (i + 1000 * depth) % 97
            base = [(150, 40, 30), (40, 70, 120), (200, 160, 60), (30, 30, 30), (220, 210, 190), (80, 120, 60), (120, 60, 110)][seed % 7]
            c = scale_c(base, dark * (0.6 + 0.4 * ((seed * 13) % 10) / 10))
            tilt = (seed % 5 - 2) * 0.6 * u
            d.polygon([(x, y0 + tilt), (x + sw * 0.9, y0 - tilt), (x + sw * 0.9, y0 + hh), (x, y0 + hh)], fill=c + (255,))
            d.line([(x + sw * 0.45, y0 + 20 * u), (x + sw * 0.45, y0 + hh * 0.6)], fill=scale_c(c, 1.5) + (180,), width=max(1, int(2 * u)))
        if blur:
            lay = lay.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(blur / 4 * u * 4)).resize((W, H))
        img.paste(lay, (0, 0), lay)
    img = add(img, radial(W, H, W * 0.5, H * 0.4, W * 0.6, (80, 60, 30), 1.5))
    return img


def _spectral(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    img = canvas(ctx, (8, 7, 10))
    d = ImageDraw.Draw(img)
    layers = ctx.ev("layer")
    starts = [e["t"] for e in layers] or [41.0, 43.29, 45.57]
    names = [e.get("name") for e in layers] or [l[0] for l in LAYERS]
    # master track bar at top, splitting into lanes
    top = H * 0.16
    d.text((80 * u, top - 40 * u), "BIG BEAT TRACK  ·  ORIGINAL RECONSTRUCTION", font=ctx.f("mono", 20), fill=GREY, anchor="lm")
    waveform(d, ctx, 80 * u, top, W - 160 * u, 60 * u, t - 6, t + 0.5, WHITE, playhead=t, dim=(70, 70, 80))
    lane_h = H * 0.17
    for i, (name, era, col, _) in enumerate(LAYERS):
        st = starts[i] if i < len(starts) else 99
        k = ease_out((t - st) / 0.6)
        if k <= 0:
            continue
        y = H * 0.32 + i * (lane_h + 18 * u) + (1 - k) * -80 * u
        x0, x1 = 80 * u, W - 80 * u
        d.rounded_rectangle([x0, y, x1, y + lane_h], radius=10 * u, outline=scale_c(col, k), width=max(1, int(2 * u)),
                            fill=scale_c(col, 0.08 * k))
        # spectral bars (lane-specific band)
        cols = int((x1 - x0) / (9 * u))
        for j in range(cols):
            tt = t - 3 + 3.5 * j / cols
            if tt > t:
                continue
            band = [ctx.feat("low", tt) * 0.6 + ctx.feat("high", tt) * 0.6, ctx.feat("low", tt), ctx.feat("rms", tt) * 0.8][i]
            hh = clamp(band * (0.5 + 0.5 * abs(math.sin(j * 0.7 + i)))) * lane_h * 0.42
            x = x0 + j * (x1 - x0) / cols
            d.rectangle([x, y + lane_h / 2 - hh, x + 5 * u, y + lane_h / 2 + hh], fill=scale_c(col, 0.4 + 0.6 * k))
        chip(d, ctx, x0 + 18 * u, y + 28 * u, era, 16, BLACK, col)
        d.text((x1 - 20 * u, y + 30 * u), name, font=ctx.f("black", 34), fill=scale_c(WHITE, k), anchor="rm")
    # plus signs between lanes → collage
    if t > starts[-1] + 1.2:
        k = ease_out((t - starts[-1] - 1.2) / 0.8)
        d.text((W / 2, H * 0.93), "= THE HUMAN CRATE-DIGGING COLLAGE", font=ctx.f("xbold", 34), fill=scale_c(GOLD, k), anchor="mm")
    img = glow(img, 18, ctx, 0.8, 50)
    return film16(ctx, img, t, 0.35, warm=True, frame_i=fi(ctx, t), scratches=False)


def _daw(ctx, t, t0):
    u, W, H = ctx.u, ctx.W, ctx.H
    lt = t - t0
    img = canvas(ctx, (30, 31, 34))
    d = ImageDraw.Draw(img)
    # toolbar
    d.rectangle([0, 0, W, 64 * u], fill=(42, 43, 47))
    d.text((24 * u, 32 * u), "AUTO-GEN STUDIO", font=ctx.f("semi", 24), fill=(200, 200, 205), anchor="lm")
    d.text((W - 24 * u, 32 * u), "120.000 BPM   4/4   QUANTIZE 1/16 · 100%", font=ctx.f("mono_r", 18), fill=(150, 150, 155), anchor="rm")
    # track grid
    x0, y0 = 220 * u, 100 * u
    rows, cols = 10, 32
    gw, gh = (W - x0 - 30 * u) / cols, (H * 0.62) / rows
    for r in range(rows + 1):
        d.line([(x0, y0 + r * gh), (W - 30 * u, y0 + r * gh)], fill=(48, 49, 54), width=1)
    for c in range(cols + 1):
        d.line([(x0 + c * gw, y0), (x0 + c * gw, y0 + rows * gh)], fill=(56, 57, 62) if c % 4 else (70, 71, 78), width=1)
    filled = int(lt * 46)
    for i in range(min(filled, rows * cols)):
        r, c = divmod(i, cols)
        d.rectangle([x0 + c * gw + 2 * u, y0 + r * gh + 3 * u, x0 + (c + 1) * gw - 2 * u, y0 + (r + 1) * gh - 3 * u], fill=(92, 120, 140))
        d.line([(x0 + c * gw + 4 * u, y0 + r * gh + gh / 2), (x0 + (c + 1) * gw - 4 * u, y0 + r * gh + gh / 2)], fill=(150, 175, 190), width=max(1, int(2 * u)))
    for r in range(rows):
        d.text((20 * u, y0 + (r + 0.5) * gh), f"AI_LOOP_{r + 1:03d}.wav", font=ctx.f("mono_r", 16), fill=(160, 160, 165), anchor="lm")
    ph = x0 + ((lt * 2) % 8) / 8 * cols * gw
    d.line([(ph, y0 - 10 * u), (ph, y0 + rows * gh)], fill=(230, 230, 230), width=max(1, int(2 * u)))
    # generator dialog
    bx, by, bw, bh = W * 0.5 - 330 * u, H * 0.78, 660 * u, 150 * u
    d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=10 * u, fill=(48, 49, 54), outline=(80, 80, 88))
    d.text((bx + 24 * u, by + 30 * u), "Generate sample pack", font=ctx.f("semi", 22), fill=WHITE, anchor="lm")
    d.text((bx + 24 * u, by + 64 * u), "genre: Festival · mood: Euphoric · variations: 1,000", font=ctx.f("mono_r", 16), fill=(170, 170, 175), anchor="lm")
    p = clamp(lt / 6.5)
    d.rounded_rectangle([bx + 24 * u, by + 96 * u, bx + bw - 24 * u, by + 116 * u], radius=6 * u, fill=(30, 31, 34))
    d.rounded_rectangle([bx + 24 * u, by + 96 * u, bx + 24 * u + (bw - 48 * u) * p, by + 116 * u], radius=6 * u, fill=(92, 140, 200))
    d.text((bx + bw - 24 * u, by + 134 * u), f"{int(p * 1000):,} / 1,000 loops", font=ctx.f("mono_r", 14), fill=(170, 170, 175), anchor="rm")
    img = kinetic(ctx, img, "AUTO-GENERATED. UNIFORM. SOULLESS.", t, t0 + 0.5, 60.0, W / 2, H * 0.5, 58, "black",
                  WHITE, "slam", shadow=True) if lt > 0.5 else img
    return img


# ====================================================================== SCENE 4
def s4(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    if t < 75.6:
        img = _field_pov(ctx, t)
        img = film16(ctx, img, t, 1.0, frame_i=fi(ctx, t))
        cap = ("WIDE · 1998", "From skepticism…", AMBER)
    elif t < 88.0:
        img = _mainstage(ctx, t)
        img = digital(ctx, img, 1.2, frame_i=fi(ctx, t))
        cap = ("MEDIUM PUSH · 2026", "…to 250,000 on the beach…", CYAN)
    else:
        img = _mixer_close(ctx, t)
        img = digital(ctx, img, 0.9, frame_i=fi(ctx, t), grain=0.18)
        cap = ("EXTREME CLOSE · HANDS ON THE MIXER", "", GOLD)
    # drop flash
    img = flash(img, ctx.pulse("drop", t, 0.25), (255, 250, 240))
    # storyboard-snapshot caption (lower left)
    d = ImageDraw.Draw(img)
    seg = 60.0 if t < 75.6 else 75.6 if t < 88 else 88.0
    k = ease_out((t - seg - 0.6) / 0.6)
    if not ctx.portrait and k > 0:
        x, y = 70 * u, H - 120 * u
        chip(d, ctx, x, y, cap[0], 18, BLACK, cap[2])
        d.text((x, y + 50 * u), cap[1], font=ctx.f("italic", 40), fill=scale_c(WHITE, k), anchor="lm")
    if t >= 88.0:
        img = kinetic(ctx, img, "MASTERY.", t, 90.2, 95.0, W / 2, H * (0.2 if not ctx.portrait else 0.16), 120, "black", WHITE, "slam")
    if 76.0 < t < 86.5 and not ctx.portrait:
        img = kinetic(ctx, img, "ONE SINGLE BREATHING ENTITY", t, 83.3, 87.7, W / 2, H * 0.42, 70, "black", WHITE, "words")
    return img


def _field_pov(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    lt = t - 60.0
    z = 1 + 0.35 * ease_io(lt / 15.6)               # slow push-in
    build = clamp((t - 73.5) / 2.1)
    img = vgrad(W, H, (6, 8, 22), (20, 14, 26))
    lay = layer(ctx)
    sx, sy = W / 2, H * 0.52
    sw, sh = W * 0.12 * z, H * 0.09 * z
    lasers(ctx, lay, t, sx, sy - sh * 0.4, 5, (255, 240, 200), 1.2, intensity=0.25 + 0.5 * build + 0.2 * math.sin(t * 2) ** 2, width=1.5)
    img.paste(lay, (0, 0), lay)
    d = ImageDraw.Draw(img)
    stage_c = mixc((255, 210, 150), WHITE, build)
    strobe = (math.sin(t * 2 * math.pi * 128 / 60 * (1 + 3 * build)) > 0.7) * build
    d.rectangle([sx - sw / 2, sy - sh, sx + sw / 2, sy], fill=scale_c(stage_c, 0.6 + 0.4 * strobe))
    img = add(img, radial(W, H, sx, sy - sh / 2, W * 0.35 * z, scale_c((200, 150, 90), 0.5 + 0.5 * build), 1.8))
    d = ImageDraw.Draw(img)
    crowd_rows(ctx, d, t, W, H, sy + 6 * u, rows=11, color=(10, 8, 10), bob=0.5 + build, hands=0.08 + 0.3 * build, seed=60,
               density=1.1, scale=z)
    if 75.35 <= t < 75.6:
        return canvas(ctx, BLACK)
    # muffled softness (earplug POV) fading as we push in
    soft = (1 - clamp(lt / 14.0)) * 3.5 * u
    if soft > 0.4 * u:
        img = img.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(soft / 4 * 1.5)).resize((W, H), Image.BILINEAR)
    return img


def _mainstage(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    lt = t - 75.6
    z = 1 + 0.12 * lt / 12.4
    img = canvas(ctx, (2, 2, 8))
    lw, lh = W * 0.86 * z, H * 0.44 * z
    wall = led_wall(ctx, t, lw, lh, "festival")
    img.paste(wall, (int((W - lw) / 2), int(H * 0.1 - (z - 1) * H * 0.2)))
    lay = layer(ctx)
    k = ctx.pulse("kick", t, 0.18, "s4")
    for i, (x, c) in enumerate([(0.18, MAGENTA), (0.4, CYAN), (0.6, CYAN), (0.82, MAGENTA)]):
        lasers(ctx, lay, t, W * x, H * 0.55, 6, c, 1.3, phase=i, intensity=0.5 + 0.5 * k)
    img.paste(lay, (0, 0), lay)
    img = add(img, radial(W, H, W / 2, H * 0.55, W * 0.75, scale_c((110, 50, 120), 0.6 + 0.5 * k), 1.2))
    d = ImageDraw.Draw(img)
    d.rectangle([W * 0.3, H * 0.5, W * 0.7, H * 0.58], fill=(12, 12, 20))
    dj_silhouette(d, W / 2, H * 0.52, 30 * u * z, (2, 2, 6), arms=(1.5 + math.sin(t * 6), 0.4), head_bob=math.sin(t * 13.4))
    crowd_rows(ctx, d, t, W, H, H * 0.58, rows=9, color=(4, 4, 10), bob=1.3, hands=0.75, seed=76, density=1.2, scale=z)
    return img


def _mixer_close(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    img = vgrad(W, H, (22, 22, 26), (8, 8, 10))
    d = ImageDraw.Draw(img)
    # brushed panel
    pad = 60 * u
    d.rounded_rectangle([pad, pad, W - pad, H - pad], radius=24 * u, fill=(28, 28, 32), outline=(70, 70, 76), width=max(1, int(3 * u)))
    ch = 4
    cw = (W - 2 * pad) / (ch + 1.6)
    for c in range(ch):
        x = pad + cw * (c + 0.5)
        for r, nm in enumerate(["HI", "MID", "LOW"]):
            ky = pad + 120 * u + r * 130 * u
            R = 44 * u
            d.ellipse([x + cw / 2 - R, ky - R, x + cw / 2 + R, ky + R], fill=(18, 18, 20), outline=(90, 90, 96), width=max(1, int(3 * u)))
            a = math.radians(-135 + 270 * (0.5 + 0.12 * math.sin(c + r)))
            d.line([(x + cw / 2, ky), (x + cw / 2 + math.sin(a) * R * 0.85, ky - math.cos(a) * R * 0.85)], fill=WHITE, width=max(1, int(5 * u)))
            d.text((x + cw / 2 - R - 14 * u, ky), nm, font=ctx.f("mono", 14), fill=GREY, anchor="rm")
        fy0, fy1 = pad + 520 * u, H - pad - 70 * u
        d.rectangle([x + cw / 2 - 4 * u, fy0, x + cw / 2 + 4 * u, fy1], fill=(10, 10, 12))
        fp = fy0 + (fy1 - fy0) * (0.15 + 0.05 * c)
        d.rounded_rectangle([x + cw / 2 - 34 * u, fp - 16 * u, x + cw / 2 + 34 * u, fp + 16 * u], radius=4 * u, fill=(200, 200, 205))
        vu_meter(d, ctx, x + cw - 18 * u, fy0, 12 * u, fy1 - fy0, clamp(ctx.feat("rms", t) * (0.85 + 0.1 * c)), segs=18, horizontal=False)
    # big filter knob following the sweep (88-94.5)
    fx, fy = W - pad - cw * 0.55, pad + 260 * u
    R = 110 * u
    ph = clamp((t - 88.0) / 6.5)
    pos = 1 - math.sin(math.pi * ph)
    a = math.radians(-135 + 270 * (0.08 + 0.9 * pos))
    lay = layer(ctx)
    ld = ImageDraw.Draw(lay)
    ld.ellipse([fx - R * 1.25, fy - R * 1.25, fx + R * 1.25, fy + R * 1.25], outline=GOLD + (200,), width=max(1, int(4 * u)))
    img.paste(lay, (0, 0), lay)
    img = glow(img, 22, ctx, 0.9, 120)
    d = ImageDraw.Draw(img)
    d.ellipse([fx - R, fy - R, fx + R, fy + R], fill=(16, 16, 18), outline=(120, 120, 126), width=max(1, int(4 * u)))
    d.line([(fx, fy), (fx + math.sin(a) * R * 0.9, fy - math.cos(a) * R * 0.9)], fill=GOLD, width=max(1, int(8 * u)))
    d.text((fx, fy + R * 1.55), "FILTER", font=ctx.f("mono", 22), fill=GOLD, anchor="mm")
    # generic hand silhouette on the knob
    hx, hy = fx + math.sin(a) * R * 0.95, fy - math.cos(a) * R * 0.95
    sk = (40, 30, 26)
    d.ellipse([hx - 70 * u, hy - 40 * u, hx + 70 * u, hy + 120 * u], fill=sk)
    d.polygon([(hx - 60 * u, hy + 60 * u), (hx + 60 * u, hy + 60 * u), (hx + 120 * u, H), (hx - 40 * u, H)], fill=sk)
    for i in range(3):
        d.rounded_rectangle([hx - 60 * u + i * 42 * u, hy - 70 * u, hx - 28 * u + i * 42 * u, hy + 20 * u], radius=16 * u, fill=sk)
    return img


# ====================================================================== SCENE 5
def s5(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    morph = clamp((t - 108.0) / 8.0)
    beach = _beach(ctx, t)
    beach = film16(ctx, beach, t, 0.6, frame_i=fi(ctx, t), scratches=False)
    if morph > 0:
        corp = _corporate(ctx, t)
        img = Image.blend(beach, corp, ease_io(morph))
    else:
        img = beach
    img = title_block(ctx, img, t, 95.4, 101.0, "BIG BEACH BOUTIQUE II · 2002", "BRIGHTON BEACH, ~250,000 PEOPLE",
                      size=64 if not ctx.portrait else 54)
    img = _charts(ctx, img, t)
    return img


def _beach(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    img = vgrad(W, int(H * 0.45), (40, 30, 70), (250, 140, 80)).resize((W, int(H * 0.45)))
    full = canvas(ctx, (30, 26, 34))
    full.paste(img, (0, 0))
    full = add(full, radial(W, H, W * 0.62, H * 0.42, W * 0.35, (255, 160, 80), 2.2))
    d = ImageDraw.Draw(full)
    hy = H * 0.45
    # sea
    d.rectangle([0, hy, W, H * 0.58], fill=(40, 46, 72))
    for i in range(26):
        y = hy + (H * 0.13) * (i / 26) ** 1.4
        x = (i * 173 + t * 30 * (1 + i % 3)) % W
        d.line([(x, y), (x + W * 0.08, y)], fill=(250, 170, 110), width=max(1, int(2 * u)))
    # pier silhouette (generic)
    d.rectangle([0, hy - 6 * u, W * 0.24, hy + 10 * u], fill=(16, 12, 18))
    for i in range(14):
        x = i * W * 0.24 / 14
        d.line([(x, hy + 10 * u), (x, H * 0.56)], fill=(16, 12, 18), width=max(1, int(4 * u)))
    for i, x in enumerate([0.04, 0.1, 0.16, 0.21]):
        d.polygon([(W * x - 30 * u, hy - 6 * u), (W * x, hy - 50 * u), (W * x + 30 * u, hy - 6 * u)], fill=(16, 12, 18))
    # stage on the beach
    sx = W * 0.62
    d.rectangle([sx - 120 * u, H * 0.47, sx + 120 * u, H * 0.56], fill=(10, 8, 12))
    d.rectangle([sx - 100 * u, H * 0.48, sx + 100 * u, H * 0.53], fill=scale_c(AMBER, 0.5 + 0.5 * ctx.pulse("kick", t, 0.2)))
    # pebble beach crowd: dense dots
    rng = np.random.default_rng(250)
    by = H * 0.58
    low = ctx.feat("low", t)
    for r in range(36):
        k = r / 36
        y = by + (H - by) * k ** 1.5
        s = (1.5 + 13 * k ** 2) * u
        cnt = int(180 * (1 - k) + 30)
        xs = rng.uniform(0, 1, cnt) * W
        ph = rng.uniform(0, 6.28, cnt)
        c = mixc((70, 50, 60), (12, 10, 14), k)
        for x, p in zip(xs, ph):
            yy = y - (math.sin(t * 8.04 + p) * 0.5 + 0.5) * s * (0.3 + low)
            d.ellipse([x - s, yy - s, x + s, yy + s], fill=c)
    return full


def _corporate(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    img = vgrad(W, H, (8, 10, 18), (2, 2, 6))
    d = ImageDraw.Draw(img)
    for i in range(0, 40):
        x = (i / 40) * W
        d.line([(x, 0), (x, H)], fill=(18, 22, 34), width=1)
    for i in range(0, 24):
        y = (i / 24) * H
        d.line([(0, y), (W, y)], fill=(18, 22, 34), width=1)
    # ticker tape
    items = ["HEADLINER FEE INDEX ▲ 3,900%", "VIP TABLES ▲ 640%", "GA TICKET ▲ 410%", "SPONSOR ACTIVATIONS ▲ 1,200%",
             "FESTIVAL DAYS / YR ▲ 88%", "MERCH ▲ 520%"]
    txt = "     ·     ".join(items) * 3
    fnt = ctx.f("mono", 26)
    x = -((t - 108) * 180 * u) % (fnt.getlength(txt) / 3)
    d.rectangle([0, H - 70 * u, W, H - 20 * u], fill=(14, 16, 24))
    d.text((x, H - 45 * u), txt, font=fnt, fill=GOLD, anchor="lm")
    # $ pops on registers
    for e in ctx.ev("register"):
        dt = t - e["t"]
        if 0 <= dt < 1.0:
            rng = np.random.default_rng(int(e["t"] * 10))
            x, y = rng.uniform(0.1, 0.9) * W, rng.uniform(0.2, 0.6) * H
            k = ease_out(dt / 0.3)
            d.text((x, y - dt * 40 * u), "$", font=ctx.f("black", 60 + 40 * k), fill=scale_c(GOLD, 1 - dt), anchor="mm")
    return img


def _fmt_money(v):
    return f"${v / 1e6:.1f}M" if v >= 1e6 else f"${v / 1e3:.0f}K" if v >= 1e3 else f"${v:.0f}"


def _charts(ctx, img, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    if t < 101.0:
        return img
    lay = layer(ctx)
    d = ImageDraw.Draw(lay)
    show_table = t >= 118.0
    a_charts = window(t, 101.0, 118.4, 0.6, 0.6)
    if a_charts > 0:
        panels = [("AVERAGE ARTIST GUARANTEE · USD (log)", "fee", GOLD), ("CROWD CAPACITY · PEOPLE (log)", "crowd", CYAN)]
        if ctx.portrait:
            boxes = [(60 * u, H * 0.30, W - 60 * u, H * 0.56), (60 * u, H * 0.60, W - 60 * u, H * 0.86)]
        else:
            boxes = [(80 * u, H * 0.30, W * 0.48, H * 0.86), (W * 0.52, H * 0.30, W - 80 * u, H * 0.86)]
        al = int(235 * a_charts)
        for (title, key, col), (x0, y0, x1, y1) in zip(panels, boxes):
            d.rounded_rectangle([x0, y0, x1, y1], radius=16 * u, fill=(8, 8, 14, int(200 * a_charts)), outline=col + (al,), width=max(1, int(2 * u)))
            d.text((x0 + 24 * u, y0 + 34 * u), title, font=ctx.f("mono", 20), fill=col + (al,), anchor="lm")
            lo_v, hi_v = (1e3, 2e6) if key == "fee" else (1e3, 4e5)
            bx0, bx1 = x0 + 40 * u, x1 - 40 * u
            by0, by1 = y0 + 80 * u, y1 - 70 * u
            bw = (bx1 - bx0) / 3
            pts = []
            for i, row in enumerate(C.DATA_TABLE):
                k = ease_out((t - 101.6 - i * 2.4) / 1.6)
                lo, hi = row[f"{key}_lo"], row[f"{key}_hi"]
                f = lambda v: (math.log10(v) - math.log10(lo_v)) / (math.log10(hi_v) - math.log10(lo_v))
                hlo, hhi = f(lo) * (by1 - by0) * k, f(hi) * (by1 - by0) * k
                x = bx0 + bw * (i + 0.2)
                d.rectangle([x, by1 - hlo, x + bw * 0.6, by1], fill=col + (int(al * 0.9),))
                if hhi > hlo + 1:
                    d.rectangle([x, by1 - hhi, x + bw * 0.6, by1 - hlo], fill=col + (int(al * 0.35),))
                pts.append((x + bw * 0.3, by1 - hhi))
                d.text((x + bw * 0.3, by1 + 30 * u), row["era"], font=ctx.f("black", 30), fill=WHITE + (al,), anchor="mm")
                if k > 0.3:
                    txt = row[f"{key}_txt"].split(" (")[0]
                    d.text((x + bw * 0.3, by1 - hhi - 22 * u), txt, font=ctx.f("bold", 24), fill=WHITE + (int(al * k),), anchor="mm")
            if len(pts) == 3 and t > 109:
                kk = ease_out((t - 109) / 1.5)
                seg = [pts[0], (lerp(pts[0][0], pts[1][0], min(1, kk * 2)), lerp(pts[0][1], pts[1][1], min(1, kk * 2)))]
                if kk > 0.5:
                    seg.append((lerp(pts[1][0], pts[2][0], (kk - 0.5) * 2), lerp(pts[1][1], pts[2][1], (kk - 0.5) * 2)))
                d.line(seg, fill=WHITE + (al,), width=max(1, int(4 * u)), joint="curve")
    if show_table:
        a = window(t, 118.0, 135.0, 0.8, 0.4)
        al = int(255 * a)
        cols = ["ERA", "KEY METRIC", "AVG. ARTIST GUARANTEE", "CROWD CAPACITY"]
        if ctx.portrait:
            x0, x1, y0 = 40 * u, W - 40 * u, H * 0.3
        else:
            x0, x1, y0 = W * 0.07, W * 0.93, H * 0.36
        cw = [0.22, 0.28, 0.24, 0.26]
        rh = 120 * u
        fs = 30 if not ctx.portrait else 19
        d.rounded_rectangle([x0 - 20 * u, y0 - 70 * u, x1 + 20 * u, y0 + rh * 3.6 + 40 * u], radius=16 * u, fill=(4, 4, 10, int(225 * a)))
        xs = [x0]
        for c in cw[:-1]:
            xs.append(xs[-1] + (x1 - x0) * c)
        for i, c in enumerate(cols):
            d.text((xs[i], y0 - 30 * u), c, font=ctx.f("mono", fs - 4), fill=GOLD + (al,), anchor="lm")
        d.line([(x0, y0 - 6 * u), (x1, y0 - 6 * u)], fill=GOLD + (al,), width=max(1, int(2 * u)))
        for r, row in enumerate(C.DATA_TABLE):
            k = ease_out((t - 118.6 - r * 0.7) / 0.6)
            y = y0 + rh * (r + 0.6)
            vals = [f"{row['era']} ({row['label']})", row["metric"], row["fee_txt"], row["crowd_txt"]]
            for i, v in enumerate(vals):
                f = ctx.f("black" if i == 0 else "semi", fs + (6 if i == 2 else 0))
                d.text((xs[i] + (1 - k) * 30 * u, y), v, font=f, fill=(WHITE if i != 2 else GOLD) + (int(al * k),), anchor="lm")
            d.line([(x0, y + rh / 2), (x1, y + rh / 2)], fill=(60, 60, 80, al), width=1)
        d.text((x0, y0 + rh * 3.4), "Figures: production research estimates, rounded. Crowd figure as widely reported.",
               font=ctx.f("mono_r", 18), fill=(160, 160, 170, al), anchor="lm")
    img.paste(lay, (0, 0), lay)
    if t > 127.0:
        img = kinetic(ctx, img, "THE SUPERSTAR DJ: LIVE ENTERTAINMENT'S MOST LUCRATIVE ASSET", t, 127.2, 134.8,
                      W / 2, H * (0.13 if not ctx.portrait else 0.18), 44, "black", WHITE, "words", maxw=W * 0.85,
                      accent=(GOLD, {"LUCRATIVE", "ASSET"}))
    return img


# ====================================================================== SCENE 6
STEP_COL = [MAGENTA, AMBER, CYAN, GOLD]


def s6(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    P = ctx.portrait
    img = vgrad(W, H, (6, 8, 20), (2, 2, 8))
    d = ImageDraw.Draw(img)
    # background grid
    gs = 60 * u
    for x in np.arange(0, W, gs):
        d.line([(x, 0), (x, H)], fill=(14, 18, 36), width=1)
    for y in np.arange(0, H, gs):
        d.line([(0, y), (W, y)], fill=(14, 18, 36), width=1)
    step = next((s for s in C.STEPS if s["start"] <= t < s["end"]), C.STEPS[-1])
    si = step["n"] - 1
    col = STEP_COL[si]
    lt = t - step["start"]
    # header
    d.text((60 * u, 70 * u), "THE FATBOY SLIM PLAYBOOK", font=ctx.f("black", 44 if not P else 40), fill=WHITE, anchor="lm")
    d.text((60 * u, 112 * u), "TACTICAL TUTORIAL  ·  LIVE 90-SECOND MINI-MIX", font=ctx.f("mono", 18), fill=GREY, anchor="lm")
    if not P:
        d.text((W - 60 * u, 70 * u), _bpm_str(t), font=ctx.f("mono", 34), fill=col, anchor="rm")
        d.text((W - 60 * u, 112 * u), f"KEY Am  ·  {_tc(t - 135.0)}", font=ctx.f("mono", 18), fill=GREY, anchor="rm")
    # step pills
    pills_y = 180 * u if not P else 200 * u
    n_ = 4
    pw = (W - 120 * u - (n_ - 1) * 20 * u) / n_
    names = ["VOCAL HOOK", "ACAPELLA / BREAK", "MINI-DROP", "CRESCENDO"]
    for i in range(n_):
        x = 60 * u + i * (pw + 20 * u)
        active, done = i == si, i < si
        c = STEP_COL[i]
        fillc = scale_c(c, 0.95) if active else scale_c(c, 0.22 if done else 0.08)
        d.rounded_rectangle([x, pills_y - 32 * u, x + pw, pills_y + 32 * u], radius=32 * u, fill=fillc,
                            outline=scale_c(c, 0.9 if active or done else 0.4), width=max(1, int(2 * u)))
        d.text((x + pw / 2, pills_y), f"0{i + 1}  {names[i]}" if not P else f"0{i + 1}", font=ctx.f("bold", 22),
               fill=BLACK if active else scale_c(WHITE, 0.8 if done else 0.45), anchor="mm")
    # overview waveform timeline (whole mini-mix)
    oy = pills_y + 70 * u
    ox0, ox1, oh = 60 * u, W - 60 * u, 70 * u
    for i, s in enumerate(C.STEPS):
        xa = ox0 + (ox1 - ox0) * (s["start"] - 135) / 90
        xb = ox0 + (ox1 - ox0) * (s["end"] - 135) / 90
        waveform(d, ctx, xa, oy, xb - xa, oh, s["start"], s["end"], scale_c(STEP_COL[i], 0.95), playhead=t,
                 dim=scale_c(STEP_COL[i], 0.3), mute=_mute)
    phx = ox0 + (ox1 - ox0) * (t - 135) / 90
    d.line([(phx, oy - 8 * u), (phx, oy + oh + 8 * u)], fill=WHITE, width=max(1, int(3 * u)))
    # step body
    by0 = oy + oh + 40 * u
    by1 = H * (0.74 if not P else 0.70)
    body = [_step1, _step2, _step3, _step4][si]
    img = body(ctx, img, t, lt, (60 * u, by0, W - 60 * u, by1), col)
    d = ImageDraw.Draw(img)
    # step title + action text
    ty = by1 + (60 if not P else 70) * u
    k = ease_out(lt / 0.6)
    d.text((60 * u + (1 - k) * 40 * u, ty), f"STEP {step['n']}:", font=ctx.f("mono", 26), fill=scale_c(col, k), anchor="lm")
    tf = ctx.f("black", 50 if not P else 44)
    lines = wrap(tf, step["title"], W - 120 * u - (0 if P else 200 * u))
    for i, ln in enumerate(lines):
        d.text((60 * u + (1 - k) * 60 * u, ty + (52 + i * 54) * u), ln, font=tf, fill=scale_c(WHITE, k), anchor="lm")
    if not P:
        img = kinetic(ctx, img, "ACTION: " + step["action"], t, step["start"] + 0.8, step["end"] - 0.2,
                      60 * u, ty + (52 + len(lines) * 54 + 26) * u, 26, "med", (200, 205, 220), "type", maxw=W * 0.8,
                      align="left", shadow=False)
    # strobes on mini-drops (step 3)
    if si == 2:
        for e in ctx.ev("minidrop"):
            if e["t"] <= t < e["end"]:
                beat = 60 / 130
                ph = ((t - e["t"]) % (beat / 2)) / (beat / 2)
                img = flash(img, 0.85 if ph < 0.35 else 0.0)
    if si == 3:
        img = flash(img, ctx.pulse("peak", t, 0.5) * 0.9, (255, 220, 160))
    img = glow(img, 20, ctx, 1.0, 60)
    img = ImageChops.multiply(img, ctx.vignette(0.3))
    return img


def _mute(tt):
    for e_t in [180.0 + k * 60 / 130 for k in C.MINI_DROP_BEATS]:
        if e_t <= tt < e_t + 2 * 60 / 130:
            return True
    return False


def _bpm_str(t):
    if t < 157.5:
        for a, b, bpm, g in C.HOOK_SECTIONS:
            if a <= t < b:
                return f"{bpm} BPM · {g}"
    return "130 BPM · BIG BEAT"


def _tc(s):
    s = max(0, s)
    return f"{int(s // 60):02d}:{int(s % 60):02d}:{int((s * 24) % 24):02d}"


def _lane(d, ctx, x0, y, x1, h, name, col, t, t0, t1, k=1.0, mute=None):
    u = ctx.u
    d.rounded_rectangle([x0, y, x1, y + h], radius=10 * u, fill=scale_c(col, 0.07 * k), outline=scale_c(col, 0.7 * k), width=max(1, int(2 * u)))
    d.text((x0 + 18 * u, y + 22 * u), name, font=ctx.f("mono", 16), fill=scale_c(col, k), anchor="lm")
    waveform(d, ctx, x0 + 10 * u, y + 30 * u, x1 - x0 - 20 * u, h - 40 * u, t0, t1, scale_c(col, k), mute=mute)


def _step1(ctx, img, t, lt, box, col):
    x0, y0, x1, y1 = box
    u = ctx.u
    d = ImageDraw.Draw(img)
    h = y1 - y0
    # isolated vocal lane
    _lane(d, ctx, x0, y0, x1, h * 0.38, "VOCAL · ISOLATED", col, t, t - 4, t)
    hk = ctx.pulse("hook", t, 0.35)
    d.text(((x0 + x1) / 2, y0 + h * 0.62), "“MAKE SOME NOISE!”", font=ctx.f("black", 68 + 14 * hk), fill=mixc(WHITE, col, hk), anchor="mm")
    secs = C.HOOK_SECTIONS
    cw = (x1 - x0) / len(secs)
    for i, (a, b, bpm, g) in enumerate(secs):
        on = a <= t < b
        cx = x0 + cw * (i + 0.5)
        d.text((cx, y0 + h * 0.86), f"{bpm}", font=ctx.f("black", 54), fill=col if on else (60, 60, 80), anchor="mm")
        d.text((cx, y0 + h * 0.97), f"BPM · {g}", font=ctx.f("mono", 18), fill=WHITE if on else (60, 60, 80), anchor="mm")
    return img


def _step2(ctx, img, t, lt, box, col):
    x0, y0, x1, y1 = box
    u = ctx.u
    d = ImageDraw.Draw(img)
    h = y1 - y0
    c = ease_io((lt - 4.0) / 8.0)        # collapse 0 → 1
    lh = h * 0.36
    ya = lerp(y0, y0 + h * 0.32, c)
    yb = lerp(y0 + h * 0.62, y0 + h * 0.32, c)
    acap_k = clamp((t - (ctx.ev("acapella") or [{"t": 161.2}])[0]["t"]) / 0.6)
    if c < 0.98:
        _lane(d, ctx, x0, ya, x1, lh, "CH A · ACAPELLA · 1970s SOUL/GOSPEL", VIOLET, t, t - 4, t, k=(1 - c) * max(0.25, acap_k))
        _lane(d, ctx, x0, yb, x1, lh, "CH B · FUNKY BREAKBEAT · 130 BPM", col, t, t - 4, t, k=1 - c)
    if c > 0.02:
        lay = layer(ctx)
        ld = ImageDraw.Draw(lay)
        _lane(ld, ctx, x0, y0 + h * 0.32, x1, lh, "A + B · HARMONIC TENSION", mixc(VIOLET, col, 0.5), t, t - 4, t)
        img.paste(alpha_mul(lay, c), (0, 0), alpha_mul(lay, c))
    return img


def _step3(ctx, img, t, lt, box, col):
    x0, y0, x1, y1 = box
    u = ctx.u
    d = ImageDraw.Draw(img)
    h = y1 - y0
    _lane(d, ctx, x0, y0, x1, h * 0.5, "MASTER · LOW END", col, t, t - 5, t, mute=_mute)
    cut = _mute(t)
    d.text(((x0 + x1) / 2, y0 + h * 0.75), "LOW END: CUT" if cut else "LOW END: ON", font=ctx.f("black", 64),
           fill=WHITE if cut else scale_c(col, 0.8), anchor="mm")
    # EQ low kill knob
    R = 46 * u
    kx, ky = x1 - 120 * u, y0 + h * 0.75
    a = math.radians(-135 + 270 * (0.0 if cut else 0.5))
    d.ellipse([kx - R, ky - R, kx + R, ky + R], fill=(14, 14, 20), outline=col, width=max(1, int(3 * u)))
    d.line([(kx, ky), (kx + math.sin(a) * R * 0.85, ky - math.cos(a) * R * 0.85)], fill=WHITE, width=max(1, int(6 * u)))
    d.text((kx, ky + R + 20 * u), "LOW", font=ctx.f("mono", 16), fill=GREY, anchor="mm")
    return img


def _step4(ctx, img, t, lt, box, col):
    x0, y0, x1, y1 = box
    u = ctx.u
    d = ImageDraw.Draw(img)
    h = y1 - y0
    P = ctx.portrait
    lanes = [("HOOK", MAGENTA), ("ACAPELLA", VIOLET), ("BREAK", AMBER), ("KICK", RED), ("SUPERSAW", CYAN)]
    xr = x1 if P else x0 + (x1 - x0) * 0.62
    lh = h / len(lanes) - 8 * u
    for i, (nm, c) in enumerate(lanes):
        k = clamp((lt - i * 0.8) / 0.6)
        _lane(d, ctx, x0, y0 + i * (lh + 8 * u), xr, lh, nm, c, t, t - 3, t, k=max(0.15, k))
    if not P:
        # stage mock-up
        sx0, sx1 = xr + 40 * u, x1
        sy0, sy1 = y0, y1
        peak = ctx.pulse("peak", t, 1.2)
        build = clamp(lt / 19.3)
        d.rounded_rectangle([sx0, sy0, sx1, sy1], radius=12 * u, fill=(4, 4, 10), outline=(60, 60, 80), width=max(1, int(2 * u)))
        sw = sx1 - sx0
        wall = led_wall(ctx, t, sw * 0.8, (sy1 - sy0) * 0.35, "festival")
        img.paste(wall, (int(sx0 + sw * 0.1), int(sy0 + (sy1 - sy0) * 0.12)))
        img = add(img, radial(ctx.W, ctx.H, (sx0 + sx1) / 2, sy0 + (sy1 - sy0) * 0.45, sw * (0.4 + 0.9 * peak),
                              scale_c(GOLD, 0.25 + 0.5 * build + 0.9 * peak), 1.4))
        d = ImageDraw.Draw(img)
        crowd_rows(ctx, d, t, sw, sy1 - sy0, (sy1 - sy0) * 0.62, rows=4, color=(4, 4, 8), bob=1.2, hands=0.5 + 0.5 * build,
                   seed=99, density=0.35, x_off=0, scale=0.6) if False else None
        rng = np.random.default_rng(5)
        for i in range(60):
            x = sx0 + rng.uniform(0.05, 0.95) * sw
            yy = sy0 + (sy1 - sy0) * rng.uniform(0.66, 0.95)
            s = 6 * u * rng.uniform(0.7, 1.3)
            yy -= (math.sin(t * 8.4 + i) * 0.5 + 0.5) * s * 0.6
            head_silhouette(d, x, yy, s, (2, 2, 6), hands=build * rng.uniform(0.3, 1))
        d.text(((sx0 + sx1) / 2, sy1 - 22 * u), "STAGE VISUAL MOCK-UP", font=ctx.f("mono", 14), fill=GREY, anchor="mm")
    return img


# ====================================================================== SCENE 7
def s7(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    if t >= 257.5:
        return _title_card(ctx, t)
    lt = t - 225.0
    slow = 0.35
    z = 1.5 - 0.5 * ease_io(lt / 32.0)          # slow-motion pull-back
    img = vgrad(W, H, (4, 6, 18), (14, 8, 20))
    # ocean of illuminated hands
    lay = layer(ctx)
    d = ImageDraw.Draw(lay)
    rng = np.random.default_rng(270)
    hz = H * (0.32 - 0.1 * (z - 1))
    for r in range(28):
        k = r / 28
        y = hz + (H - hz) * k ** 1.4
        s = (2 + 16 * k ** 1.6) * u * z
        cnt = int(110 * (1 - k) + 20)
        for _ in range(cnt):
            x = (rng.uniform(-0.2, 1.2) - 0.5) * W * z + W / 2
            p = rng.uniform(0, 6.28)
            yy = y - (math.sin(t * slow * 6 + p) * 0.5 + 0.5) * s * 1.2
            warm = rng.uniform() < 0.5
            c = (255, 214, 150) if warm else (190, 220, 255)
            a = int(140 + 110 * math.sin(t * slow * 3 + p) ** 2)
            d.ellipse([x - s * 0.5, yy - s * 0.9, x + s * 0.5, yy + s * 0.2], fill=c + (a,))
            d.line([(x, yy + s * 0.2), (x - s * 0.2, yy + s * 3)], fill=(10, 8, 14, 255), width=max(1, int(s * 0.5)))
    img.paste(lay, (0, 0), lay)
    img = glow(img, 30, ctx, 1.3, 30)
    d = ImageDraw.Draw(img)
    # booth + DJ from behind (foreground), revealed by pull-back
    fz = 1 + (z - 1) * 1.6
    by = H * (0.78 + 0.25 * (fz - 1))
    dj_silhouette(d, W * 0.44, by, 78 * u * fz, (2, 2, 4), arms=(0.25, 1.75 + 0.15 * math.sin(t * slow * 4)), phones=True)
    booth(d, ctx, -W * 0.1, by, W * 1.2, H, (3, 3, 6), (30, 28, 40))
    for i, x in enumerate([0.12, 0.88]):
        R = 140 * u * fz
        cx, cy = W * x, by + 90 * u * fz
        d.ellipse([cx - R, cy - R * 0.4, cx + R, cy + R * 0.4], outline=(50, 46, 60), width=max(1, int(3 * u)))
    img = digital(ctx, img, 0.6, frame_i=fi(ctx, t), grain=0.18)
    # VO pull-quotes
    P = ctx.portrait
    if not P:
        img = kinetic(ctx, img, "It's not about pushing buttons.", t, 237.6, 242.4, W / 2, H * 0.2, 60, "black", WHITE, "words")
        for i, (a, txt) in enumerate([(243.0, "CONTROL THE ROOM."), (245.4, "RECLAIM THE SOUL OF THE SAMPLE."),
                                      (248.6, "ENGINEER COLLECTIVE HUMAN JOY.")]):
            img = kinetic(ctx, img, txt, t, a, 256.8, W / 2, H * (0.16 + i * 0.09), 54, "black",
                          [CYAN, AMBER, GOLD][i], "words")
    return img


def _title_card(ctx, t):
    u, W, H = ctx.u, ctx.W, ctx.H
    lt = t - 257.5
    img = canvas(ctx, (250, 248, 242))
    d = ImageDraw.Draw(img)
    k = back_out(lt / 0.45)
    size = (300 if not ctx.portrait else 230) * (1.6 - 0.6 * clamp(k, 0, 1.2))
    lay = layer(ctx)
    ld = ImageDraw.Draw(lay)
    ld.text((W / 2, H * 0.44), C.TITLE, font=ctx.f("black", size), fill=(8, 8, 10, 255), anchor="mm")
    # offset 'remix' ghost (misregistration)
    ghost = layer(ctx)
    gd = ImageDraw.Draw(ghost)
    off = (1 - clamp(lt / 0.8)) * 30 * u + 6 * u
    gd.text((W / 2 + off, H * 0.44 + off * 0.4), C.TITLE, font=ctx.f("black", size), fill=RED + (200,), anchor="mm")
    img.paste(ghost, (0, 0), ghost)
    img.paste(lay, (0, 0), lay)
    d = ImageDraw.Draw(img)
    a = clamp((lt - 0.7) / 0.8)
    for i, ln in enumerate(wrap(ctx.f("semi", 34), C.SUBTITLE.upper(), W * 0.86)):
        d.text((W / 2, H * 0.62 + i * 44 * u), ln, font=ctx.f("semi", 34), fill=mixc((250, 248, 242), (20, 20, 24), a), anchor="mm")
    a2 = clamp((lt - 4.0) / 0.8)
    if a2 > 0:
        fnt = ctx.f("mono_r", 19 if not ctx.portrait else 17)
        for i, ln in enumerate(wrap(fnt, C.DISCLAIMER, W * 0.8)):
            d.text((W / 2, H * 0.82 + i * 28 * u), ln, font=fnt, fill=mixc((250, 248, 242), (70, 70, 76), a2), anchor="mm")
    # fade to black at the very end
    fade = clamp((t - (C.DURATION - 0.75)) / 0.75)
    if fade > 0:
        img = Image.blend(img, canvas(ctx, BLACK), fade)
    return img


# ====================================================================== dispatcher + global overlays
SCENE_FN = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7}

GLOBAL_TEXT = [  # (a, b, text, style, y_frac, size, color, accent words)
    (1.0, 4.8, "NOSTALGIA IS A COMFORTABLE LIE.", "words", 0.5, 76, WHITE, None),
    (6.6, 9.1, "DID IT EVOLVE —", "slam", 0.24, 64, WHITE, None),
    (9.4, 12.8, "— OR GET COMMERCIALIZED TO DEATH?", "slam", 0.5, 64, WHITE, (RED, {"COMMERCIALIZED", "DEATH?"})),
]


def act_card(ctx, img, t):
    for name, title, a, _ in C.ACTS[1:]:
        vis = window(t, a + 0.2, a + 3.8, 0.4, 0.5)
        if vis > 0:
            d = ImageDraw.Draw(img)
            u = ctx.u
            x, y = 60 * u, (ctx.H - 110 * u) if ctx.portrait else 60 * u
            w = chip(d, ctx, x, y, name, 20, BLACK, scale_c(AMBER, vis))
            d.text((x + w + 16 * u, y), title.upper(), font=ctx.f("bold", 26), fill=scale_c(WHITE, vis), anchor="lm")
    return img


def render_frame(ctx, t, captions=None):
    s = C.scene_at(t)
    img = SCENE_FN[s.id](ctx, t)
    if not ctx.portrait:
        for a, b, txt, style, yf, size, colr, acc in GLOBAL_TEXT:
            if a <= t <= b:
                img = kinetic(ctx, img, txt, t, a, b, ctx.W / 2, ctx.H * yf, size, "black", colr, style, accent=acc)
        img = scene_tag(ctx, img, t, s)
    img = act_card(ctx, img, t)
    # scene-boundary glitch cuts (except the designed black gap and the title)
    for sc in C.SCENES[1:]:
        dt = t - sc.start
        if -2 / ctx.fps <= dt < 3 / ctx.fps and sc.id not in ("s3",):
            img = glitch(ctx, img, 0.8 * (1 - abs(dt) * ctx.fps / 3), t)
    if 34.45 <= t < 35.0:                       # the withheld drop
        img = canvas(ctx, BLACK)
    if captions:
        img = captions(ctx, img, t)
    return img
