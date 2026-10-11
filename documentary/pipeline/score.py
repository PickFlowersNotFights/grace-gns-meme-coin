"""Original score + sound design for the full 4:30 timeline.

Produces two stereo stems (music, sfx) and an event log (kicks, snares, strobes,
CO2 blasts, drops...) that the visual renderer uses for frame-accurate sync.
No third-party recordings are used: every sound is synthesised here.
"""
import numpy as np

from . import config as C
from . import synth as S
from .synth import SR, n, mtof

DUR = C.DURATION


class Bus:
    def __init__(self, dur):
        self.x = np.zeros((n(dur) + SR, 2))

    def put(self, x, t, g=1.0, pan=0.0):
        if t < 0:
            x = x[n(-t):]
            t = 0
        o = n(t)
        if o >= len(self.x):
            return
        x = np.asarray(x, dtype=np.float64)
        L = min(len(x), len(self.x) - o)
        if x.ndim == 1:
            a = (pan + 1) * np.pi / 4
            self.x[o:o + L, 0] += x[:L] * g * np.cos(a) * 1.414
            self.x[o:o + L, 1] += x[:L] * g * np.sin(a) * 1.414
        else:
            self.x[o:o + L] += x[:L] * g


class Events:
    def __init__(self):
        self.e = {}

    def add(self, kind, t, **kw):
        self.e.setdefault(kind, []).append(dict(t=round(float(t), 4), **kw))


# ------------------------------------------------------------------ kits
def make_kits():
    rng = np.random.default_rng(5)
    raw_k = S.bitcrush(S.lp(S.kick(0.4, 120, 50, click=0.5, drive=2.5), 5000), 11, 2)
    raw_s = S.bitcrush(S.reverb(S.snare(0.35, 200, snap=1.2), 0.6, 0.22, 4000), 11, 2)
    kits = {
        "raw": dict(k=raw_k, s=raw_s, h=S.lp(S.hat(), 9000), o=S.lp(S.hat(open_=True), 9000),
                    c=S.clap(), t=S.tom(120)),
        "edm": dict(k=S.kick(0.5, 180, 48, click=0.6, drive=2.2), s=S.clap() * 1.2, h=S.hat(),
                    o=S.hat(open_=True), c=S.clap(), t=S.tom(140)),
        "ai": dict(k=S.sine(52, 0.3) * S.expdecay(0.3, 0.12) * 0.9, s=S.hp(S.noise(0.15, rng), 2000) * S.expdecay(0.15, 0.03) * 0.5,
                   h=S.hat() * 0.6, o=S.hat() * 0.6, c=S.clap() * 0.6, t=S.tom(100)),
        "boom": dict(k=S.lp(S.kick(0.55, 110, 44, click=0.3, drive=3.0), 3000),
                     s=S.reverb(S.snare(0.4, 170, snap=1.4, body=0.9), 0.9, 0.3, 5000),
                     h=S.lp(S.hat(), 7000), o=S.hat(open_=True), c=S.clap(), t=S.tom(90)),
        "dnb": dict(k=S.kick(0.3, 200, 55, click=0.7, drive=2.0), s=S.snare(0.25, 230, snap=1.4, body=0.5),
                    h=S.hat(0.04), o=S.hat(open_=True), c=S.clap(), t=S.tom(160)),
    }
    return kits


GAIN = {"k": 0.95, "s": 0.7, "h": 0.32, "o": 0.28, "c": 0.6, "t": 0.5}
PAN = {"k": 0, "s": 0.04, "h": 0.3, "o": -0.25, "c": -0.05, "t": 0.2}


def drums(bus, ev, kit, t0, t1, bpm, pat, swing=0.5, human=0.0, rng=None, gain=1.0, tag="", mute=None, off=0.0):
    """Step sequencer. pat: {inst: 'x..o' 16 steps per bar, may span bars ('|' ignored)}.

    mute(t) -> True suppresses k (used for mini-drops). off: bus start on the master timeline.
    """
    rng = rng or np.random.default_rng(int(t0 * 100))
    step = 60 / bpm / 4
    nsteps = int(np.floor((t1 - t0) / step + 1e-6))
    for inst, p in pat.items():
        p = p.replace("|", "").replace(" ", "")
        for i in range(nsteps):
            ch = p[i % len(p)]
            if ch == ".":
                continue
            t = t0 + i * step
            if i % 2 == 1:
                t += (swing - 0.5) * 2 * step
            t += rng.normal(0, human) if human else 0
            vel = {"x": 1.0, "X": 1.25, "o": 0.38}[ch] * (1 + (rng.normal(0, human * 8) if human else 0))
            if mute and inst == "k" and mute(t):
                continue
            bus.put(kit[inst], t, GAIN[inst] * vel * gain, PAN[inst])
            if inst == "k":
                ev.add("kick", off + t, v=vel, tag=tag)
            elif inst in "sc":
                ev.add("snare", off + t, v=vel, tag=tag)
            elif inst == "h" and ch != "o":
                ev.add("hat", off + t, tag=tag)
    for b in range(int(np.floor((t1 - t0) / (step * 4) + 1e-6))):
        ev.add("beat", off + t0 + b * step * 4, bpm=bpm, tag=tag)


def bassline(bus, t0, t1, bpm, notes, kind="acid", gain=1.0, mute=None, pan=0.0):
    """notes: list of (step, len_steps, midi[, accent]) per bar (16 steps)."""
    step = 60 / bpm / 4
    bar = step * 16
    nbars = int(np.ceil((t1 - t0) / bar))
    cache = {}
    for b in range(nbars):
        for nt in notes:
            s, ln, m = nt[:3]
            acc = len(nt) > 3 and nt[3]
            t = t0 + b * bar + s * step
            if t >= t1:
                continue
            if mute and mute(t):
                continue
            d = min(ln * step * 0.95, t1 - t)
            key = (m, round(d, 3), acc, kind)
            if key not in cache:
                cache[key] = S.acid_note(mtof(m), d, acc) if kind == "acid" else S.bass_note(mtof(m), d)
            bus.put(cache[key], t, gain, pan)


def chords(bus, t0, bpm, prog, beats_each, fn, gain=1.0, pump=None, width=0.5):
    """Render chord progression with fn(freqs, dur) and optional sidechain pump."""
    beat = 60 / bpm
    for i, ch in enumerate(prog):
        d = beats_each * beat
        x = fn(mtof(ch), d + 0.05)
        if pump:
            tt = S.t_axis(d + 0.05) % beat
            x = x * (1 - pump * np.exp(-tt / 0.11))
        t = t0 + i * d
        bus.put(x, t, gain * 0.7, -width)
        bus.put(np.roll(x, 240), t, gain * 0.7, width)


def section(dur):
    return Bus(dur)


def apply_stereo(x, fn):
    return np.stack([fn(x[:, 0]), fn(x[:, 1])], axis=1)


# ------------------------------------------------------------------ patterns
FUNK = {"k": "x.x.......x..x..|..x.......x.....", "s": "....x..o.o..x..o|.o..x..o....x..o",
        "h": "xoxoxoxoxoxoxoxo"}
BIGBEAT = {"k": "x.....x...x..x..|x.....x...x.....", "s": "....X.......X...|....X.......X..o",
           "h": "x.xox.xox.xox.xo", "o": "..............x."}
EDM = {"k": "x...x...x...x...", "c": "....x.......x...", "o": "..x...x...x...x.", "h": "x.x.x.x.x.x.x.x."}
BOOM = {"k": "x......x..x.....|x.x....x..x..x..", "s": "....x.......x...", "h": "x.x.x.x.x.x.x.x."}
HOUSE = {"k": "x...x...x...x...", "c": "....x.......x...", "o": "..x...x...x...x.", "h": "xxxxxxxxxxxxxxxx"}
DNB = {"k": "x.x.......xx....|..x.......x.....", "s": "....x..o.o..x..o|.o..x..o.o..x..o", "h": "x.x.x.x.x.x.x.x."}
AI = {"k": "x...x...x...x...", "h": "xxxxxxxxxxxxxxxx", "s": "....x.......x..."}

ACID = [(0, 1, 45, 1), (2, 1, 45), (3, 1, 57, 1), (5, 1, 45), (6, 1, 48), (8, 1, 45, 1), (10, 1, 55),
        (11, 1, 57), (12, 1, 45), (14, 1, 52, 1), (15, 1, 50)]
ROCK = [(0, 3, 33), (3, 1, 33), (6, 2, 40), (8, 3, 33), (11, 1, 36), (12, 2, 38), (14, 2, 40)]
FUNKBASS = [(0, 2, 33), (3, 1, 45), (4, 1, 33), (7, 1, 36), (8, 2, 38), (10, 1, 40), (11, 1, 43), (13, 2, 33)]
REESE = [(0, 6, 33), (6, 2, 36), (8, 6, 31), (14, 2, 28)]
AM, F, Cc, G, E7 = [57, 60, 64], [53, 57, 60], [55, 60, 64], [55, 59, 62], [52, 56, 59, 62]
GOSPEL = [[57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 65], [52, 56, 59, 62]]


# ------------------------------------------------------------------ main composer
def compose(hook_audio, log=print):
    music, sfx, ev = Bus(DUR), Bus(DUR), Events()
    kits = make_kits()
    rng = np.random.default_rng(1998)

    # ---------------- SCENE 1  (0-10) ignition
    log("  score: scene 1")
    sfx.put(S.fade(S.vinyl_crackle(10.6), 0.6, 0.4), 0.0, 0.55)
    sfx.put(S.needle_drop(), 1.6, 0.8)
    ev.add("needle", 1.6)
    b = Bus(10.0)
    drums(b, ev, kits["raw"], 2.0, 4.0, 128, {"k": "x...x...x...x..."}, tag="s1")
    drums(b, ev, kits["raw"], 4.0, 6.6, 128, BIGBEAT, swing=0.56, human=0.004, tag="s1-98")
    bassline(b, 4.0, 6.6, 128, ROCK, kind="bass", gain=0.7)
    muff = apply_stereo(b.x, lambda c: S.lp(c, 220, 4))
    wall = apply_stereo(b.x, lambda c: S.lp(c, 1400, 2))
    tt = np.arange(len(b.x)) / SR
    mix1 = np.where((tt < 4.0)[:, None], muff * 1.6, wall)
    music.put(mix1[: n(6.6)], 0.0, 0.9)
    m2 = Bus(3.4)
    drums(m2, ev, kits["edm"], 0.0, 3.4, 128, EDM, tag="s1-26", off=6.6)
    chords(m2, 0.0, 128, [AM, F, Cc], 2, lambda f, d: S.supersaw(f, d, cutoff=5000) * S.adsr(d, 0.01, 0.2, 0.7, 0.1), 0.5, pump=0.7)
    music.put(m2.x[: n(2.6)], 6.6, 0.8)
    sfx.put(S.crowd(2.8, 0.8), 6.6, 0.5)
    pre = music.x[n(6.0):n(9.0), 0] + music.x[n(6.0):n(9.0), 1]
    sfx.put(S.tape_rewind(pre, 1.0), 9.0, 0.8)
    music.x[n(9.15):n(10.0)] *= np.linspace(1, 0, n(10.0) - n(9.15))[:, None] ** 2
    for t in (1.6, 4.0, 6.6, 9.2):
        ev.add("cut", t)

    # ---------------- SCENE 2  (10-35) contrarian dashboard
    log("  score: scene 2")
    sfx.put(S.impact(2.0), 10.0, 0.6)
    b = Bus(12.5)
    drums(b, ev, kits["raw"], 0.0, 12.0, 112, FUNK, swing=0.6, human=0.006, tag="s2-98", off=10.0)
    bassline(b, 0.0, 12.0, 112, FUNKBASS, kind="bass", gain=0.75)
    beat = 60 / 112
    for i in range(int(12 / (beat * 2))):
        ch = [[57, 60, 64, 67], [55, 59, 62, 65]][(i // 4) % 2]
        b.put(S.organ_stab(mtof(ch), 0.3), i * beat * 2 + beat * 1.5, 0.45, 0.2)
    sec = S.reverb(b.x[:, 0] * 0.5 + b.x[:, 1] * 0.5, 0.7, 0.12, 6000, stereo_out=True)
    sec = apply_stereo(sec, lambda c: S.bitcrush(S.lp(c, 9000), 12, 1))
    # tape-stop into the EDM build
    ts = n(11.5)
    stop_len = n(1.1)
    src = sec[ts - stop_len * 2: ts + stop_len]
    rate = np.linspace(1, 0.05, stop_len)
    pos = np.clip(np.cumsum(rate).astype(int), 0, len(src) - 1)
    sec[ts:ts + stop_len] = src[pos] * np.linspace(1, 0, stop_len)[:, None]
    sec[ts + stop_len:] = 0
    music.put(sec, 10.0, 0.95)
    sfx.put(S.riser(2.0, 300, 3000), 21.0, 0.25)
    b = Bus(11.0)
    t0 = 0.0
    drums(b, ev, kits["edm"], t0, 11.0, 128, EDM, tag="s2-26", off=23.5)
    chords(b, 0, 128, [AM, F, Cc, G, AM, F], 4, lambda f, d: S.supersaw(f, d, cutoff=6500) * S.adsr(d, 0.01, 0.1, 0.8, 0.05), 0.55, pump=0.75)
    # accelerating snare roll + riser
    beat = 60 / 128
    t = 5.5
    k = 0
    while t < 10.95:
        prog = (t - 5.5) / 5.45
        b.put(kits["edm"]["s"], t, 0.25 + 0.5 * prog, 0.0)
        ev.add("snare", 23.5 + t, v=0.5 + prog, tag="roll")
        t += beat / (1 if prog < 0.35 else 2 if prog < 0.7 else 4)
        k += 1
    b.put(S.riser(5.5, 200, 5000), 5.5, 0.4)
    sec = apply_stereo(b.x, lambda c: S.soft(c * 1.8, 2.5) * 0.6)    # over-compressed
    sec[n(10.95):] = 0
    music.put(sec, 23.5, 0.9)
    for t in (25.0, 28.75, 32.5):
        sfx.put(S.fade(S.hp(S.noise(1.4), 900) * S.expdecay(1.4, 0.4), 0.02, 0.3), t, 0.28, rng.uniform(-0.4, 0.4))
        ev.add("co2", t)
    ev.add("cue_button", 23.5)
    ev.add("drop_withheld", 34.45)

    # ---------------- SCENE 3  (35-60) the lost art of sampling
    log("  score: scene 3")
    sfx.put(S.fade(S.vinyl_crackle(6.5), 0.5, 1.0), 35.0, 0.4)
    sfx.put(S.tape_hiss(17.5), 35.0, 1.0)
    music.put(S.pad(mtof([50, 57, 60, 64, 69]), 6.6, cutoff=1200, a=1.5, r=1.5), 35.0, 0.35)
    bpm3 = 105
    bar3 = 60 / bpm3 * 4
    L0 = 41.0
    b = Bus(12.0)
    drums(b, ev, kits["raw"], 0.0, bar3 * 5, bpm3, FUNK, swing=0.58, human=0.005, tag="s3", off=L0)
    bassline(b, bar3, bar3 * 5, bpm3, ACID, kind="acid", gain=0.55)
    bassline(b, bar3, bar3 * 5, bpm3, ROCK, kind="bass", gain=0.45)
    gospel = Bus(bar3 * 3 + 1)
    for i, ch in enumerate(GOSPEL[:3]):
        gospel.put(S.choir(mtof(ch), bar3 + 0.2, "o" if i % 2 else "a"), i * bar3, 0.55)
    gx = S.reverb(gospel.x[:, 0], 1.6, 0.3, 5000, stereo_out=True)
    gx = apply_stereo(gx, lambda c: S.bp(c, 220, 7000))      # 'vinyl strip'
    b.put(gx, bar3 * 2, 0.9)
    sec = apply_stereo(b.x, lambda c: S.bitcrush(S.lp(c, 8500), 12, 1))
    cut = bar3 * 5
    sec[n(cut):] = 0
    music.put(sec, L0, 0.9)
    ev.add("layer", L0, name="60s FUNK DRUMS")
    ev.add("layer", L0 + bar3, name="70s ROCK BASSLINE")
    ev.add("layer", L0 + bar3 * 2, name="GOSPEL ACAPELLA")
    t_ai = L0 + cut
    ev.add("ai_cut", t_ai)
    b = Bus(60 - t_ai)
    drums(b, ev, kits["ai"], 0.0, 60 - t_ai, 120, AI, tag="s3-ai", off=t_ai)
    step = 60 / 120 / 4
    seqn = [69, 72, 76, 72, 69, 72, 76, 79]
    for i in range(int((60 - t_ai) / step)):
        if i % 2 == 0:
            b.put(S.pluck(mtof(seqn[(i // 2) % len(seqn)])), i * step, 0.5, 0.0)
    music.put(S.fade(b.x[:, 0], 0, 0.6), t_ai, 0.8)

    # ---------------- SCENE 4  (60-95) the illusion of "just playing records"
    log("  score: scene 4")
    anthem = Bus(36.0)
    A0 = 0.0
    drums(anthem, ev, kits["edm"], 0, 35.0, 128, {"k": "x...x...x...x..."}, tag="s4", off=60.0)
    drums(anthem, ev, kits["raw"], 0, 35.0, 128, BIGBEAT, swing=0.55, human=0.003, tag="s4-break", gain=0.8, off=60.0)
    bassline(anthem, 0, 35.0, 128, ACID, kind="acid", gain=0.6)
    chords(anthem, 0, 128, [AM, F, Cc, G] * 5, 4, lambda f, d: S.pad(f, d, cutoff=2500, a=0.05, r=0.3), 0.35)
    beat = 60 / 128
    hk = hook_audio
    for i in range(int(35 / (beat * 8))):
        anthem.put(hk, i * beat * 8 + beat * 4, 0.55, 0.0)
    ax = anthem.x[: n(35.0)]
    # earplug: cutoff sweeps 300 Hz -> 1.1 kHz (60-73.5), build to 3k, gap, drop at 75.6 full range
    tt = np.arange(len(ax)) / SR + 60
    fc = np.where(tt < 73.5, 300 * (1100 / 300) ** ((tt - 60) / 13.5),
                  np.where(tt < 75.6, 1100 * (3000 / 1100) ** ((tt - 73.5) / 2.1), 20000))
    # 'mastery' filter sweep at 88-94.5 (hand on the filter knob)
    sw = (tt >= 88) & (tt < 94.5)
    ph = (tt[sw] - 88) / 6.5
    fc[sw] = 20000 * (500 / 20000) ** np.sin(np.pi * ph)
    ax = apply_stereo(ax, lambda c: S.svf_sweep(c, fc, q=0.9, mode="lp", block=128))
    gain = np.where(tt < 75.6, 1.25, 1.0)
    ax = ax * gain[:, None]
    ax[(tt >= 75.35) & (tt < 75.6)] = 0
    music.put(ax, 60.0, 0.9)
    snr = Bus(2.0)
    t = 0.0
    while t < 1.85:
        snr.put(kits["edm"]["s"], t, 0.3 + 0.4 * t / 1.85)
        t += beat / 4
    music.put(snr.x, 73.5, 0.8)
    sfx.put(S.riser(1.85, 300, 6000), 73.5, 0.35)
    sfx.put(S.impact(3.0), 75.6, 0.9)
    ev.add("drop", 75.6)
    distant = S.crowd(15.6, 0.6)
    sfx.put(apply_stereo(distant, lambda c: S.lp(c, 900)), 60.0, 0.35)
    sfx.put(S.fade(S.crowd(19.4, 1.4), 0.05, 2.0), 75.6, 0.55)
    ev.add("filter_sweep", 88.0, end=94.5)
    for t in (60.0, 75.6, 88.0):
        ev.add("cut", t)

    # ---------------- SCENE 5  (95-135) brighton beach -> billion-dollar mainstages
    log("  score: scene 5")
    b = Bus(41.0)
    drums(b, ev, kits["raw"], 0, 15.0, 128, BIGBEAT, swing=0.55, human=0.003, tag="s5-beach", off=95.0)
    drums(b, ev, kits["edm"], 0, 15.0, 128, {"k": "x...x...x...x..."}, tag="s5-beach", gain=0.8, off=95.0)
    bassline(b, 0, 15.0, 128, ROCK, kind="bass", gain=0.6)
    chords(b, 0, 128, [AM, F, Cc, G] * 2, 4, lambda f, d: S.organ_stab(f, d) * 0.8, 0.3)
    beach = S.reverb(b.x[:, 0] * 0.5 + b.x[:, 1] * 0.5, 2.5, 0.35, 5000, stereo_out=True)
    tt = np.arange(len(beach)) / SR
    beach *= np.clip((16.0 - tt) / 3.0, 0, 1)[:, None]
    music.put(beach, 95.0, 0.85)
    sfx.put(S.fade(S.crowd(17.0, 1.6), 0.6, 4.0), 95.0, 0.6)
    ev.add("cut", 95.0)
    # corporate machine
    c = Bus(25.0)
    drums(c, ev, kits["ai"], 0, 24.5, 128, {"k": "x...x...x...x...", "h": "..x...x...x...x."}, tag="s5-corp", off=110.0)
    bassline(c, 0, 24.5, 128, [(0, 2, 33), (4, 2, 33), (8, 2, 33), (12, 2, 33)], kind="bass", gain=0.55)
    cx = S.fade(c.x[:, 0], 2.5, 0.5)
    music.put(cx, 110.0, 0.8)
    sfx.put(S.fade(S.ticker(26.0), 2.0, 1.5), 108.0, 0.4)
    for t in np.arange(110.0, 133.0, 60 / 128 * 8):
        sfx.put(S.cash_register(1.0), t + 60 / 128 * 2, 0.5, rng.uniform(-0.5, 0.5))
        ev.add("register", t + 60 / 128 * 2)
    for t in (112.0, 120.0, 127.5):
        sfx.put(S.fade(S.rumble(4.0), 1.5, 2.0), t, 0.45)
        ev.add("rumble", t)
    sfx.put(S.riser(1.5, 200, 3500), 133.5, 0.3)

    # ---------------- SCENE 6  (135-225) the playbook: a live 90s mini-mix
    log("  score: scene 6")
    hookv = S.hp(hook_audio, 160)
    hook_tel = S.bp(hookv, 450, 3200) * 1.6
    hook_full = S.soft(hookv * 1.3, 1.4)

    def echo(x, dt, fb=0.35, taps=3):
        y = np.zeros(len(x) + n(dt * taps))
        for k in range(taps + 1):
            y[n(dt * k): n(dt * k) + len(x)] += x * fb ** k
        return y

    def scratch(x, dur=0.5):
        tt = S.t_axis(dur)
        pos = (len(x) * 0.15 + np.sin(2 * np.pi * 5.5 * tt) * SR * 0.06).astype(int)
        y = x[np.clip(pos, 0, len(x) - 1)] * (np.sin(2 * np.pi * 11 * tt) > -0.2)
        return S.bp(y, 300, 6000)

    for (a, z, bpm, genre), kit, pat, hk in zip(C.HOOK_SECTIONS, ["boom", "edm", "dnb"], [BOOM, HOUSE, DNB],
                                               [hook_tel, hook_full, hook_full]):
        bb = Bus(z - a + 1)
        drums(bb, ev, kits[kit], 0, z - a, bpm, pat, swing=0.58 if genre == "HIP-HOP" else 0.5,
              human=0.004 if genre == "HIP-HOP" else 0, tag=f"s6-1-{genre}", off=a)
        bar = 60 / bpm * 4
        if genre == "HIP-HOP":
            bassline(bb, 0, z - a, bpm, [(0, 6, 33), (10, 4, 36)], kind="bass", gain=0.6)
            chords(bb, 0, bpm, [[57, 60, 64, 67], [53, 57, 60, 64]] * 2, 4, lambda f, d: S.pad(f, d, cutoff=1400, a=0.02, r=0.4), 0.3)
        elif genre == "HOUSE":
            bassline(bb, 0, z - a, bpm, [(2, 1, 45), (6, 1, 45), (10, 1, 45), (14, 1, 48)], kind="bass", gain=0.6)
            for i in range(int((z - a) / (bar / 2))):
                bb.put(S.organ_stab(mtof([69, 72, 76]), 0.25), i * bar / 2 + 60 / bpm * 0.5, 0.35, 0.3)
        else:
            bassline(bb, 0, z - a, bpm, REESE, kind="bass", gain=0.7)
        nb = int(np.floor((z - a) / bar + 1e-6))
        for i in range(nb):
            bb.put(echo(hk, 60 / bpm * 0.75), i * bar, 0.8, 0.0)
            ev.add("hook", a + i * bar, genre=genre, bpm=bpm)
        music.put(S.fade(bb.x[: n(z - a), 0] * 0.5 + bb.x[: n(z - a), 1] * 0.5, 0.0, 0.03), a, 0.9)
        if a > 135.0:
            sfx.put(scratch(hook_full, 0.45), a - 0.4, 0.5)
            ev.add("scratch", a - 0.4)
        ev.add("section", a, genre=genre, bpm=bpm)

    bpm6 = 130
    beat6 = 60 / bpm6
    bar6 = beat6 * 4

    # step 2: acapella over funky break (157.5-180)
    b = Bus(23.0)
    drums(b, ev, kits["raw"], 0, 22.5, bpm6, FUNK, swing=0.57, human=0.004, tag="s6-2", off=157.5)
    bassline(b, 0, 22.5, bpm6, FUNKBASS, kind="bass", gain=0.7)
    ac = Bus(20.0)
    prog = [GOSPEL[0], GOSPEL[1], GOSPEL[2], GOSPEL[3]] * 3
    for i, ch in enumerate(prog[:10]):
        ac.put(S.choir(mtof(ch), bar6 + 0.25, ["a", "o", "e", "o"][i % 4]), i * bar6, 0.7)
    acx = S.reverb(ac.x[:, 0], 2.0, 0.3, 6000, stereo_out=True)
    b.put(acx, bar6 * 2, 0.75)
    music.put(b.x[: n(22.5)], 157.5, 0.9)
    ev.add("acapella", 157.5 + bar6 * 2)

    # step 3: chaotic mini-drops (180-202.5)
    drops = [180.0 + k * beat6 for k in C.MINI_DROP_BEATS]
    in_drop = lambda t, base=0.0: any(d <= t + base < d + 2 * beat6 - 1e-3 for d in drops)
    b = Bus(23.0)
    drums(b, ev, kits["edm"], 0, 22.5, bpm6, {"k": "x...x...x...x..."}, tag="s6-3", mute=lambda t: in_drop(t, 180.0), off=180.0)
    drums(b, ev, kits["raw"], 0, 22.5, bpm6, BIGBEAT, swing=0.55, tag="s6-3b", gain=0.75, off=180.0)
    bassline(b, 0, 22.5, bpm6, ACID, kind="acid", gain=0.6, mute=lambda t: in_drop(t, 180.0))
    for i in range(int(22.5 / bar6)):
        if i % 2 == 1:
            b.put(hook_full, i * bar6, 0.6)
    bx = b.x[: n(22.5)]
    tt = np.arange(len(bx)) / SR + 180.0
    mask = np.array([in_drop(t) for t in tt[::480]])
    mask = np.repeat(mask, 480)[: len(bx)]
    hp_x = apply_stereo(bx, lambda c: S.hp(c, 500, 4))
    bx = np.where(mask[:, None], hp_x * 0.9, bx)
    music.put(bx, 180.0, 0.9)
    for d in drops:
        ev.add("minidrop", d, end=d + 2 * beat6)
        sfx.put(S.impact(1.2) * 0.6, d + 2 * beat6, 0.6)

    # step 4: cinematic crescendo (202.5-225)
    b = Bus(23.0)
    drums(b, ev, kits["edm"], 0, 22.5, bpm6, {"k": "x...x...x...x...", "c": "....x.......x..."}, tag="s6-4", off=202.5)
    drums(b, ev, kits["raw"], 0, 22.5, bpm6, FUNK, swing=0.56, tag="s6-4b", gain=0.7, off=202.5)
    drums(b, ev, kits["dnb"], bar6 * 4, 22.5, bpm6, {"t": "..x..x....x..x.x", "o": "..x...x...x...x."}, tag="s6-4c", gain=0.6, off=202.5)
    bassline(b, 0, 22.5, bpm6, ACID, kind="acid", gain=0.55)
    chords(b, 0, bpm6, [AM, F, Cc, G] * 3, 4, lambda f, d: S.supersaw(np.append(f, f[0] * 2), d, cutoff=8000) * S.adsr(d, 0.02, 0.2, 0.85, 0.1), 0.45, pump=0.5)
    for i in range(int(22.5 / bar6)):
        b.put(hook_full, i * bar6, 0.55)
    for i, ch in enumerate(GOSPEL * 3):
        if i * bar6 + bar6 <= 22.5:
            b.put(S.choir(mtof(ch), bar6 + 0.2, "a"), i * bar6, 0.45, 0.0)
    b.put(S.riser(6.0, 200, 7000), 13.0, 0.45)
    bx = b.x[: n(22.5)]
    tt = np.arange(len(bx)) / SR
    bx = bx * (0.75 + 0.35 * np.clip(tt / 19.0, 0, 1))[:, None]
    music.put(bx, 202.5, 0.85)
    sfx.put(S.impact(4.0), 221.8, 0.9)
    ev.add("peak", 221.8)
    sfx.put(S.crowd(6.0, 1.6), 219.5, 0.45)
    for s in C.STEPS:
        ev.add("step", s["start"], n=s["n"])

    # ---------------- SCENE 7  (225-270) resolution
    log("  score: scene 7")
    final = S.choir(mtof([45, 57, 60, 64, 67, 71]), 14.0, "a")
    final = S.reverb(final, 4.0, 0.5, 4000, stereo_out=True)
    music.put(S.fade(final[:, 0], 0.05, 6.0), 221.8, 0.5, -0.4)
    music.put(S.fade(final[:, 1], 0.05, 6.0), 221.8, 0.5, 0.4)
    b = Bus(40.0)
    drums(b, ev, kits["boom"], 0, 34.0, 86, {"k": "x.......x.x.....", "s": "....x.......x...", "h": "x.x.x.x.x.x.x.x."},
          swing=0.6, human=0.004, tag="s7", off=225.0)
    chords(b, 0, 86, [[57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 65], [57, 60, 64, 67]] * 3, 4,
           lambda f, d: S.pad(f, d, cutoff=1100, a=0.4, r=0.8), 0.35)
    tape = apply_stereo(b.x[: n(37.0)], lambda c: S.bitcrush(S.lp(c, 3500), 10, 2))
    tt = np.arange(len(tape)) / SR
    tape *= (np.clip(tt / 2.0, 0, 1) * np.clip((37.0 - tt) / 10.0, 0, 1))[:, None]
    tape = apply_stereo(tape, lambda c: S.reverb(c, 2.5, 0.35, 3000))
    music.put(tape, 225.0, 0.75)
    sfx.put(S.fade(S.tape_hiss(45.0) * 2.2, 3.0, 2.5), 225.0, 1.0)
    sfx.put(S.fade(S.vinyl_crackle(14.0), 3.0, 3.0), 255.0, 0.35)
    sfx.put(S.impact(5.0), 257.5, 0.65)
    ev.add("title", 257.5)
    for t in (225.0,):
        ev.add("cut", t)

    # hard stop at duration, gentle tail
    N = n(DUR)
    for bus in (music, sfx):
        bus.x = bus.x[:N]
        bus.x[-n(0.8):] *= np.linspace(1, 0, n(0.8))[:, None]
    return music.x, sfx.x, ev.e
