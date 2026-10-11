"""DSP primitives: oscillators, drums, filters, FX. All mono float64 @ 48 kHz unless noted."""
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve, lfilter

from .config import SR

RNG = np.random.default_rng(1998)


def t_axis(dur):
    return np.arange(int(round(dur * SR))) / SR


def n(dur):
    return int(round(dur * SR))


# ------------------------------------------------------------------ filters
def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, max(fc, 5), "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos"), x)


def lp0(x, fc, order=2):
    """Zero-phase low-pass (for crossovers)."""
    return sosfiltfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def svf_sweep(x, cutoff, q=0.7, mode="lp", block=64):
    """Time-varying resonant biquad (RBJ). cutoff: array (per-sample) or scalar. q up to ~12."""
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), x.shape)
    y = np.empty_like(x)
    zi = np.zeros(2)
    for i in range(0, len(x), block):
        fc = float(np.clip(cutoff[i], 20, SR * 0.45))
        w0 = 2 * np.pi * fc / SR
        alpha = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        if mode == "lp":
            b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        elif mode == "hp":
            b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        else:  # band-pass, constant peak
            b = np.array([alpha, 0, -alpha])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
        seg, zi = lfilter(b / a[0], a / a[0], x[i:i + block], zi=zi)
        y[i:i + block] = seg
    return y


# ------------------------------------------------------------------ oscillators
def blsaw(freq, dur, phase=0.0, max_h=None):
    """Band-limited sawtooth by additive synthesis (freq scalar or per-sample array)."""
    t = t_axis(dur)
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), t.shape)
    ph = 2 * np.pi * np.cumsum(f) / SR + phase
    fmax = float(np.max(f))
    H = int(min(max_h or 200, (SR * 0.45) // max(fmax, 1)))
    out = np.zeros_like(t)
    for k in range(1, H + 1):
        out += np.sin(k * ph) / k
    return out * (2 / np.pi)


def blsquare(freq, dur, max_h=None):
    t = t_axis(dur)
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), t.shape)
    ph = 2 * np.pi * np.cumsum(f) / SR
    H = int(min(max_h or 150, (SR * 0.45) // max(float(np.max(f)), 1)))
    out = np.zeros_like(t)
    for k in range(1, H + 1, 2):
        out += np.sin(k * ph) / k
    return out * (4 / np.pi)


def sine(freq, dur, phase=0.0):
    t = t_axis(dur)
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), t.shape)
    return np.sin(2 * np.pi * np.cumsum(f) / SR + phase)


def noise(dur, rng=RNG):
    return rng.standard_normal(n(dur))


def pink(dur, rng=RNG):
    w = rng.standard_normal(n(dur))
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    y = lfilter(b, a, w)
    return y / (np.std(y) + 1e-9)


def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


# ------------------------------------------------------------------ envelopes / dynamics
def adsr(dur, a=0.005, d=0.1, s=0.7, r=0.1):
    N = n(dur)
    env = np.full(N, s, dtype=np.float64)
    na, nd, nr = n(a), n(d), n(r)
    na = min(na, N)
    env[:na] = np.linspace(0, 1, na, endpoint=False)
    nd = min(nd, N - na)
    env[na:na + nd] = np.linspace(1, s, nd, endpoint=False)
    nr = min(nr, N)
    if nr:
        env[N - nr:] *= np.linspace(1, 0, nr)
    return env


def expdecay(dur, tau):
    return np.exp(-t_axis(dur) / tau)


def soft(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def bitcrush(x, bits=10, down=2):
    q = 2 ** (bits - 1)
    y = np.round(x * q) / q
    if down > 1:
        y = np.repeat(y[::down], down)[: len(x)]
    return y


def fade(x, fin=0.0, fout=0.0):
    x = x.copy()
    a, b = n(fin), n(fout)
    shape = (-1,) + (1,) * (x.ndim - 1)
    if a:
        x[:a] *= (np.linspace(0, 1, a) ** 2).reshape(shape)
    if b:
        x[-b:] *= (np.linspace(1, 0, b) ** 2).reshape(shape)
    return x


def env_follow(x, attack=0.005, release=0.15):
    """Fast vectorised envelope (abs → one-pole smoothing with asymmetric times approximated)."""
    a = np.abs(x)
    ar = np.exp(-1 / (release * SR))
    y = lfilter([1 - ar], [1, -ar], a)
    aa = np.exp(-1 / (attack * SR))
    # attack smoothing on rises: take max of slow and fast-follow
    yf = lfilter([1 - aa], [1, -aa], a)
    return np.maximum(y, yf * 0.9)


# ------------------------------------------------------------------ reverb
_IR_CACHE = {}


def reverb_ir(seconds=1.8, damp=6000, seed=7, stereo=False):
    key = (seconds, damp, seed, stereo)
    if key not in _IR_CACHE:
        rng = np.random.default_rng(seed)
        chans = []
        for c in range(2 if stereo else 1):
            t = t_axis(seconds)
            ir = rng.standard_normal(len(t)) * np.exp(-t * 6.9 / seconds)
            ir = lp(ir, damp)
            ir[: n(0.012)] *= np.linspace(0, 1, n(0.012))
            chans.append(ir / np.sqrt(np.sum(ir ** 2)))
        _IR_CACHE[key] = chans
    return _IR_CACHE[key]


def reverb(x, seconds=1.8, wet=0.25, damp=6000, stereo_out=False):
    irs = reverb_ir(seconds, damp, stereo=stereo_out)
    if stereo_out:
        wets = [fftconvolve(x, ir)[: len(x)] for ir in irs]
        return np.stack([x * (1 - wet) + w * wet for w in wets], axis=1)
    return x * (1 - wet) + fftconvolve(x, irs[0])[: len(x)] * wet


# ------------------------------------------------------------------ drums
def kick(dur=0.45, f0=160, f1=46, punch=1.0, click=0.35, drive=1.6):
    t = t_axis(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.035)
    body = sine(f, dur) * expdecay(dur, 0.22)
    c = hp(noise(dur), 2500) * expdecay(dur, 0.004) * click
    return soft((body * punch + c), drive) * 0.95


def snare(dur=0.3, tone=185, snap=1.0, body=0.6):
    t = t_axis(dur)
    b = sine(tone + 40 * np.exp(-t / 0.01), dur) * expdecay(dur, 0.06) * body
    nz = bp(noise(dur), 1200, 9000) * expdecay(dur, 0.075) * snap
    return soft(b + nz, 1.3) * 0.8


def clap(dur=0.35):
    out = np.zeros(n(dur))
    nz = bp(noise(dur), 900, 6000)
    for i, off in enumerate([0, 0.011, 0.023, 0.034]):
        o = n(off)
        e = expdecay(dur - off, 0.012 if i < 3 else 0.09)
        out[o:] += nz[o:] * e[: len(out) - o]
    return out * 0.6


def hat(dur=0.06, open_=False):
    d = 0.35 if open_ else dur
    x = hp(noise(d), 7000, 4) * expdecay(d, 0.12 if open_ else 0.018)
    return x * (0.35 if open_ else 0.4)


def tom(f=110, dur=0.35):
    t = t_axis(dur)
    return sine(f * (1 + 0.6 * np.exp(-t / 0.03)), dur) * expdecay(dur, 0.12) * 0.8


def ride(dur=0.9):
    t = t_axis(dur)
    partials = [3110, 4570, 5230, 6890, 8120]
    x = sum(np.sin(2 * np.pi * p * t + i) for i, p in enumerate(partials))
    x = x / len(partials) + 0.5 * hp(noise(dur), 6000)
    return x * expdecay(dur, 0.35) * 0.18


# ------------------------------------------------------------------ instruments
def acid_note(freq, dur, accent=False, env_amt=None, q=8.0):
    """303-style: band-limited saw through swept resonant LP."""
    x = blsaw(freq, dur, max_h=60)
    t = t_axis(dur)
    env_amt = env_amt or (4200 if accent else 2400)
    cutoff = 280 + env_amt * np.exp(-t / (0.09 if accent else 0.14))
    y = svf_sweep(x, cutoff, q=q, mode="lp")
    amp = adsr(dur, 0.003, 0.05, 0.75, 0.02)
    return soft(y * amp * (1.4 if accent else 1.0), 1.8) * 0.5


def bass_note(freq, dur, grit=2.2, cutoff=900):
    x = 0.65 * blsaw(freq, dur, max_h=40) + 0.45 * sine(freq / 2, dur)
    x = lp(x, cutoff)
    return soft(x * adsr(dur, 0.004, 0.08, 0.85, 0.03), grit) * 0.55


def supersaw(freqs, dur, detune=0.18, voices=7, cutoff=7000, rng=RNG):
    out = np.zeros(n(dur))
    for f in np.atleast_1d(freqs):
        for v in range(voices):
            d = (v - (voices - 1) / 2) / ((voices - 1) / 2) * detune
            out += blsaw(f * 2 ** (d / 12), dur, phase=rng.uniform(0, 6.28), max_h=50)
    out = lp(out / (voices * len(np.atleast_1d(freqs))), cutoff)
    return out


def pad(freqs, dur, cutoff=1800, a=0.6, r=1.0):
    out = np.zeros(n(dur))
    for f in np.atleast_1d(freqs):
        for d in (-0.08, 0.0, 0.07):
            out += blsaw(f * 2 ** (d / 12), dur, phase=RNG.uniform(0, 6.28), max_h=30)
    out = lp(out / (3 * len(np.atleast_1d(freqs))), cutoff)
    return out * adsr(dur, a, 0.2, 0.9, r)


def organ_stab(freqs, dur=0.35):
    out = np.zeros(n(dur))
    for f in np.atleast_1d(freqs):
        for k, g in [(1, 1), (2, 0.5), (3, 0.35), (4, 0.2), (6, 0.12)]:
            out += np.sin(2 * np.pi * f * k * t_axis(dur)) * g
    out = soft(out / len(np.atleast_1d(freqs)) * 0.6, 1.6)
    return out * adsr(dur, 0.004, 0.08, 0.6, 0.08) * 0.5


def pluck(freq, dur=0.25):
    """Sterile 'AI sample pack' pluck: pure sine + quick envelope."""
    return (sine(freq, dur) + 0.3 * sine(freq * 2, dur)) * expdecay(dur, 0.06) * 0.4


VOWELS = {  # formant (freq, bw, gain)
    "a": [(800, 80, 1.0), (1150, 90, 0.5), (2900, 120, 0.25), (3900, 130, 0.1)],
    "o": [(450, 70, 1.0), (800, 80, 0.45), (2830, 100, 0.15), (3500, 130, 0.08)],
    "u": [(325, 50, 1.0), (700, 60, 0.3), (2530, 170, 0.1), (3500, 180, 0.05)],
    "e": [(400, 60, 1.0), (1700, 80, 0.4), (2600, 120, 0.3), (3200, 120, 0.15)],
}


def voice_note(freq, dur, vowel="a", vib=5.3, vib_depth=0.012, breath=0.06, rng=RNG):
    """Formant-synthesis sung vowel (for the gospel/soul acapella stand-in)."""
    t = t_axis(dur)
    onset = np.minimum(1, t / 0.35)
    f = freq * (1 + vib_depth * onset * np.sin(2 * np.pi * vib * t + rng.uniform(0, 6))) \
        * (1 + 0.004 * rng.standard_normal())
    src = blsaw(f, dur, max_h=70) + breath * noise(dur)
    out = np.zeros_like(t)
    for fc, bw, g in VOWELS[vowel]:
        out += bp(src, max(fc - bw, 40), fc + bw, 2) * g
    return out * adsr(dur, 0.08, 0.2, 0.85, min(0.3, dur * 0.4))


def choir(freqs, dur, vowel="a", voices=3, rng=RNG):
    out = np.zeros(n(dur))
    for f in np.atleast_1d(freqs):
        for v in range(voices):
            out += voice_note(f * 2 ** (rng.uniform(-0.12, 0.12) / 12), dur, vowel, rng=rng)
    return out / (voices * len(np.atleast_1d(freqs))) * 2.2


# ------------------------------------------------------------------ sfx
def vinyl_crackle(dur, density=30, rng=RNG):
    N = n(dur)
    x = np.zeros(N)
    k = rng.poisson(density * dur)
    pos = rng.integers(0, N, k)
    x[pos] = rng.standard_normal(k) * rng.uniform(0.2, 1.0, k) ** 3
    pops = rng.integers(0, N, max(1, int(dur * 1.2)))
    x[pops] += rng.choice([-1, 1], len(pops)) * 1.2
    x = bp(x, 900, 12000)
    rumble = lp(noise(dur, rng), 40) * 0.6
    hiss = hp(pink(dur, rng), 3000) * 0.04
    return x * 0.6 + rumble * 0.25 + hiss


def tape_hiss(dur, rng=RNG):
    return hp(pink(dur, rng), 1800, 2) * 0.05


def tape_rewind(src, dur=1.0):
    """Reverse + accelerating speed (spooling) from a source buffer."""
    N = n(dur)
    t = np.linspace(0, 1, N)
    rate = 1 + 14 * t ** 1.6
    pos = np.cumsum(rate)
    pos = (len(src) - 1 - pos % (len(src) - 1)).astype(int)
    y = src[pos]
    y = bp(y, 300, 6000) * np.sin(np.pi * t) ** 0.5
    return y * 0.8


def needle_drop(dur=0.6):
    thump = sine(55 * (1 + np.exp(-t_axis(dur) / 0.02)), dur) * expdecay(dur, 0.06)
    scr = bp(noise(dur), 1500, 7000) * expdecay(dur, 0.02)
    return thump * 0.9 + scr * 0.4


def crowd(dur, intensity=1.0, rng=RNG):
    """Stereo crowd roar: band-limited pink bed + many filtered 'whoo' voices."""
    N = n(dur)
    out = np.zeros((N, 2))
    for c in range(2):
        bed = bp(pink(dur, rng), 250, 3500) * 0.35
        mod = lp(rng.standard_normal(N), 1.5)
        mod = 0.75 + 0.2 * mod / (np.std(mod) + 1e-9)
        out[:, c] = bed * np.clip(mod, 0.3, 1.4)
    nv = int(dur * 18 * intensity)
    for _ in range(nv):
        d = rng.uniform(0.4, 1.4)
        f0 = rng.uniform(380, 900)
        fv = f0 * (1 + 0.35 * np.sin(np.linspace(0, np.pi, n(d))))
        v = voice_note(fv, d, rng.choice(["o", "u", "a"]), vib=0, vib_depth=0, breath=0.3, rng=rng)
        v = v[:N] * np.sin(np.linspace(0, np.pi, min(len(v), N))) * rng.uniform(0.05, 0.18)
        p = rng.uniform(0, 1)
        o = rng.integers(0, max(1, N - len(v)))
        out[o:o + len(v), 0] += v * np.sqrt(1 - p)
        out[o:o + len(v), 1] += v * np.sqrt(p)
    return out * intensity


def cash_register(dur=1.0):
    t = t_axis(dur)
    bell = sum(np.sin(2 * np.pi * f * t) * g for f, g in [(2093, 1), (2637, 0.6), (3520, 0.45), (5274, 0.2)])
    bell *= expdecay(dur, 0.35)
    bell[: n(0.12)] *= 0.0
    drawer = bp(noise(dur), 300, 3000) * expdecay(dur, 0.05) * 0.5
    clunk = sine(90, dur) * expdecay(dur, 0.04)
    return bell * 0.35 + drawer + clunk * 0.5


def ticker(dur, rate=18, rng=RNG):
    x = np.zeros(n(dur))
    for k in range(int(dur * rate)):
        o = n(k / rate + rng.uniform(0, 0.01))
        c = bp(noise(0.01, rng), 2500, 7000) * expdecay(0.01, 0.002)
        x[o:o + len(c)] += c[: len(x) - o]
    return x * 0.5


def riser(dur, f0=200, f1=2400):
    t = t_axis(dur)
    sweep = svf_sweep(noise(dur), f0 * (f1 / f0) ** (t / dur), q=4, mode="bp")
    tone = sine(f0 * (f1 / f0) ** (t / dur) * 0.5, dur) * 0.15
    return (sweep * 0.6 + tone) * (t / dur) ** 2


def impact(dur=3.0):
    t = t_axis(dur)
    sub = sine(42 + 30 * np.exp(-t / 0.08), dur) * expdecay(dur, 0.7)
    nz = lp(noise(dur), 3000) * expdecay(dur, 0.25) * 0.5
    return soft(sub + nz, 1.5)


def rumble(dur, f=34):
    return sine(f, dur) * 0.5 + lp(noise(dur), 60) * 0.8
