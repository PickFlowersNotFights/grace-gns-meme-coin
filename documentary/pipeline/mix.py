"""Dialogue-first mix, mastering (BS.1770 loudness + true-peak limiting), 5.1 bed, visual features."""
import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter, resample_poly, butter, sosfilt

from . import config as C
from . import synth as S
from .synth import SR, n

STEREO_LUFS, STEREO_TP = -14.0, -1.0       # streaming / social spec
SURROUND_LUFS, SURROUND_TP = -24.0, -2.0   # broadcast-style 5.1 bed
VO_MARGIN_DB = 13.0                       # dialogue-over-bed target during VO lines


# ------------------------------------------------------------------ measurement
def _k_weight(x):
    # ITU-R BS.1770-4 K-weighting coefficients for 48 kHz
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    return lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)


def lufs(x, weights=None):
    """Integrated loudness (BS.1770-4, gated). x: (N, ch)."""
    x = np.atleast_2d(x.T).T
    weights = np.ones(x.shape[1]) if weights is None else np.asarray(weights)
    y = _k_weight(x)
    blk, hop = n(0.4), n(0.1)
    nb = (len(y) - blk) // hop + 1
    if nb < 1:
        return -70.0
    cs = np.cumsum(np.vstack([np.zeros((1, y.shape[1])), y ** 2]), axis=0)
    idx = np.arange(nb) * hop
    ms = (cs[idx + blk] - cs[idx]) / blk
    z = ms @ weights
    l = -0.691 + 10 * np.log10(z + 1e-12)
    g = z[l > -70]
    if len(g) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g2 = z[(l > -70) & (l > rel)]
    return float(-0.691 + 10 * np.log10(np.mean(g2) + 1e-12))


def true_peak_db(x):
    x = np.atleast_2d(x.T).T
    up = resample_poly(x, 4, 1, axis=0)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-12))


def limit(x, ceiling_db, lookahead=0.004, release=0.06):
    """Look-ahead true-peak-aware brickwall limiter (linked stereo)."""
    ceil = 10 ** (ceiling_db / 20) * 0.985
    up = resample_poly(x, 4, 1, axis=0)
    pk = np.max(np.abs(up), axis=1).reshape(-1, 4).max(axis=1)[: len(x)]
    pk = np.pad(pk, (0, len(x) - len(pk)), mode="edge")
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = n(lookahead)
    g = minimum_filter1d(g, size=2 * la + 1, origin=0)
    g = uniform_filter1d(g, size=la)                        # smooth attack
    r = np.exp(-1 / (release * SR))                          # release: one-pole on gain-reduction depth
    gr = lfilter([1 - r], [1, -r], 1 - g)
    g = np.minimum(g, 1 - gr * 0.0 + 0) if False else g      # attack already handled
    g = 1 - np.maximum(1 - g, gr)
    return x * g[:, None]


def compress(x, thresh_db=-18, ratio=2.5, attack=0.01, release=0.18, makeup_db=0.0):
    mono = np.sqrt(np.mean(x ** 2, axis=1))
    a = np.exp(-1 / (attack * SR))
    r = np.exp(-1 / (release * SR))
    env = lfilter([1 - r], [1, -r], mono ** 2)
    env = np.maximum(env, lfilter([1 - a], [1, -a], mono ** 2))
    lvl = 10 * np.log10(env + 1e-12)
    over = np.maximum(0, lvl - thresh_db)
    gr = over * (1 - 1 / ratio)
    return x * (10 ** ((makeup_db - gr) / 20))[:, None]


def master(x, target_lufs, tp_db, weights=None, iters=3):
    for _ in range(iters):
        g = target_lufs - lufs(x, weights)
        x = x * 10 ** (g / 20)
        x = limit(x, tp_db)
    return x


# ------------------------------------------------------------------ VO bus
def vo_bus(vo_items):
    N = n(C.DURATION)
    vo = np.zeros(N)
    act = np.zeros(N)
    for it in vo_items:
        x = S.hp(it["audio"].astype(np.float64), 85, 2)
        presence = S.bp(x, 2500, 5000)
        x = x + presence * 0.25
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.1     # equal level per line
        o = n(it["start"])
        L = min(len(x), N - o)
        vo[o:o + L] += x[:L]
        act[max(0, o - n(0.25)): min(N, o + L + n(0.35))] = 1.0
    # light dialogue compression
    vo = compress(vo[:, None], -26, 3.0, 0.004, 0.12)[:, 0]
    ramp = n(0.18)
    act = uniform_filter1d(act, ramp * 2)
    return vo, act


def mixdown(music, sfx, vo_items, out_dir, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    N = n(C.DURATION)
    music, sfx = music[:N], sfx[:N]
    vo, act = vo_bus(vo_items)

    # level-aware ducking: per VO line, duck the bed just enough for a fixed dialogue margin
    bed0 = compress(music + sfx, -16, 2.0, 0.012, 0.2)
    rms = lambda x: np.sqrt(np.mean(x ** 2) + 1e-12)
    vo_lvl = rms(vo[act > 0.5])
    g = vo_lvl * 10 ** (-1.0 / 20) / rms(bed0[np.abs(bed0).max(axis=1) > 1e-4])   # unducked bed ≈ VO -1 dB
    bed0 = bed0 * g
    duck_db = np.zeros(N)
    for it in vo_items:
        a, b = n(it["start"]), min(N, n(it["end"]))
        margin = 20 * np.log10(rms(vo[a:b]) / rms(bed0[a:b]))
        depth = float(np.clip(VO_MARGIN_DB - margin, 6.0, 30.0))
        w = np.zeros(N)
        w[max(0, a - n(0.25)): min(N, b + n(0.35))] = 1.0
        w = uniform_filter1d(w, n(0.36))
        duck_db = np.maximum(duck_db, w * depth)
        log(f"    duck {it['cue'].id}: {depth:4.1f} dB")
    bed = bed0 * (10 ** (-duck_db / 20))[:, None]
    vo_st = np.stack([vo, vo], axis=1)
    mix = bed + vo_st

    stereo = master(mix, STEREO_LUFS, STEREO_TP)
    m_l, m_tp = lufs(stereo), true_peak_db(stereo)
    log(f"  stereo master: {m_l:.2f} LUFS, {m_tp:.2f} dBTP")

    # 5.1 bed  (L R C LFE Ls Rs). Enhanced LFE for festival simulation.
    bed_m = bed.mean(axis=1)
    lfe = S.lp(bed_m, 110, 4) * 1.9 + S.lp(bed_m, 50, 2) * 0.6
    front = S.hp(bed[:, 0], 60), S.hp(bed[:, 1], 60)
    amb = S.reverb(S.hp(bed_m, 180), 2.2, 1.0, 5000, stereo_out=True)
    sfx_s = sfx * g * (10 ** (-duck_db / 20))[:, None]
    surr_l = amb[:, 0] * 0.35 + np.roll(sfx_s[:, 0], n(0.012)) * 0.5
    surr_r = amb[:, 1] * 0.35 + np.roll(sfx_s[:, 1], n(0.017)) * 0.5
    six = np.stack([front[0] * 0.55, front[1] * 0.55, vo * 1.25 + bed_m * 0.08, lfe * 0.5, surr_l * 0.55, surr_r * 0.55], axis=1)
    w = np.array([1, 1, 1, 0, 1.41, 1.41])
    for _ in range(3):
        g = SURROUND_LUFS - lufs(six, w)
        six = six * 10 ** (g / 20)
        six = limit(six, SURROUND_TP)
    s_l, s_tp = lufs(six, w), true_peak_db(six)
    log(f"  5.1 master:    {s_l:.2f} LUFS, {s_tp:.2f} dBTP")

    files = {
        "stereo": out_dir / "REMIX_mix_stereo_48k24.wav",
        "surround": out_dir / "REMIX_mix_5.1_48k24.wav",
        "stem_music": out_dir / "stem_music.wav",
        "stem_sfx": out_dir / "stem_sfx.wav",
        "stem_vo": out_dir / "stem_vo.wav",
    }
    sf.write(files["stereo"], stereo.astype(np.float32), SR, subtype="PCM_24")
    sf.write(files["surround"], six.astype(np.float32), SR, subtype="PCM_24")
    pk = max(np.max(np.abs(music)), np.max(np.abs(sfx)), 1e-9)
    sf.write(files["stem_music"], (music / pk * 0.89).astype(np.float32), SR, subtype="PCM_24")
    sf.write(files["stem_sfx"], (sfx / pk * 0.89).astype(np.float32), SR, subtype="PCM_24")
    sf.write(files["stem_vo"], (vo / max(np.max(np.abs(vo)), 1e-9) * 0.89).astype(np.float32), SR, subtype="PCM_24")
    stats = dict(stereo_lufs=m_l, stereo_tp=m_tp, surround_lufs=s_l, surround_tp=s_tp)
    return files, stereo, bed, vo, stats


# ------------------------------------------------------------------ features for audio-reactive visuals
def features(bed, vo, fps=C.FPS):
    N = n(C.DURATION)
    hop = SR // fps
    nf = int(round(C.DURATION * fps))
    mono = bed[:N].mean(axis=1)
    low = sosfilt(butter(4, 150, "low", fs=SR, output="sos"), mono)
    high = sosfilt(butter(4, 4000, "high", fs=SR, output="sos"), mono)

    def frame_rms(x):
        x = np.pad(x, (0, nf * hop - len(x) + hop))
        f = x[: nf * hop].reshape(nf, hop)
        return np.sqrt(np.mean(f ** 2, axis=1))

    out = {}
    for k, x in (("rms", mono), ("low", low), ("high", high), ("vo", vo[:N])):
        r = frame_rms(x)
        r = r / (np.percentile(r, 98) + 1e-9)
        out[k] = np.clip(r, 0, 1.5).round(4).tolist()
    # 100 Hz envelope for waveform drawing
    h2 = SR // 100
    m = np.pad(mono, (0, int(C.DURATION * 100) * h2 + h2 - len(mono)))[: int(C.DURATION * 100) * h2]
    e = np.sqrt(np.mean(m.reshape(-1, h2) ** 2, axis=1))
    out["env"] = np.clip(e / (np.percentile(e, 98) + 1e-9), 0, 1.2).round(3).tolist()
    return out
