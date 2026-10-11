"""Neural voiceover via Kokoro (ONNX). Cached, resampled to 48 kHz, fit-to-window."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config as C

MODEL_DIR = Path(os.environ.get("REMIX_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))

_kokoro = None


def _engine():
    global _kokoro
    if _kokoro is None:
        onnx, voices = MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin"
        if not (onnx.exists() and voices.exists()):
            raise SystemExit(f"TTS model missing in {MODEL_DIR}. Run scripts/fetch_models.sh")
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(str(onnx), str(voices))
    return _kokoro


def _trim(x, thresh=0.004):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    a, b = max(0, idx[0] - 480), min(len(x), idx[-1] + 2400)
    return x[a:b]


def speak(text, cache_dir, speed=None, voice=None, lang=None):
    """Return mono float32 48 kHz audio for text (cached on disk)."""
    voice = voice or C.VOICE["voice"]
    lang = lang or C.VOICE["lang"]
    speed = speed or C.VOICE["speed"]
    key = hashlib.sha1(json.dumps([text, voice, lang, round(speed, 3), "v2"]).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"tts_{key}.wav"
    if path.exists():
        x, _ = sf.read(path, dtype="float32")
        return x
    samples, sr = _engine().create(text, voice=voice, speed=speed, lang=lang)
    x = resample_poly(np.asarray(samples, dtype=np.float64), C.SR, sr).astype(np.float32)
    x = _trim(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, C.SR, subtype="FLOAT")
    return x


def build_vo(cache_dir, log=print):
    """Render every cue, speeding up (within limits) any that overrun their window.

    Returns list of dicts {cue, audio, start, end, speed}.
    """
    out = []
    for cue in C.VO:
        window = cue.hard_end - cue.start
        speed = C.VOICE["speed"]
        x = speak(cue.text, cache_dir, speed)
        while len(x) / C.SR > window and speed < C.VOICE["max_speed"]:
            speed = min(C.VOICE["max_speed"], speed * (len(x) / C.SR) / window * 1.01)
            x = speak(cue.text, cache_dir, speed)
        dur = len(x) / C.SR
        log(f"  VO {cue.id}: {dur:5.2f}s in {window:5.2f}s window @ speed {speed:.3f}")
        out.append(dict(cue=cue, audio=x, start=cue.start, end=cue.start + dur, speed=speed))
    return out
