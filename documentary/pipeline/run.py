"""REMIX production pipeline — one command from brief to verified deliverables.

    python -m pipeline.run --profile smoke     # full-length thumbnail-res run + QC (CI gate)
    python -m pipeline.run --profile preview   # 1080p review copy
    python -m pipeline.run --profile master    # 4K UHD 24p master + 9:16 social + stems + QC
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent


def log(*a):
    print(*a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def preflight():
    from .gfx import fonts_ok
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    need = ["libx264", "aac", "eac3", "pcm_s24le", "mov_text"]
    missing = [e for e in need if f" {e} " not in enc]
    if missing:
        raise SystemExit(f"ffmpeg missing encoders: {missing}")
    f = fonts_ok()
    if not all(f.values()):
        log(f"  warning: missing fonts (fallback used): {[k for k, v in f.items() if not v]}")
    from .tts import MODEL_DIR
    if not (MODEL_DIR / "kokoro-v1.0.onnx").exists():
        raise SystemExit(f"TTS model missing in {MODEL_DIR}; run scripts/fetch_models.sh")
    log("  preflight ok: ffmpeg encoders, fonts, TTS model")


# ------------------------------------------------------------------ audio stage (cached, resolution-independent)
def audio_stage(cache_root):
    from . import mix, score, tts
    key = hashlib.sha1(b"".join((SRC / f).read_bytes() for f in ["config.py", "synth.py", "score.py", "mix.py", "tts.py"])).hexdigest()[:12]
    adir = cache_root / f"audio_{key}"
    done = adir / "DONE"
    vo = tts.build_vo(cache_root / "tts", log=log)
    if not done.exists():
        log(f"  composing score + mixing -> {adir.name}")
        hook = tts.speak(C.HOOK_TEXT, cache_root / "tts", speed=0.95)
        music, sfx, events = score.compose(hook, log=log)
        files, stereo, bed, vo_stem, stats = mix.mixdown(music, sfx, vo, adir, log=log)
        feats = mix.features(bed, mix.vo_bus(vo)[0])
        (adir / "features.json").write_text(json.dumps(feats))
        (adir / "events.json").write_text(json.dumps(events))
        (adir / "stats.json").write_text(json.dumps(stats, indent=1))
        done.write_text(time.ctime())
    else:
        log(f"  audio cache hit: {adir.name}")
    timing = [dict(id=v["cue"].id, start=v["start"], end=v["end"], speed=v["speed"]) for v in vo]
    (adir / "vo_timing.json").write_text(json.dumps(timing, indent=1))
    return adir, vo


# ------------------------------------------------------------------ mux helpers
def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def mux_master(video, stereo, surround, srt, out):
    ff("-i", video, "-i", stereo, "-i", surround, "-i", srt,
       "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
       "-c:v", "copy", "-c:a:0", "aac", "-b:a:0", "320k", "-c:a:1", "eac3", "-b:a:1", "640k",
       "-c:s", "mov_text", "-t", f"{C.DURATION:.3f}",
       "-metadata", f"title={C.TITLE}: {C.SUBTITLE}",
       "-metadata:s:a:0", "title=Stereo (-14 LUFS)", "-metadata:s:a:0", "language=eng",
       "-metadata:s:a:1", "title=5.1 (-24 LUFS, enhanced LFE)", "-metadata:s:a:1", "language=eng",
       "-metadata:s:s:0", "language=eng", "-disposition:a:0", "default", "-disposition:a:1", "0",
       "-movflags", "+faststart", out)


def social_audio(stereo_path, scene, out):
    from . import mix
    from .synth import fade
    x, sr = sf.read(stereo_path, always_2d=True)
    seg = x[int(scene.start * sr): int(scene.end * sr)].copy()
    seg = fade(seg, 0.12, 0.5)
    seg = mix.master(seg, mix.STEREO_LUFS, mix.STEREO_TP)
    sf.write(out, seg.astype(np.float32), sr, subtype="PCM_24")
    return out


def contact_sheet(stills, out, cols=4):
    from PIL import Image, ImageDraw
    from .gfx import font
    ims = [Image.open(p) for p, _ in stills]
    w, h = 480, int(480 * ims[0].height / ims[0].width)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 28)), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    for i, (im, (_, lab)) in enumerate(zip(ims, stills)):
        x, y = (i % cols) * w, (i // cols) * (h + 28)
        sheet.paste(im.resize((w, h)), (x, y))
        d.text((x + 8, y + h + 14), lab, font=font("mono", 14), fill=(220, 220, 220), anchor="lm")
    sheet.save(out)
    return out


STILL_TIMES = [0.9, 2.8, 5.2, 8.0, 12.0, 26.0, 33.0, 38.0, 47.0, 56.0, 68.0, 74.6, 80.0, 91.5, 98.0, 106.0,
               114.0, 126.0, 140.0, 165.0, 186.5, 215.0, 240.0, 250.0, 262.0]


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(C.PROFILES), default="smoke")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--skip-social", action="store_true")
    args = ap.parse_args(argv)
    P = dict(C.PROFILES[args.profile])
    if args.workers:
        P["workers"] = args.workers
    out = args.out or ROOT / "build" / args.profile
    cache = ROOT / "build" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    from . import captions, qc, render

    log(f"[1/7] preflight ({args.profile}: {P['w']}x{P['h']} @ {C.FPS}p)")
    preflight()

    log("[2/7] voiceover + score + mix")
    adir, vo = audio_stage(cache)
    stats = json.loads((adir / "stats.json").read_text())
    feats, events = adir / "features.json", adir / "events.json"
    deliv = out / "deliverables"
    deliv.mkdir(exist_ok=True)
    stereo = deliv / "REMIX_mix_stereo_48k24.wav"
    surround = deliv / "REMIX_mix_5.1_48k24.wav"
    shutil.copy(adir / "REMIX_mix_stereo_48k24.wav", stereo)
    shutil.copy(adir / "REMIX_mix_5.1_48k24.wav", surround)
    stems = deliv / "stems"
    stems.mkdir(exist_ok=True)
    for s in ("music", "sfx", "vo"):
        shutil.copy(adir / f"stem_{s}.wav", stems / f"REMIX_stem_{s}.wav")

    log("[3/7] captions")
    caps = captions.chunks(vo)
    srt, vtt = deliv / "REMIX_captions_en.srt", deliv / "REMIX_captions_en.vtt"
    captions.write_srt(caps, srt)
    captions.write_vtt(caps, vtt)
    log(f"  {len(caps)} caption events")

    log("[4/7] picture: 16:9 master")
    vid = out / "picture_16x9.mp4"
    render.render(vid, P["w"], P["h"], 0.0, C.DURATION, feats, events, P["crf"], P["preset"], P["workers"], log=log)
    tag = {"smoke": "SMOKE_360p", "preview": "PREVIEW_1080p", "master": "4K_UHD"}[args.profile]
    master = deliv / f"REMIX_master_{tag}_24p_16x9.mp4"
    mux_master(vid, stereo, surround, srt, master)
    vid.unlink()

    socials = []
    if not args.skip_social:
        log("[5/7] picture: 9:16 social cutdowns (scenes 2, 4, 6)")
        for sc in [s for s in C.SCENES if s.social]:
            sv = out / f"social_{sc.id}.mp4"
            sc_caps = [c for c in caps if c[1] > sc.start and c[0] < sc.end]
            render.render(sv, P["vw"], P["vh"], sc.start, sc.end, feats, events, P["crf"], P["preset"], P["workers"],
                          captions=sc_caps, log=log)
            sa = social_audio(stereo, sc, out / f"social_{sc.id}.wav")
            so = deliv / f"REMIX_social_9x16_scene{sc.id[1]}_{sc.title.split(':')[0].replace(' ', '-').replace(chr(39), '')}.mp4"
            ff("-i", sv, "-i", sa, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
               "-t", f"{sc.dur:.3f}", "-movflags", "+faststart", so)
            sv.unlink()
            sa.unlink()
            socials.append((sc, so))
    else:
        log("[5/7] social cutdowns skipped")

    log("[6/7] review stills + contact sheet")
    sd = out / "stills"
    sd.mkdir(exist_ok=True)
    stills = []
    for t in STILL_TIMES:
        p = sd / f"still_{t:06.1f}.png"
        ff("-ss", f"{t:.3f}", "-i", master, "-frames:v", "1", p)
        stills.append((p, f"{int(t // 60):02d}:{t % 60:04.1f}  {C.scene_at(t).id.upper()}"))
    contact_sheet(stills, deliv / "REMIX_contact_sheet.png")

    log("[7/7] QC")
    results = []
    results += qc.check_vo(vo)
    results += qc.check_video(master, P["w"], P["h"], C.DURATION, audio_tracks=2, subs=True)
    r, st = qc.check_audio_file(stereo, -14.0, -1.0, 2)
    results += r
    r, _ = qc.check_audio_file(surround, -24.0, -2.0, 6, weights=[1, 1, 1, 0, 1.41, 1.41])
    results += r
    results += qc.check_silence(st)
    vo_stem, _ = sf.read(stems / "REMIX_stem_vo.wav")
    results += qc.check_dialogue_clarity(st, vo_stem, vo)
    results += qc.check_subs(srt, len(caps))
    results += qc.check_stills([p for p, _ in stills])
    six, _ = sf.read(surround)
    lfe_rms = 20 * np.log10(np.sqrt(np.mean(six[:, 3] ** 2)) + 1e-12)
    results.append(("5.1 LFE channel active (enhanced)", lfe_rms > -40, f"LFE RMS {lfe_rms:.1f} dBFS"))
    c_corr = np.corrcoef(six[:, 2], np.pad(vo_stem, (0, len(six) - len(vo_stem)))[: len(six)])[0, 1]
    results.append(("5.1 dialogue anchored in centre", c_corr > 0.8, f"C/VO correlation {c_corr:.2f}"))
    for sc, so in socials:
        results += qc.check_video(so, P["vw"], P["vh"], sc.dur, audio_tracks=1, allow_black_at=((sc.dur - 0.6, sc.dur + 0.1),) if sc.id == "s2" else ())
    ok = qc.report(results, deliv / "QC_report.json", deliv / "QC_report.md", f"REMIX QC — {args.profile} profile")

    manifest = dict(title=C.TITLE, subtitle=C.SUBTITLE, profile=args.profile, duration=C.DURATION, fps=C.FPS,
                    audio_stats=stats, qc_pass=ok, render_seconds=round(time.time() - t_start, 1),
                    files={str(p.relative_to(deliv)): dict(bytes=p.stat().st_size, sha256=sha256(p))
                           for p in sorted(deliv.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"})
    (deliv / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    log(f"\nQC {'PASS' if ok else 'FAIL'} — {sum(r[1] for r in results)}/{len(results)} checks — "
        f"{time.time() - t_start:.0f}s — {deliv}")
    for n, p, d in results:
        if not p:
            log(f"  FAIL: {n}: {d}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
