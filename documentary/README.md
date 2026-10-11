# REMIX — Fatboy Slim, Big Beat & the Legacy of Electronic Festival Culture

A pipeline that turns the 04:30 storyboard and script into a finished, QC-verified set of deliverables with one command. Everything is original and generated: a neural-TTS narrator, a synthesised score and sound design, and procedurally rendered 16mm-style "archival" and 4K digital plates.

```
python -m pipeline.run --profile master
```

## Deliverables (`build/<profile>/deliverables/`)

| File | Spec |
|---|---|
| `REMIX_master_4K_UHD_24p_16x9.mp4` | 3840×2160, 24p, H.264 High 4:2:0, BT.709, CRF 14 slow, `-tune grain`. **Track 1** Stereo AAC 320k at −14 LUFS / −1 dBTP. **Track 2** 5.1 E-AC-3 640k at −24 LUFS / −2 dBTP. **Track 3** English captions (mov_text). |
| `REMIX_social_9x16_scene2_…mp4`, `…scene4_…`, `…scene6_…` | 1080×1920, 24p. Laid out natively for vertical (stacked split screen, not a centre crop), with captions burned in for sound-off viewing and audio loudness-normalised per cut. |
| `REMIX_mix_stereo_48k24.wav` | Stereo print master, 48 kHz / 24-bit |
| `REMIX_mix_5.1_48k24.wav` | 5.1 bed (L R C LFE Ls Rs), dialogue in the centre channel, enhanced LFE for festival simulation. Ready to use as the **Atmos bed** (see the note below). |
| `stems/REMIX_stem_{music,sfx,vo}.wav` | Separate stems for re-mixing or localisation |
| `REMIX_captions_en.srt` / `.vtt` | Caption sidecars |
| `REMIX_contact_sheet.png` | 25 review frames, one per beat of the storyboard |
| `QC_report.md` / `.json` | Automated delivery QC (see below) |
| `MANIFEST.json` | SHA-256 and byte size of every deliverable, plus loudness stats and render time |

## Profiles

| Profile | Picture | Purpose | Approx. time (4 vCPU) |
|---|---|---|---|
| `smoke` | 640×360 and 360×640 | Full-length, full-pipeline CI gate | about 12 min cold, about 5 min with the audio cache |
| `preview` | 1920×1080 and 1080×1920 | Editorial review copy | about 40 min |
| `master` | 3840×2160 and 1080×1920 | Release master | several hours |

All three profiles use the same timeline, audio and QC. Only the raster changes. The audio stage is cached by a hash of its source code, so a picture-only change does not re-render sound.

## Setup

```bash
cd documentary
pip install -r requirements.txt
scripts/fetch_models.sh               # Kokoro v1.0 TTS model (~350 MB, Apache-2.0)
# needs: ffmpeg ≥ 6 with libx264, aac, eac3; Inter + DejaVu fonts
```

## Verified smoke testing

```bash
scripts/smoke.sh                      # unit tests + full-length end-to-end smoke render + QC gate
```

- **Unit tests** (`tests/test_pipeline.py`, a few seconds) cover:
  - the timeline matches the brief to the second;
  - VO windows;
  - DSP primitives are finite and bounded;
  - BS.1770 loudness calibration;
  - the true-peak limiter;
  - every scene renders in landscape and portrait;
  - caption timing.
- **End-to-end smoke** (`REMIX_E2E=1`) runs the whole pipeline at thumbnail resolution, full length and 24 fps. It asserts that every QC check passes and that every deliverable exists.
- **QC gate** (`pipeline/qc.py`). It runs on every profile, and a failure makes the run exit non-zero:
  - codec, profile, resolution, 24p, yuv420p, BT.709 tags;
  - exact frame count and duration;
  - A/V sync on each audio track;
  - no unintended black (`blackdetect`) and no frozen picture (`freezedetect`);
  - stereo −14 ±1 LUFS and ≤ −1 dBTP; 5.1 −24 ±1 LUFS and ≤ −2 dBTP;
  - no audio dropouts;
  - every VO line fits its window;
  - dialogue-to-bed ratio of at least 6 dB on every line;
  - SRT well-formed, with reading speed of 25 cps or less;
  - LFE active and dialogue anchored in the centre channel;
  - every review still has picture content;
  - the specs of all three social cuts.

## Architecture

```
config.py    storyboard timeline: scenes, VO cues and windows, data table, playbook steps (single source of truth)
tts.py       Kokoro neural VO → 48 kHz, cached, auto-fit to window (bounded speed-up)
synth.py     DSP: band-limited oscillators, resonant filters, drums, 303 acid, supersaw, formant choir, crowd, tape/vinyl FX
score.py     original 4:30 score + sound design per scene; emits an event log (kicks, drops, CO2 blasts, mini-drops…)
mix.py       dialogue ducking, bus compression, BS.1770-4 loudness, true-peak limiter, 5.1 bed, per-frame audio features
gfx.py       resolution-independent 2D toolkit: 16mm film emulation, bloom, kinetic typography, VU, waveforms, silhouettes
scenes.py    the seven scene renderers + global type, act cards, glitch cuts
render.py    multiprocess frame renderer → x264 chunks → lossless concat
captions.py  caption chunking, SRT/VTT, burned-in social captions
qc.py        automated delivery QC
run.py       orchestrator: preflight → audio → captions → 16:9 → 9:16 → stills → QC → manifest
```

The visuals are **audio-reactive by construction**. Strobes, LED walls, CO2 blasts, crowd bounce and the waveform decks are all driven by the score's event log and by per-frame features of the final mix, so picture and sound stay in sync frame for frame.

## How the brief maps to the cut

| Brief | Implementation |
|---|---|
| S1 needle drop → 1998 basement → 2026 LED stage, tape rewind | VU-meter cold open, then a 16mm vinyl macro, a 4:3 basement rave and a 2026 LED wall with lasers. The cut ends on a glitch rewind. |
| S2 split screen, "Praise You" swing into rigid EDM build | A swung 112 BPM original break (60% swing, humanised) tape-stops into a quantised 128 BPM supersaw build. The 2026 side has the single CUE button and CO2 cannons on the bar. The drop is withheld (0.55 s of black). |
| S3 vinyl archive, track disassembly, AI sample pack | Crate dolly with depth of field, then three spectral lanes (60s funk drums / 70s rock bassline / gospel acapella), then a hard cut to an "Auto-Gen Studio" filling a perfect grid. |
| S4 POV muffled → clear, drop, storyboard snapshot | 16mm POV push-in with an earplug low-pass (300 Hz → full range), a build, a blackout, then the drop at 75.6 s. The shot sequence is WIDE 1998 → MEDIUM PUSH 2026 → EXTREME CLOSE of hands on the mixer, with the filter knob moving in sync with the audible sweep. |
| S5 Brighton data viz, crowd roar → tickers and registers | Dusk beach with about 250k crowd dots, log-scale bar and line charts, a morph into a corporate ticker floor where cash registers fire on the beat, and the brief's table. |
| S6 four-step playbook with a live 90 s mini-mix | Step 1: the hook "Make some noise!" over 96 → 128 → 172 BPM. Step 2: a formant-synth gospel acapella over a 130 BPM funky break, with the channels collapsing together. Step 3: six unannounced 2-beat low-end cuts with strobes. Step 4: the crescendo with a stage mock-up flaring warm at the peak. |
| S7 pull-back, title card, tape hiss | Slow-motion pull-back from behind the booth over an ocean of phone-lit hands, then the "REMIX" title slam with a non-affiliation disclaimer, ending on hiss and a fade. |
| 9:16 for Scenes 2, 4, 6 | Re-rendered natively in portrait with burned captions |
| Stereo & Dolby Atmos | Stereo print master plus a 5.1 bed with enhanced LFE (see the note below) |

## Notes and limits

- **Dolby Atmos.** An Atmos master (ADM BWF / IMF) needs a Dolby renderer (the Dolby Atmos Renderer or a DAW with Atmos) to place objects and write the metadata. No open tool does this. The pipeline delivers what that session needs: a 5.1 bed with enhanced LFE, plus separate music, SFX and VO stems to use as objects.
- **Archival footage and likeness.** The brief calls for archival footage of Norman Cook and of Brighton 2002. This cut uses original reconstructions and generic silhouettes, so it is publishable without clearances. To upgrade to licensed archive, load a frame from the plate inside the matching `_panel_1998` / `_basement_1998` / `_beach` function in `scenes.py`. The 16mm treatment, overlays, type and timing stay as they are.
- **Claims.** Read `FACT_CHECK.md` before release. Several figures in Scene 5 come from the brief and are unsourced.
