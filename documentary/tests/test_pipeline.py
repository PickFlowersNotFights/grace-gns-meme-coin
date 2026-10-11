"""REMIX pipeline tests.

Fast suite (seconds):      python -m unittest discover -s tests -v
Full end-to-end smoke:     REMIX_E2E=1 python -m unittest tests.test_pipeline.EndToEndSmoke -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config as C  # noqa: E402
from pipeline import mix, synth as S  # noqa: E402


class Timeline(unittest.TestCase):
    def test_scenes_cover_brief_exactly(self):
        self.assertEqual(C.SCENES[0].start, 0.0)
        self.assertEqual(C.SCENES[-1].end, C.DURATION)
        self.assertEqual(C.DURATION, 270.0)                     # 04:30
        for a, b in zip(C.SCENES, C.SCENES[1:]):
            self.assertEqual(a.end, b.start, f"gap between {a.id} and {b.id}")
        brief = {"s1": (0, 10), "s2": (10, 35), "s3": (35, 60), "s4": (60, 95), "s5": (95, 135), "s6": (135, 225), "s7": (225, 270)}
        for s in C.SCENES:
            self.assertEqual((s.start, s.end), brief[s.id])

    def test_social_cutdowns_are_scenes_2_4_6(self):
        self.assertEqual([s.id for s in C.SCENES if s.social], ["s2", "s4", "s6"])

    def test_vo_cues_ordered_and_inside_film(self):
        prev = 0
        for c in C.VO:
            self.assertGreaterEqual(c.start, prev)
            self.assertLess(c.start, c.hard_end)
            self.assertLessEqual(c.hard_end, C.DURATION)
            prev = c.hard_end

    def test_playbook_steps_contiguous(self):
        self.assertEqual(C.STEPS[0]["start"], 135.0)
        self.assertEqual(C.STEPS[-1]["end"], 225.0)
        for a, b in zip(C.STEPS, C.STEPS[1:]):
            self.assertEqual(a["end"], b["start"])

    def test_delivery_profiles(self):
        m = C.PROFILES["master"]
        self.assertEqual((m["w"], m["h"]), (3840, 2160))
        self.assertEqual((m["vw"], m["vh"]), (1080, 1920))
        self.assertEqual(C.FPS, 24)


class DSP(unittest.TestCase):
    def test_primitives_finite_and_bounded(self):
        for name, x in [("kick", S.kick()), ("snare", S.snare()), ("hat", S.hat()), ("acid", S.acid_note(110, 0.2, True)),
                        ("supersaw", S.supersaw(S.mtof([57, 60, 64]), 0.3)), ("choir", S.choir(S.mtof([57, 64]), 0.4)),
                        ("crowd", S.crowd(0.6)), ("riser", S.riser(0.5)), ("crackle", S.vinyl_crackle(0.5))]:
            self.assertTrue(np.isfinite(x).all(), name)
            self.assertLess(np.max(np.abs(x)), 4.0, name)
            self.assertGreater(np.max(np.abs(x)), 1e-3, name)

    def test_bs1770_calibration(self):
        # 997 Hz sine at -20 dBFS peak, one channel -> -23.0 LUFS (BS.1770-4 reference behaviour)
        t = np.arange(C.SR * 5) / C.SR
        x = 0.1 * np.sin(2 * np.pi * 997 * t)
        self.assertAlmostEqual(mix.lufs(x[:, None]), -23.0, delta=0.15)

    def test_limiter_true_peak(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((C.SR * 3, 2)) * 0.5
        y = mix.limit(x, -1.0)
        self.assertLessEqual(mix.true_peak_db(y), -0.95)

    def test_master_hits_target(self):
        rng = np.random.default_rng(1)
        x = S.lp(rng.standard_normal(C.SR * 6), 4000)[:, None].repeat(2, 1) * 0.2
        y = mix.master(x, -14.0, -1.0)
        self.assertAlmostEqual(mix.lufs(y), -14.0, delta=0.5)
        self.assertLessEqual(mix.true_peak_db(y), -0.95)


class Frames(unittest.TestCase):
    """Every scene renders at landscape and portrait sizes, with or without event data."""

    def _ctx(self, W, H):
        from pipeline.gfx import Ctx
        nf = int(C.DURATION * C.FPS)
        feats = {k: list(np.abs(np.sin(np.arange(nf) * 0.3)).round(3)) for k in ("rms", "low", "high", "vo")}
        feats["env"] = list(np.abs(np.sin(np.arange(int(C.DURATION * 100)) * 0.05)).round(3))
        ev_path = next((ROOT / "build" / "cache").glob("audio_*/events.json"), None) if (ROOT / "build" / "cache").exists() else None
        events = json.loads(ev_path.read_text()) if ev_path else {}
        return Ctx(W, H, C.FPS, feats, events)

    def test_every_scene_landscape_and_portrait(self):
        from pipeline.scenes import render_frame
        times = [0.5, 2.5, 5.0, 8.0, 9.6, 20.0, 30.0, 37.0, 45.0, 55.0, 65.0, 80.0, 90.0, 100.0, 112.0, 125.0,
                 140.0, 165.0, 190.0, 215.0, 240.0, 260.0]
        for W, H in ((320, 180), (180, 320)):
            ctx = self._ctx(W, H)
            for t in times:
                img = render_frame(ctx, t)
                self.assertEqual(img.size, (W, H), f"t={t}")
                self.assertEqual(img.mode, "RGB")
                a = np.asarray(img, dtype=np.float32)
                self.assertGreater(a.std(), 2.0, f"frame at t={t} ({W}x{H}) is flat")

    def test_captions_timing(self):
        from pipeline.captions import chunks
        fake = [dict(cue=c, start=c.start, end=c.start + 0.8 * (c.hard_end - c.start)) for c in C.VO]
        caps = chunks(fake)
        self.assertTrue(all(a < b for a, b, _ in caps))
        self.assertTrue(all(caps[i][1] <= caps[i + 1][0] + 1e-6 for i in range(len(caps) - 1)))
        self.assertTrue(all(len(s) <= 64 for _, _, s in caps))


@unittest.skipUnless(os.environ.get("REMIX_E2E"), "set REMIX_E2E=1 to run the full-length end-to-end smoke render")
class EndToEndSmoke(unittest.TestCase):
    def test_full_pipeline_smoke_profile_passes_qc(self):
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, "-m", "pipeline.run", "--profile", "smoke", "--out", td], cwd=ROOT)
            self.assertEqual(r.returncode, 0, "pipeline/QC failed — see QC_report.md")
            deliv = Path(td) / "deliverables"
            qc = json.loads((deliv / "QC_report.json").read_text())
            self.assertTrue(all(c["passed"] for c in qc), [c for c in qc if not c["passed"]])
            names = {p.name for p in deliv.iterdir()}
            for need in ["REMIX_master_SMOKE_360p_24p_16x9.mp4", "REMIX_mix_stereo_48k24.wav", "REMIX_mix_5.1_48k24.wav",
                         "REMIX_captions_en.srt", "REMIX_captions_en.vtt", "REMIX_contact_sheet.png", "MANIFEST.json"]:
                self.assertIn(need, names)
            self.assertEqual(len([n for n in names if n.startswith("REMIX_social_9x16")]), 3)


if __name__ == "__main__":
    unittest.main()
