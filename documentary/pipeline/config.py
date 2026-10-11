"""Storyboard timeline for REMIX — the single source of truth.

Every renderer, the score, the VO engine, subtitles, cutdowns and QC read
timings from here. Times are seconds on the master timeline (00:00-04:30).
"""
from dataclasses import dataclass, field

TITLE = "REMIX"
SUBTITLE = "Fatboy Slim, Big Beat & the Legacy of Electronic Festival Culture"
DURATION = 270.0
SR = 48000
FPS = 24

# ---------------------------------------------------------------- profiles
PROFILES = {
    # full-length, full-pipeline run at thumbnail resolution (CI / pre-flight)
    "smoke": dict(w=640, h=360, vw=360, vh=640, crf=28, preset="veryfast", workers=4),
    # editorial review copy
    "preview": dict(w=1920, h=1080, vw=1080, vh=1920, crf=20, preset="medium", workers=4),
    # release master: 4K UHD 24p 16:9, 9:16 social at 1080x1920
    "master": dict(w=3840, h=2160, vw=1080, vh=1920, crf=14, preset="slow", workers=4),
}


@dataclass(frozen=True)
class Scene:
    id: str
    act: str
    title: str
    start: float
    end: float
    social: bool = False          # gets a 9:16 cutdown

    @property
    def dur(self):
        return self.end - self.start


SCENES = [
    Scene("s1", "ACT I", "The Ignition", 0.0, 10.0),
    Scene("s2", "ACT I", "Contrarian Dashboard", 10.0, 35.0, social=True),
    Scene("s3", "ACT I", "What They Missed: The Lost Art of Sampling", 35.0, 60.0),
    Scene("s4", "ACT II", "The Illusion of 'Just Playing Records'", 60.0, 95.0, social=True),
    Scene("s5", "ACT II", "Brighton Beach to Billion-Dollar Mainstages", 95.0, 135.0),
    Scene("s6", "ACT III", "The Fatboy Slim Playbook", 135.0, 225.0, social=True),
    Scene("s7", "ACT III", "Resolution & The Outro", 225.0, 270.0),
]

ACTS = [
    ("ACT I", "The Crate & The Algorithm", 0.0, 60.0),
    ("ACT II", "The Psychology of the Decibel", 60.0, 135.0),
    ("ACT III", "The Blueprint & The Future", 135.0, 270.0),
]


def scene_at(t):
    for s in SCENES:
        if s.start <= t < s.end:
            return s
    return SCENES[-1]


# ---------------------------------------------------------------- voiceover
@dataclass(frozen=True)
class Cue:
    id: str
    start: float
    hard_end: float               # VO must finish before this (QC-enforced)
    text: str                     # spoken text (TTS-friendly spelling)
    caption: str = ""             # on-screen caption text (defaults to text)

    @property
    def display(self):
        return self.caption or self.text


VOICE = dict(voice="bm_george", lang="en-gb", speed=1.0, max_speed=1.18)

VO = [
    Cue("vo1", 0.8, 13.0,
        "Nostalgia is a comfortable lie. We look at the nineties big beat explosion as the birth "
        "of modern festival culture. But did it evolve? Or did it get commercialized to death?",
        "Nostalgia is a comfortable lie. We look at the nineties big beat explosion as the birth "
        "of modern festival culture — but did it evolve, or did it get commercialized to death?"),
    Cue("vo2", 13.4, 34.6,
        "Fatboy Slim's sets were chaotic, organic, and unpredictable. Today's glorified stadium "
        "performances obscure a harsher truth: mainstages have commodified his organic style into "
        "predictable, paint-by-numbers drops. Why are thirty-year-old breakbeats still outshining "
        "contemporary headliners?"),
    Cue("vo3", 36.0, 59.6,
        "We lost something along the way: the art of aggressive, creative, copyright-challenging "
        "sampling. In an era dominated by sample packs and automated music generators, we're losing "
        "the human crate-digging collage that gave Big Beat its soul."),
    Cue("vo4a", 62.0, 73.2,
        "I used to be a skeptic. Standing at the back of a crowded field, I thought: he's just "
        "standing up there, playing other people's records.",
        "I used to be a skeptic. Standing at the back of a crowded field, I thought: "
        "'He's just standing up there playing other people's records.'"),
    Cue("vo4b", 76.2, 94.6,
        "Then came the drop. It wasn't just music. It was a high-art psychological game. In that "
        "single moment, pacing, tension, and collective memory turned fifty thousand strangers into "
        "one single breathing entity.",
        "Then came the drop. It wasn't just music — it was a high-art psychological game. In that "
        "single moment, pacing, tension, and collective memory turned 50,000 strangers into one "
        "single breathing entity."),
    Cue("vo5", 99.0, 134.6,
        "Dance music didn't just grow. It transformed structurally. From unpoliced underground "
        "gatherings in nineteen ninety-eight, to Brighton Beach's quarter-million-person inflection "
        "point, to today's billion-dollar corporate festival infrastructure. The superstar DJ became "
        "live entertainment's most lucrative asset.",
        "Dance music didn't just grow; it transformed structurally. From unpoliced underground "
        "gatherings in 1998 to Brighton Beach's quarter-million-person inflection point, to today's "
        "billion-dollar corporate festival infrastructure. The superstar DJ became live "
        "entertainment's most lucrative asset."),
    Cue("vo6a", 136.4, 147.0,
        "Step one. Anchor with universal vocal hooks. Loop a recognizable three-word hook across "
        "wildly different tempos and genres.",
        "Step 1 — Anchor with universal vocal hooks. Loop a recognizable 3-word vocal hook across "
        "wildly different BPMs and genres."),
    Cue("vo6b", 158.6, 170.0,
        "Step two. Layer contrasting acapellas over breakbeats. Drop a seventies acapella over a "
        "hundred-and-thirty B.P.M. funky break, to build harmonic tension.",
        "Step 2 — Layer contrasting acapellas over breakbeats. Drop a 1970s acapella over a "
        "130 BPM funky breakbeat to build harmonic tension."),
    Cue("vo6c", 181.0, 191.4,
        "Step three. Execute chaotic mini-drops. Cut the low end instantly, without warning, to "
        "reset the crowd's attention span.",
        "Step 3 — Execute chaotic mini-drops. Cut the low end instantly without warning to reset "
        "the crowd's attention span."),
    Cue("vo6d", 203.4, 214.0,
        "Step four. Build a cinematic crescendo. Combine classic hooks with fresh rhythms, for a "
        "payoff that feels timeless, not stagnant.",
        "Step 4 — Build a cinematic crescendo. Combine classic hooks with fresh rhythms for a "
        "payoff that feels timeless, not stagnant."),
    Cue("vo7", 229.5, 257.0,
        "The tools have changed, but the fundamental mission of the DJ remains unchanged. It's not "
        "about pushing buttons. It's about controlling the room, reclaiming the soul of the sample, "
        "and engineering collective human joy.",
        "The tools have changed, but the fundamental mission of the DJ remains unchanged: it's not "
        "about pushing buttons. It's about controlling the room, reclaiming the soul of the sample, "
        "and engineering collective human joy."),
]

# ---------------------------------------------------------------- data (Scene 5)
# Figures as supplied in the production brief. They are flagged in
# FACT_CHECK.md and must be signed off before broadcast/release.
DATA_TABLE = [
    dict(era="1998", label="Underground", metric="Free Raves / Club Dates",
         fee_lo=2_500, fee_hi=10_000, fee_txt="$2.5K – $10K",
         crowd_lo=1_500, crowd_hi=5_000, crowd_txt="1,500 – 5,000"),
    dict(era="2002", label="Transition", metric="Unpoliced Mega-Events",
         fee_lo=50_000, fee_hi=50_000, fee_txt="$50K",
         crowd_lo=250_000, crowd_hi=250_000, crowd_txt="250,000 (Brighton Beach)"),
    dict(era="2026", label="Corporate", metric="Global Festival Circuits",
         fee_lo=300_000, fee_hi=1_000_000, fee_txt="$300K – $1M+",
         crowd_lo=80_000, crowd_hi=80_000, crowd_txt="80,000+ per day"),
]

# ---------------------------------------------------------------- playbook (Scene 6)
STEPS = [
    dict(n=1, title="ANCHOR WITH UNIVERSAL VOCAL HOOKS", start=135.0, end=157.5,
         visual="Isolated vocal track running across the deck",
         action="Loop a recognizable 3-word vocal hook across wildly different BPMs/genres."),
    dict(n=2, title="LAYER CONTRASTING ACAPELLAS OVER BREAKBEATS", start=157.5, end=180.0,
         visual="Two waveform channels collapsing into one another",
         action="Drop a 1970s acapella over a 130 BPM funky breakbeat to build harmonic tension."),
    dict(n=3, title="EXECUTE CHAOTIC MINI-DROPS", start=180.0, end=202.5,
         visual="Waveform cuts out for 2 beats; strobe on beat",
         action="Cut the low end instantly without warning to reset the crowd's attention span."),
    dict(n=4, title="BUILD A CINEMATIC CRESCENDO", start=202.5, end=225.0,
         visual="Multi-layered tracks peak simultaneously",
         action="Combine classic hooks with fresh rhythms for a payoff that feels timeless, not stagnant."),
]

HOOK_TEXT = "Make some noise!"     # original 3-word hook spoken by the TTS voice

# Step 1 genre/BPM sections for the hook loop
HOOK_SECTIONS = [(135.0, 142.5, 96, "HIP-HOP"), (142.5, 150.0, 128, "HOUSE"), (150.0, 157.5, 172, "DRUM & BASS")]
# Step 3 mini-drops: (bar, beat) positions inside step 3 where the low end is cut for 2 beats
MINI_DROP_BEATS = [9, 18, 23, 30, 37, 42]   # beat indices relative to step start (130 BPM)

DISCLAIMER = ("Independent documentary. Not affiliated with, sponsored or endorsed by Norman Cook / "
              "Fatboy Slim. All music, voice and imagery in this cut are original works created for "
              "this production; archival plates are represented by original reconstructions.")
