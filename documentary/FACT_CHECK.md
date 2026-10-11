# REMIX — Fact-check & legal sign-off sheet

The pipeline renders the brief's claims as written. These items need an editorial or legal sign-off before public release. QC checks the technical spec only; it cannot check whether a claim is true.

| # | Where | Claim (as rendered) | Status | Note |
|---|---|---|---|---|
| 1 | Scene 5, table and charts | 1998 average artist guarantee **$2,500–$10,000** | ⚠ Unsourced | Figure comes from the brief. Needs a source, or label it "estimate". |
| 2 | Scene 5, table and charts | 2002 guarantee **$50,000** | ⚠ Unsourced | Big Beach Boutique II was a **free** event. Say what the $50K stands for, or drop it. |
| 3 | Scene 5, table | 2002 key metric "**Unpoliced** Mega-Events" | ⚠ Risky wording | Police were present at Brighton 2002. The event was criticised for overcrowding and under-planning. Suggested wording: "Free, Under-Planned Mega-Events". |
| 4 | Scene 5, table and VO | Brighton Beach crowd **~250,000** | ✅ Widely reported | Contemporary press put the crowd at about 250,000 against about 60,000 planned. Attribute it on screen as "as widely reported", which the pipeline already does. |
| 5 | Scene 5, table and charts | 2026 guarantee **$300K–$1M+**, crowd **80,000+ per day** | ⚠ Unsourced | These are plausible tier-one festival ranges. Cite an industry source or label them as estimates. |
| 6 | Scene 2, VO | Mainstages "commodified his organic style into predictable, paint-by-numbers drops" | ℹ Opinion | This is editorial opinion. The VO frames it as an argument, which is fine. |
| 7 | Scene 4, VO and captions | "50,000 strangers" in the VO, but the storyboard caption says "250,000 on the beach" | ℹ Intentional? | The two numbers describe different events. Confirm the mismatch is intended. |
| 8 | Title, Scene 6 | Uses the name "Fatboy Slim" and "The Fatboy Slim Playbook" | ℹ Nominative use | This is editorial use in a documentary. The end card carries a non-affiliation disclaimer, and no likeness, logo or recording of the artist is used. Legal should confirm for the target territories. |

On-screen footnote in Scene 5, set in `pipeline/scenes.py` (`_charts`): *"Figures: production research estimates, rounded. Crowd figure as widely reported."*
To correct a figure, edit `DATA_TABLE` in `pipeline/config.py` and re-run. Charts, table and labels all update from that one source.

## Rights position of this cut

- **Music:** every note is synthesised by `pipeline/score.py`. There are no samples from third-party recordings. The brief's reference to the drums of *"Praise You"* is matched with an **original swung 112 BPM breakbeat**, not the record.
- **Voice:** the narration and the "Make some noise!" hook are generated with the Kokoro-82M TTS model (Apache-2.0), voice `bm_george`.
- **Picture:** every plate is an original procedural reconstruction. Generic silhouettes stand in for any real person, and there is no archival footage. Each scene renderer is a function of time, so licensed archival plates can be swapped in later (see README, "Upgrading to licensed archive").
- **Fonts:** Inter (SIL OFL 1.1), DejaVu Sans Mono (Bitstream Vera licence) and Liberation Mono (SIL OFL).
