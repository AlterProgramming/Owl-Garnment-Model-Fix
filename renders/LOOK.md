# LOOK — owl-kente-wrap, rebuild of 2026-08-23 03:1x (`--cloth-res 76,42`)

Supersedes `../diag_wrap/LOOK.md` (that build's sheets are still there for comparison).
Sheets here: `renders/sheet_shipped_turntable.png` (textured, the one to judge),
`sheet_garment_gray_turntable.png`, `sheet_garment_gray_posed.png`, `sheet_garment_gray_zooms.png`
(now aimed at the knot / flank / tail / hem the build measured for itself).
Viewer: `../owl-kente-wrap.html`. Kiosk: `owl-guide-kente-wrap.glb`, 1.68 MB, already in
`Afromaha-Visit2D/src/renderer/public/models/guide/` behind `?guideModel=owl-guide-kente-wrap`.

## Gates — 6/8 (was 5/8)

| gate | value | threshold | verdict |
|---|---|---|---|
| G1 worn: pinned row drift | 0.0015 | <= 0.010 H | PASS |
| G2 support only | 0 | = 0 | PASS |
| G3 no penetration, worst clip | **6** (was 58) | <= 20 | **PASS** |
| G4 hem above the beads and level | 0.1427, sigma 0.0075 | >= 0.122 H, sigma <= 0.015 | PASS |
| G5 root covered | **0.419** (was 0.528) | >= 0.90 | FAIL — ticket W-G5 |
| G6 tail clear of the waving wing | 0 | = 0 | PASS |
| G7 stretch | median 1.041, p90 1.16 | median <= 1.05, p90 <= 1.12 | FAIL |
| G8 collar visible | 0 | = 0 | PASS |

The build these numbers come from is the final one of the night (knot 0.095 H, tail 0.15 H wide).
G3 was fixed structurally: the support loop now clears where the raised wing's **root band sweeps**
through the clips, not only where it rests (`wave` turns `wing_left` 22 deg of lift, 20 of swing, and the
cloth may only be skinned to the torso). G5 fell for the same reason — covering that root and clearing its
sweep are one lever pulled in opposite directions. **G5 is yours to decide (W-G5).**
G3's *count* is also a poor statistic — it swings 6 ↔ 67 on changes nothing can see (ticket **W-G3**);
the guide build at 14k faces reads 57 with this same code, and `../probes/sheet_wave_peak.png` shows the
flagged frame from four angles behind the owl with nothing visible.

## What I see (defects first)

1. **It reads as a wrapper around the hips more than a toga over the shoulder.** From the front the cloth
   starts under the AI-CCORE collar sash and covers belly and hips; the diagonal over the raised shoulder is
   there but thin. The tie is at 0.43 H because the collar rim is at 0.46 H — it cannot go higher without
   crossing the band (G8). Raising the read means widening the shoulder diagonal, not raising the tie.
2. **The tail** was a flat panel reading as a bib. It is now tapered (0.62), cut on the diagonal (0.18 of its
   length) and given a per-column twist (22°), and — the change that mattered most — **narrowed from 0.22 H to
   0.15 H**, which turns it from a hanging panel into a sash end. It is still flat cloth, not a fold; real
   secondary motion is IL-1/W-IL2, post-demo.
3. **The hem flares like a lampshade** with a flat shelf jutting at the front-left. Structural (the radial
   hull is a running maximum; the hop's swing bound holds the hem out) — not fixed tonight.
4. **The knot** is now a cinched bunch with a band and two short ends (1016 faces, was 2132 sausages). It
   reads as a small tied bundle rather than the epaulette it was. At `knot_size` 0.07 H it vanished at kiosk
   distance (measured on the guide turntable), so it is now **0.095 H** and legible from the front. The sheet
   still arrives at it smooth, so the cloth does not visibly *gather* into the knot. Three earlier shapes were rejected on
   sight: sausage saucer (`../knot_k1.png`), trefoil hoop (`../knot_k1b.png`), clenched claw (`../knot_k1c.png`);
   this is `../knot_k1d.png`.

What is right: the diagonal across the chest **under** the tablet; the folded wing lying **over** the cloth's
edge; the collar band fully visible; big soft folds; the hem level at 0.14–0.18 H above the anklets; the
garment travels with the torso through every clip; nothing pokes through the wing on the wave any more.

## Two more sheets

* `renders/sheet_before_after.png` — the first wrap build over tonight's, front / right / back / left.
* `renders/sheet_guide_wave.png` — the **shipped guide** through the whole wave (t = 0.40 / 0.80 / 1.28 / 1.60,
  from yaw 200 and 250, right at the raised wing's base). Nothing pokes through the wing in any frame, which is
  the evidence behind ticket W-G3: the guide's G3 count of 57 is not something anyone can see. (There is a small
  dark wedge at the cloth's edge behind the wing in every frame — constant, so it is geometry rather than a
  contact; worth a look if it bothers you.)
* `renders/sheet_guide_turntable.png` — the guide as it ships, 14k faces, 1.68 MB.
