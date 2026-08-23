# Kente robe — done (2026-08-22)

The owl's kente is now a garment, not a print. Nothing is committed;
`git status` lists every new file. Design, measurements and the bug list:
`docs/superpowers/specs/2026-08-22-owl-kente-robe-design.md`.

## What changed and why

The 2026-08-21 kente pass painted the weave onto an offset copy of the
belly (a 6 mm decal). It read as body paint and failed the standard the
owl's clothing is held to — *ample cloth, air between the subject and the
garment, kente as a robe*. It also did not build: `--kente` failed
`validate_rig` on the real owl with zero-length normals on the vest
primitive (the sash's `_repair_zero_normals` was never applied to the
vest). That is fixed, and both decals are still reachable with
`--kente-style vest|sash`; the default `--kente` is now the robe.

`meshforge/robe.py` builds the robe from the owl's own measurements: a ray
scan in (angle, height) cells, a mirrored torso envelope, a top edge that
stops under the raised wing and drapes over the folded one, a gravity hull
(the cloth falls vertically past the widest point instead of following the
body in), clearance that grows with the fall, vertical pleats, one
rectangular UV chart with the weave baked by position, skin weights from
the nearest torso vertices restricted to the torso joints — and then it
*measures* what it built.

## Numbers (master build, `--kente --beads`)

| | min | p05 | median | vertices inside |
|---|---|---|---|---|
| at rest (distance to 3 M hi-res surface samples) | 0.015 | 0.029 | 0.116 | 0 |
| worst through all six clips (`hop` t=0.42) | 0.008 | 0.029 | 0.114 | 0 |

Owl height 1.90, so the hem stands ~0.10 (5 %) off the body; the top
edge, where the cloth is held, keeps ~1 %; the one tight moment is the
hem beside the folded wing's tip on the hop (0.4 % of the height — that
tip is labelled `leg_right` by the region map, so it tucks with the leg).
Guide build (`--faces 14000 --tex 512 ...`): 0.014 / 0.035 / 0.118, 0
inside; worst through clips 0.015 (`idle`), 0 inside. "Inside" is
confirmed two ways (surface-sample normal *and* ray parity); the one
unconfirmed suspect per build is listed as `suspect_count`.

Earlier prototypes in this session, for scale of what the checks catch:
first build 3,061 vertices inside (33 %), then 76, then 0 at rest but 14
(`tablet_show`), then 56 (`tablet_show`), then 0. Each step is a bug in
the spec's list, and the luna review that followed added four more
(collar centre, the inside test itself, what-is-tested vs what-ships, a
margin measured from the wrong row).

## Pictures

* `2026-08-22-kente-before-after.png` — the vest decal (top row) against
  the robe (bottom row), five views each, same renderer and lighting.
* `2026-08-22-kente-robe-views.png` — front / three-quarter / side / back /
  left, plus `tablet_show` at 0.7 s.
* `2026-08-22-kente-robe-posed.png` — `wave` 0.55 s, `hop` 0.4 s,
  `tablet_show` 0.4 s: the cloth stays put while the wing waves, the
  legs tuck inside the bell on the hop, the hand comes out of the notch.
  (Rendered from the final build; every picture here is.)
* `2026-08-22-kente-robe-guide-build.png` — the 2,208-triangle guide robe.
* `2026-08-22-kente-robe-colorways.png` — `asante_gold` (default) beside
  `ewe_adanudo --kente-seed 3`: same garment, different cloth; the
  clearance report is identical, as it should be.
* `2026-08-22-kente-robe-parameter-map.png` — the (angle, height) cell
  map behind the shape: blue torso, orange draped-over (folded wing, tail),
  red stop (raised wing root), pale = a held-out solid on that ray (hand,
  tablet); green = the collar cap, white = the final top edge.

## Commands

```bash
# master (25 s with the cache)
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-kente-robe.glb --cache .owl_cache --kente --beads
# guide
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-guide-kente-robe.glb --cache .owl_cache \
    --faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256 --kente --beads
# look
python3 tools/owl_shots.py out/owl_kente/owl-kente-robe.glb --out shots \
    --view front:0:4 --view side:90:0 --view back:180:4 --clip tablet_show=0.7
# tests
python3 -m pytest tests/meshforge -q
```

To promote: copy `out/owl_kente/owl-kente-robe.glb` over
`viewer/assets/owl.glb` and `out/owl_kente/owl-guide-kente-robe.glb` over
`viewer/assets/owl-guide.glb` (and re-run `python3 artifact/build.py` /
the Afromaha-Visit2D guide build). Deliberately not done here — the look
should be signed off first, and the kiosk's guide copy lives in another
repo.

## Left deliberately

* The guide build's back panel tops out lower (0.39 of the height, not
  0.50): at 14k faces `collar.measure_band_frame` falls back and the
  coarse collar rim sits lower at the back. Pre-existing measurement
  quality, not a robe defect.
* The woven-relief normal map (`--kente-normal`) is off by default. Its
  green-channel sign is now the glTF one (B = decreasing v, per face from
  the UV winding), verified against the bundled Three.js r160 by reading
  its loader (`normalScale.y*=-1` without TANGENT) and shader (B follows
  increasing v). The previous unconditional `cross(N, T)` was right on the
  robe's mirrored chart and could flip on xatlas charts. Not rendered on
  the owl yet.
* `beads.py` is untouched: the anklets still sit at 0.093 of the height,
  just below the hem at 0.12, so they remain visible.

## Next: from robe to tunic (agreed 2026-08-22, afternoon)

User's read of the robe above: 20/100 — the body of the garment is right,
but nothing *holds it up*. A garment stays up only where the body is
wider below the opening than the opening itself; on this owl that is the
neck base alone. Agreed construction, staged with a render after each:

1. **Neckline + shoulders + armholes.** Neckline at the collar band's
   lower rim all round (tucked under the scarf, ~0 clearance at the rim;
   the band frame's fitted rim at the front, the coarse rim elsewhere).
   T-shirt shoulders over the top of the egg. Left armhole cut around the
   raised wing's root *swept through the wave* (+ margin). Right side
   closed over the folded wing (it is fused to the flank, there is nothing
   to pass cloth through) with a hand slit for the forearm + tablet,
   swept through `tablet_show`. Hull / clearance / pleats / hem as built.
2. **Bindings.** Plain **gold** band along neckline, armhole, hand slit and
   hem — the selvedge that makes cuts read as seams.
3. **Placket + buttons.** Centre-front slit from the neckline (under the
   bib) to ~0.30 of the height, faced in gold, modelled buttoned with
   **three gold (brass) buttons**; the bib hides the top, the lower
   buttons show. That is the "small when fastened, large when open"
   opening the user described as how this kind of clothing is made.
4. Later, as a **second outfit**: the flat kente sheet wrapped under the
   raised wing and thrown over the folded-wing shoulder.

Garment weights: `body`/`chest` only (not `neck` — the neck joint is
runtime-driven by cursor follow, and a tunic's neckline does not turn
with the head).

## Stage 1+2 built; direction changed to SLEEVES (2026-08-22, later afternoon)

Stages 1 (neckline/shoulders/armhole/slit) and 2 (gold bindings 0.035)
are built and rendered (`2026-08-22-kente-tunic-stage{1,2,2-posed}.png`,
WIP `out/owl_kente/owl-kente-tunic-wip.glb` + `.html`). Verified: 0
vertices inside the body at rest and through all six clips (min 0.008
rest, 0.004 mid-wave); `test_robe.py` 24 pass. Cloth at wing height rides
`wing_right` (blend 0.28–0.34 of height), the hem stays on body/chest.

User's review: "it does not wrap the wings — there is simply nothing over
the wings." Agreed new construction, not yet built:

* **Right side:** the wing's shoulder bump pressed against the chest must
  be *draped over*, not cut around — merge held-out crossings labelled
  `wing_right` that start within ~0.05 of the first solid into that solid
  (`classify_cells`), so the cloth runs unbroken from the neckline over
  shoulder and upper arm; the slit shrinks to forearm/hand only (a tight
  cuff).
* **Left side:** a real **sleeve** around the raised wing, built from the
  wing's own cross-sections along its axis (pivot `wing_left` → tip):
  wide at the shoulder (clearance ~0.08, a cap overlapping under the
  tunic's shoulder), tapering to a snug cuff (~0.02); bound to the wing's
  joints (`wing_left` → `wing_left_tip` along its length, root ring
  blended toward `chest`) so it waves with the wing; gold cuff binding;
  weave strips along the sleeve. The tunic's armhole becomes the sleeve
  seam and is cut tight (rest-pose root + small margin).
* **Open decision (asked, unanswered):** sleeve length — stop at ~3/4 of
  the wing so the feathered tip shows past the cuff (recommended) or run
  to the tip.
* Then stage 3 as before: centre-front placket from the bib's rim (0.377)
  to ~0.30, gold facings, three brass buttons. Outfit #2 (the wrap) later.

## Sleeve built (2026-08-22, evening)

The raised wing wears a sleeve and both shoulders are covered. Built,
measured, rendered, not promoted. WIP: `out/owl_kente/owl-kente-sleeve-wip.glb`
(+ `.report.json`, `.html`). Pictures: `2026-08-22-kente-sleeve-views.png`
(front / three-quarter / side / back / right), `-posed.png` (`wave` 0.55 s
front, `wave` 0.95 s left, `tablet_show` 0.6 s front and right),
`-shoulders.png` (left armscye from the side and from behind, right shoulder).

**Decision taken without an answer:** sleeve length = 0.72 of the free wing,
so the feathered tip shows past the cuff (`--sleeve-length`, 0–1, to change).

### Construction (what changed since stage 2)

* **Both wing roots are draped over** (`RobeParams.drape_over` now lists
  `wing_left`, `wing_left_tip`, `wing_right`, `tail`; `stop_below` is empty;
  the armhole machinery stays available through `stop_below`, the synthetic
  tests use it). A drape-over solid that starts within `drape_merge_gap`
  (0.13) of the *first* solid is the same shoulder (the folded wing's upper
  arm, pressed to the chest with 0.05–0.12 of air) and is draped over too —
  that closes the gap the user circled at the right shoulder. Measured from
  the first solid, not chained: chaining swallowed the hand and the tablet
  and the cloth fell *outside* the tablet. Held solids of one label on one
  ray (palm, then tablet) are one obstacle; the cloth does not thread
  between them.
* **`meshforge/sleeve.py`** (new): the wing's own frame (principal axis
  from the `wing_left` pivot (−0.28, −0.17, −0.10), axis (−0.49, 0.87,
  −0.09), extent −0.15..0.59 along it); `separation` from surface samples
  (the wing is free of the body from 0.229 along the axis — the elbow pivot
  sits at 0.212); per-station slices + rays from the wing's centroid →
  `wing_r`, `body_r`; radius = min(wing_r + clearance(s), body_r − 0.03,
  the margin the turning head needs); clearance 0.03 at the root ring
  (0.189, under the tunic) → 0.06 at 0.06 past the separation → 0.03 at
  the cuff (0.489; the wing tapers through the ring, so ~2/3 of it is air). Fused cells are
  cut; cut cells take the smallest kept radius around them so nothing
  interpolates into the body. Grid 31 stations × 48, 2,880 triangles, one
  chart 1024 × 256, strips along the wing, gold cuff band 0.03. Weights
  from the nearest `wing_left` / `wing_left_tip` / `chest` vertices,
  Gaussian-smoothed over the grid (`robe.smooth_grid_weights`, also applied
  to the tunic now).
* **The tunic is cut around the sleeve** where the sleeve leaves it
  (`obstacle_points` = sleeve vertices past the separation): the armscye,
  −126..−81°, 0.42–0.55 of the height. The cuts are decided against the
  cloth's *actual* surface radius (hull + clearance + drape slack + pleats)
  ± `obstacle_near`, not a conservative band — the band put the hand slit
  up to the neckline where the hand was 0.09 off the cloth.
* **Holes are signed-distance fields** (`HoleField`, from the dilated
  mask, exact to the cell boundary, smoothed 1.5 cells without ever
  shrinking) instead of star curves, which inflated the diagonal hand slit.
* `cut_grid` is the shared grid cutter (robe + sleeve); `garment_overlap` /
  `posed_overlap` measure one garment under another.

### Numbers (master, `--kente --beads`, 8.79 MB)

| | tunic | sleeve |
|---|---|---|
| triangles | 17,414 | 2,880 |
| min air at rest (3 M hi-res samples) | 0.0078 | 0.0106 |
| vertices inside the body at rest | 0 | 0 |
| worst of the six clips | `nod` 0.0043 (0 inside) | `wave` 0.0039 (0 inside) |
| head-follow probe (neck yaw ±25°, head pitch ±15°, not a clip) | 0.0020 (0 inside, 3 suspects) | 0.0033 (0 inside) |

Holes: hand + tablet slit 36–64°, 0.31–0.51 of the height (it reaches the
neckline: the hand is held right under the bib, 0.03 off the shoulder
cloth); armscye as above. Sleeve root under the tunic: 196 root-ring
vertices, median 0.032 under the cloth, 19 read as "outside" by ≤ 0.02 —
they sit beside the armscye's cut edge, whose nearest sample faces
sideways (an artefact of the measure at a hole edge, not cloth poking
through: 0 tunic vertices are inside the sleeve). Through the clips: ≤ 25
such vertices, ≤ 0.018.

### Commands

```bash
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-kente-sleeve-wip.glb --cache .owl_cache --kente --beads
#   --sleeve-length 0.72   --sleeve-clearance 0.06   --no-sleeve
python3 tools/owl_shots.py out/owl_kente/owl-kente-sleeve-wip.glb --out shots/x \
    --view front:0:4 --view left:-60:5            # rest; add --clip wave=0.55 for a posed frame (clips apply to every view)
python3 tools/owl_shots.py ... --view lshoulder_side:-75:5 --zoom 2.2 --target=-0.30,0.15,-0.05   # zoom on the armscye
python3 artifact/build.py --glb out/owl_kente/owl-kente-sleeve-wip.glb --out out/owl_kente/owl-kente-sleeve-wip.html
python3 -m pytest tests/meshforge -q                 # test_robe.py 31, test_sleeve.py 11
```

### Left deliberately / next

* The tunic's left neckline stands ~0.16 off the collar around the wing
  root (a boat-neck shelf): the wing's free part emerges exactly at the
  collar's lower rim, so the cloth over the root cannot close back to the
  neck there. The sleeve's armscye sits in that shelf.
* The sleeve's cuff comes within 4 mm of the wing at the peak of the wave
  (the cloth's and the skin's LBS weights differ by a few %); no vertex
  enters the wing. Raise `clearance_cuff` (0.03) if it ever shows.
* Guide build (`--faces 14000 --tex 512 ...`): `measure_band_frame` falls
  back at 14k faces and the coarse collar rim reads 0.40 all round, which
  left the tunic's neckline 0.1 under the sleeve's root. Builds under half
  the reference size now borrow the 120k mesh's collar measurement
  (`collar_ref` cache stage) for the neckline; see
  `2026-08-22-kente-sleeve-guide-build.png` and the numbers in
  `out/owl_kente/owl-guide-kente-sleeve-wip.report.json`.
* Stage 3 (placket + three gold buttons) not started. Outfit #2 (the wrap)
  later.

## Rebuilt as a drafted, draped tunic (2026-08-22, after the outside review)

An outside review scored the sleeve build above **60/100** and listed
seven faults. All seven follow from one construction choice — `robe.py`
builds the garment as a radial height field whose radius is a running
maximum from the top down — and none of them can be tuned out of it. A
height field cannot fold over a shoulder, and a running maximum cannot
come back in below the widest thing it passes: at the hem the body is
0.704 wide and that cloth was **1.615**, on a body 1.899 tall.

`meshforge/drape.py` (new) replaces it: measure the body, **draft a flat
pattern**, and **drape it** with position-based dynamics. Design,
measurements and the whole argument:
`docs/superpowers/specs/2026-08-22-owl-kente-tunic-design.md`.

| | before (hull) | after (drafted) |
|---|---|---|
| garment width / owl height | 0.851 | **0.638** |
| air per side at y/H 0.15 / 0.25 | 0.456 / 0.348 | **0.157 / 0.113** |
| hem height around the figure | 0.110–0.128 (σ 0.006) | **0.125–0.182 of H (σ 0.015)** |
| tunic triangles | 17,414 | 21,802 (guide: 6,410) |
| min air at rest / inside | 0.008 / 0 | 0.007 / **0** |
| worst through six clips + head probe | 0.004 / 0 | 0.0004 / **7** |
| tunic vertices inside the sleeve | 53 | **0** |

Pictures: `2026-08-22-kente-tunic-before-after.png` (both builds, the
review's eight angles), `2026-08-22-kente-tunic-posed.png` (`wave`,
`tablet_show`, `hop`).

### Commands

```bash
# master (~3 min with the cache); --kente-style tunic is now the default
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-kente-tunic.glb --cache .owl_cache --kente --beads
# guide (14k faces, 1.70 MB)
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-guide-kente-tunic.glb --cache .owl_cache \
    --faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256 --kente --beads
# the old garment is still reachable
python3 -m meshforge.owl_pipeline ... --kente --kente-style robe
python3 -m pytest tests/meshforge -q          # 319 pass, test_drape.py 25 of them
```

### Still open

* **The right side has no cuff.** The folded wing is fused to the flank,
  so a tube cannot pass around it; the cloth covers the shoulder and the
  forearm and tablet come out of a bound slit. A cap sleeve built from
  the wing's cross-sections is the next piece of work.
* The guide build reports 87 vertices inside on the head-follow probe
  (2.3 %, ≤ a few mm, and only during cursor follow) against 4 on the
  master: at 14k faces the decimated body's facets sit inside the hi-res
  surface the check measures against. Raising `neck_margin` moves it a
  little and is not the binding constraint.
* `sleeve root under the tunic` reads 66 of 196 "outside" on the master
  (median 16 mm *under* the cloth). Nothing shows in the renders — these sit
  beside the armscye's cut edge, where the nearest surface sample faces
  sideways (the same measure artefact the sleeve section above records).
* Nothing is promoted. `viewer/assets/owl.glb`, `viewer/assets/owl-guide.glb`
  and the Afromaha-Visit2D guide copy still carry the older garment.

## Gray-cloth diagnosis and the v9 rebuild (2026-08-22, night)

The user supplied `~/Downloads/connected_fabric_research_volume.md` (support ->
tension -> contact -> release -> gravity; ease is not collision offset; openings
need construction, not clearance; gray cloth first). Following its Stage 0,
`tools/garment_diag.py` now reads a *shipped* GLB and produces the evidence:
gray-material turntables through `owl_shots.py`, signed distance per zone,
stretch `l/l0` from the UV chart (which *is* the pattern), boundary loops,
dihedral, silhouette vs the naked body, and every clip through an LBS evaluated
from the GLB itself. Evidence for the 20:10 production is in `out/owl_kente/diag/`
(`inline_findings.md`, `inline_visual.md`, `agent_numeric.json`, the `map_*.png`
false-colour sheets and `renders/sheet_*.png`).

### What the diagnosis found on the 20:10 build (all measured)

1. **The pattern's one seam was never sewn.** `_pairs` did not wrap (its docstring
   said it did), nothing tied column 0 to its copy, and no face straddled them:
   a slit down the centre back, 0.009 at the neckline, **0.151 at the hem**.
   Visible as the red line in every back render.
2. **The rest state was cut for the bare torso.** Chest circumference = mirrored
   torso girth 2.09 x 1.20 = 2.50, while the cloth wraps the folded wing: every
   row was draped 35-75 % over its rest length (weft p90 1.44), 37 % of edges
   over 1.10, and the back/right sat at the solver offset from yoke to skirt
   (86-94 % contact) — a vacuum-formed shell; only the front panel hung free.
3. **The armscye was cut into the neckline** (welded neckline loop spanned v 0-0.37;
   no closed ring over theta -144..-111), and **no tunic cloth lay over the sleeve's
   root** (0 of 196 root-ring vertices covered by a radial ray test; the report's
   `under_tunic` measure is a nearest-sample sign, not a cover test).
4. **Grid-wavelength corrugation** on the free front panel (bend 0.035 on cells
   0.016 x 0.009) and a stiff horizontal brim at the hem.
5. **`tablet_show` drives ~100 tunic vertices into the collar band** (the cloth over
   the folded wing's shoulder rides `wing_right` 0.87 while the collar under it
   rides `neck`); the summary table above only quoted the head-probe "7".

### What changed (`meshforge/drape.py`, tests in `test_drape.py` 29 pass; meshforge suite 323 pass)

* `_pairs` is periodic over the `cols-1` real columns and `seam_columns` welds the
  chart's copy column to column 0 after every projection and after `push_out`.
* `measure_body` adds `wrap_r` (torso envelope + the fused limbs from
  `classify_cells.drape_r`); `draft` cuts every row for the **measured perimeter**
  (`_perimeter`, a real polygon, not 2*pi*mean r) at the widest height any of its
  columns reaches, plus the swing bounds (built before the draft now, via
  `_swing_bounds`), with ease grading 8 % -> 4 % and the hem never below 0.75 of
  the widest row. The rest profile ships in the report (`pattern.profile`) and
  the harness reads it.
* `yoke_keep` 0.12: no opening may cut the yoke; `shoulder_keep` 0.24 over the
  sleeve's own opening, so the second drape lays a shoulder over the sleeve root.
* `bend` 0.035 -> 0.25 and a four-cell `bend_wide` 0.15.
* Tried and removed: handing the shoulder cloth's wing weight back to the chest
  where `tablet_show` drives it into the collar — it oscillated (139/42/62/97
  inside per round) because the wing's shoulder then pierces cloth that stopped
  following it.

### Numbers: production (20:10) -> v9

| | before | after (v9) |
|---|---|---|
| seam gap at the hem | 0.151 | 0 |
| pattern chest / hem circumference | 2.505 / 2.213 | 4.565 / 3.470 (measured wrap 4.27 / 2.43) |
| edge stretch median / p90 / > 1.10 | 1.063 / 1.367 / 37 % | 1.050 / 1.194 / 27 % |
| weft stretch median / p90 | 1.076 / 1.441 | 1.034 / 1.151 |
| contact fraction at y/H 0.40-0.45; back median gap | 87 %; 0.016 | 35 %; 0.041 |
| neckline | open over 33 deg (armscye merged) | closed ring, 152 vertices |
| sleeve root ring under tunic cloth | 0 / 196 | 152 / 196, cloth 0.013-0.023 above |
| vertices inside at rest / hop / wave | 0 / 0 / 0 | 0 / 0 / 0 |
| `tablet_show` worst inside (t = 0.56) | 97-100 | 133-135 (hidden in the collar band; see above) |
| head-follow probe inside | 7 | 22 |
| dihedral median / > 20 deg | 5.6 / 24 % (card-flat) | 11.7 / 33 % (pleated) |

Pictures: `2026-08-22-kente-tunic-v9-before-after.png` (gray, six views + the sleeve
root from behind + the kente look). Full sheets: `out/owl_kente/diag_v9/renders/`.

Build: `out/owl_kente/owl-kente-tunic-v9.glb` (+ `.report.json`), not promoted.
Intermediate builds v2-v8 in the same directory document the path (v2 = seam +
perimeter only: crumpled; v3 = 8 % ease + bend: pleated bell; v5/v6 = two-point
taper: stretched again; v7-v9 = per-height profile).

### Still open

* The skirt is still pleated at a few cells' wavelength: the 152 x 84 grid with
  0.028 x 0.009 cells buckles at its own scale whatever the bend stiffness. The
  research volume's answer is a coarser design mesh for structure and a finer
  one for detail (section 12-13), not more stiffness.
* Cut edges are sawtooth (hem zigzag p90 4 mm; the armscye and hand-slit edges
  are ragged in the zooms). The selvedge hides some of it; a smoothed boundary
  after `cut_grid` would fix the rest.
* `tablet_show` and the head-follow probe still push cloth into the collar band
  (baked folds skinned with LBS cannot both follow the wing and stay out of the
  collar). Re-draping per pose or a corrective shape on that clip is the honest fix.
* The hem rides up to 0.17 over the folded wing's tip (theta 120); the hem stands
  0.1-0.2 off the legs all round, held by the swing bound.
* Nothing promoted; the guide build (`--faces 14000`) has not been rebuilt with v9.

## The kente wrap (2026-08-23, overnight)

The tunic line stopped at v9 (user: "wrinkles added to a shell"). The second research volume named
the cause — class F (skinned from the nearest skin, not from what holds the cloth up) and class I (baked
folds on a rigid carrier) — and the user chose a **wrapped kente tied over the raised (−X) wing**, tablet
shoulder bare, hem above the anklets, measured from the body at build time. Spec
`docs/superpowers/specs/2026-08-23-owl-kente-wrap-design.md`, plan
`docs/superpowers/plans/2026-08-23-owl-kente-wrap.md` (with a deviations section), tickets
`docs/superpowers/tickets/2026-08-23-owl-kente-wrap-tickets.md`.

### What was built

* `meshforge/wrap.py` — `measure_support` (the diagonal support loop read off the body: knot on the
  front edge of the raised wing's root ring under the collar rim, front diagonal to the tablet wing's
  lower contour measured over 12 poses per clip, flank under the fused wing, back diagonal; radius from
  the torso envelope, clamped under the rim), `build_kente_wrap` (sheet hung from the loop with the
  gathered columns crowding into the knot arc, tail strip pinned behind the root ring's back edge,
  procedural knot from `meshforge/knot.py`, one chart per piece), `support_weights` (skin from the
  support graph: pins take the torso's chest/body blend, columns blend to `body` by height; zero weight
  on any other joint).
* `meshforge/drape.py` — `column_angles` (columns spaced by the ring's 3D length × gather),
  `Pattern.theta_cols`, non-periodic strips (`periodic=` on pairs/seam/constraints/tethers/drape),
  `_swing_bounds(..., with_neck=)`.
* `meshforge/gates.py` — G1–G8 as numbers with thresholds, printed by the build and stored in
  `report["gates"]`. `owl_pipeline.py --kente-style wrap` (+ `--wrap-*` flags); no sleeve in wrap mode;
  beads at 80 faces in guide builds (they outweighed the garment).
* `tools/garment_diag.py` grays any `owl_kente*` material, so `--stage renders` works for the wrap.
* Tests: `tests/meshforge/test_wrap.py` (16), `test_knot.py` (3), `test_gates.py` (8), `test_drape.py` +4.

### Numbers (master, `--cloth-res 76,42`, 8.45 MB; guide 1.70 MB)

Support loop: θ_k -65.5°, y_k 0.432 H (rim 0.462), under-arm θ_a 21.2° at
0.327 H, flank to θ_b 146.2° at 0.226 H; root ring 503 vertices over θ
[-140.1, -65.5]. Sheet grid [42, 77], hem [0.142, 0.183] H (beads top 0.102), width by
height {'0.15': 1.068, '0.20': 1.113, '0.25': 0.834, '0.30': 0.724, '0.35': 0.584, '0.40': 0.242, '0.45': 0.083}. Tail [80, 16] pinned at θ [-190.1, -145.1]. Knot {'faces': 2132, 'vertices': 1128, 'strands': 7, 'size': 0.1329, 'arc_length': 0.3062}.

| gate | value | threshold | verdict |
|---|---|---|---|
| G1 worn: pinned row drift from its torso support | 0.0015 | <= 0.010 H | PASS |
| G2 support only: weight on forbidden joints | 0.0 | = 0 | PASS |
| G3 no penetration: cloth vertices inside the body, worst clip | 58 | <= 20 | FAIL |
| G4 hem: above the beads and level | 0.1416 | >= 0.122 H, sigma <= 0.015 H | PASS |
| G5 root covered: ring rays meeting cloth | 0.528 | >= 0.90 | FAIL |
| G6 tail clear of the waving wing | 0 | = 0 inside | PASS |
| G7 stretch: warp/weft l/l0 | 1.038 | median in [0.97, 1.05], p90 <= 1.12 | FAIL |
| G8 collar visible: cloth vertices above the band's rim | 0 | = 0 | PASS |

Per clip inside: { idle: 0, wave: 58, blink: 0, nod: 0, hop: 0, tablet_show: 0, head_follow_probe: 0 }.

### What the pictures say (`out/owl_kente/diag_wrap/LOOK.md`, sheets in `diag_wrap/renders/`)

Right: the diagonal from the shoulder across the chest **under** the tablet, the folded wing lying **over**
the cloth's edge, the collar fully visible, large soft folds (a 76×42 grid; 152×84 ruffled the hem at its
own cell size), the back diagonal, the hanging end; the garment travels with the torso through every
clip. Wrong, in order: the knot is a bundle of pipes (K1's limit; W-K2); the tail is a flat panel; a puffy
fold under the knot; a hem flap behind the folded wing's tip; `wave` pushes 58 sheet vertices into the
raised wing's base at its peak (hidden under the wing; G3 fails on that clip only).

### What changed during the night and why (the findings)

1. The first master was a **box**: pins 0.45 off the body where `wrap_r` merges the wing base into the
   torso, and `draft` widens a whole row for anything any bound holds in it. Fix: loop radius from the
   torso envelope; only the hem's swing bound sizes the pattern; root/neck bounds act in the solve only.
2. Columns spaced by angle tore the top row where the loop dives under the tablet wing (ratio 3.9):
   columns are now spaced by the ring's 3D length.
3. Three samples per clip missed the wave's peaks; the wrap sweeps 12.
4. A straight diagonal crossed the collar's dip at the front (56 vertices above the rim): the loop is
   clamped under `rim − knot_drop`.
5. The pin at 0.012 under the wing's 5th percentile sat in its lowest feathers: 2nd percentile, 0.025.
6. G5 rays from the wing axis pointed into the body for the ring's inner half: rays from the torso
   axis, ring below the tie only. G8 used the coarse band dict: it uses the measured rim now.
7. The hem at ≥ 75 % of the widest row vs 55 %: no visible difference — the hem's cloth is the hips'
   circumference (with the folded wing), the ruffle was pleat scale.

### Commands

```bash
# master (~1.5 min with the cache)
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-kente-wrap.glb --cache .owl_cache --kente --kente-style wrap --beads --cloth-res 76,42
# guide (14k faces, 1.70 MB) — copied to Afromaha-Visit2D/src/renderer/public/models/guide/owl-guide-kente-wrap.glb
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-guide-kente-wrap.glb --cache .owl_cache \
    --faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256 --kente --kente-style wrap --beads
# gray sheets (~4 min) and the Safari viewer
python3 tools/garment_diag.py out/owl_kente/owl-kente-wrap.glb --out out/owl_kente/diag_wrap \
    --report out/owl_kente/owl-kente-wrap.report.json --stage renders
python3 artifact/build.py --glb out/owl_kente/owl-kente-wrap.glb --out out/owl_kente/owl-kente-wrap.html
# kiosk: VITE_GUIDE_MODEL=owl-guide-kente-wrap npm run dev   (or ?guideModel=owl-guide-kente-wrap)
```

### Still open

* W-K2 / a better K1 for the knot; the tail's body (IL-1 is the real answer); the fold under the knot.
* G3 on `wave` (58 under the wing's base), G5 at 53 %, G7 p90 1.15 (the root bound vs the pattern at
  the wing base), G10 at 1.70 MB — all reported, none hidden.
* Nothing promoted: `viewer/assets/owl.glb` and the kiosk's `owl-guide.glb` are untouched; nothing committed.

## The wrap, second night (2026-08-23 02:00–04:00)

Still ticket W-0. Four luna workers ran concurrently (`summon_luna` ×4 — the `summon_luna_workflow*` tools
are not registered in this session): one on the knot, three instrumented gate probes. Two probes hit the
MCP idle cap at 1800 s; G7's subprocess finished its artifacts anyway, G3's was killed and its work done here.
Probe scripts, npz dumps and reports are in `out/owl_kente/probes/`.

### What changed in the code

* `meshforge/wrap.py`
  * `root_mask(mesh, labels, wing, band)` — the ring as a boolean mask (the ring itself now uses it).
  * `measure_support` clears the loop under where the **raised wing's root band sweeps** through the clips,
    blended in outside the knot arc so the tie itself does not move, and limited by `sweep_drop_max` so a
    stray contour cannot collapse the piece. New `WrapParams`: `sweep_band` 0.12, `sweep_blend_deg` 15,
    `sweep_drop_max` 0.08. **G3 58 → 6.**
  * `build_kente_wrap` grew a local `hang(pattern, cloth)` helper (pin the top row, push out, constraints,
    tethers, drape) so a piece can be hung more than once.
  * `WrapParams.taut_cut` (default **off**, see below).
* `meshforge/drape.py`
  * `taut_fall(...)` — each column's cut length as the taut string from its support, over the running maximum
    of everything it must clear, down to the hem, plus the height at 64 stations along that path.
  * `draft(..., taut=False)` uses it, capped by `ClothParams.taut_cap` (0.15), and maps the circumference
    profile through the path instead of subtracting the drop.
* `meshforge/knot.py` — K1 is now a **cinched bunch**: an icosphere squashed in the frame, pinched at the
  waist, fluted, a flat band wrapped round it, two short curled ends, three folds tucked under. 1016 faces
  (was 2132). Rejected on sight first: sausage saucer, trefoil hoop, clenched claw (`knot_k1.png`,
  `knot_k1b.png`, `knot_k1c.png`; the shipped one is `knot_k1d.png`, metrics in `out/owl_kente/knot_metrics.py`).
* `tools/garment_diag.py` — `_zoom_plan(src)` finds the garment's parts in the build and aims the close-ups at
  their own centres (knot from its side / front / above, the wing-base flank, the tail, hem front and back-left).
  Falls back to the tunic-era crops when there is no wrap.
* Tests: +6 (`taut_fall` ×3, `root_mask`, the swept clearance ×2). `tests/meshforge` is **360 passed**.

### What was measured and rejected

* **The taut cut.** Cutting every column to the path it actually travels rather than its vertical drop is the
  physically right answer to the 1.4 warp stretch under the knot — and on this owl it costs more than it buys.
  Uncapped, the hem pools at 0.02 H (the anklets start at 0.10). Capped at +15 % the hem lands at 0.104 H;
  capped at +8 % everything passes but G3 lands on 16 — and at +5 % it is 56. Off by default; the code and its
  cap stay for a body that needs them.
* **A fit pass** (cut, hang at 2/3 iterations, re-cut each column by where its hem landed, hang again). It does
  not converge on this body: hem sigma 0.052 → 0.047 over two passes, the stretch unmoved, G3 wandering 17–70.
  Removed. The reason it cannot work as written: a column of a *gathered* piece does not fall in its own
  vertical plane, so its length and its neighbours' are coupled through the drape.
* **The root radial bound** is what drove the old G3: it holds the cloth out at the swept root radius, and the
  wing then rotates into the cloth it pushed out. Turning it off alone takes G3 58 → 20. The swept-contour
  clearance does better (→ 6) and keeps the bound.

### The two open gates

* **G5 root covered 0.416** (was 0.528). The bare vertices are the *upper half* of the raised wing's root ring;
  the loop's back diagonal tops out at the tie. 202/202 are visible from at least one turntable view but almost
  only from behind (yaw 180: 163, yaw 225: 146, front: 32/0/0), and what shows there is the wing's own underside.
  Covering that root and clearing its sweep are the same lever pulled opposite ways. **Ticket W-G5** — a
  decision, not a bug.
* **G7 stretch p90 1.16.** Warp, not weft: columns 12–19 (θ −115°…−73°, the back diagonal and the knot arc)
  reach 1.35–1.5. Removing every radial bound takes p90 to 1.121, so the bounds are most of it; the knot-arc
  columns keep ~1.35 regardless, because cloth fanning out of a gather does not hang in vertical planes.
  Extra cut length does not fix it (see above).
* **G3's statistic** is the wrong one — ticket **W-G3**: the count swings 6 ↔ 67 on parameter changes that alter
  nothing visible (master 6 / guide 57 with identical code). Depth, and whether any of it is unoccluded, is
  what the gate should report.

### Commands (unchanged) and outputs

Master `out/owl_kente/owl-kente-wrap.glb` (8.42 MB, 6/8), guide `owl-guide-kente-wrap.glb` (1.68 MB, copied to
the kiosk), sheets `out/owl_kente/diag_wrap2/` with `LOOK.md`, viewer `out/owl_kente/owl-kente-wrap.html`.
The first night's build survives as `diag_wrap/` + `build-wrap-v1.log` + `owl-kente-wrap-v1.report.json`.

### Look changes adopted after judging the renders

* `WrapParams.knot_size` 0.07 → **0.095**: at 0.07 the knot was invisible on the guide turntable at kiosk scale.
* `WrapParams.tail_width_frac` 0.22 → **0.15**, plus `tail_taper` 0.62, `tail_cut_frac` 0.18, `tail_curl_deg` 22:
  the tail read as a bib panel hanging on the back; narrowing it is what turned it into a sash end (the taper
  and the diagonal cut help less than the width did). `out/owl_kente/probes/sheet_tail_before_after.png`.
* Gates are unchanged by both (master 6/8: G3 6, G4 0.1427 sigma 0.0075, G5 0.419, G7 p90 1.16).
