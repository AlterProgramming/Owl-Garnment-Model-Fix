# The kente tunic: a drafted pattern, draped

*2026-08-22. Supersedes the garment half of
`2026-08-22-owl-kente-robe-design.md`; the measurement half of that
document (radial scan, torso envelope, neckline, clearance harness) is
still what this is built on.*

## Why the robe had to be rebuilt, not tuned

An outside review scored the previous garment **60/100** and listed seven
faults: neckline and shoulders too wide, sleeves attached on top of the
body rather than growing from it, a flat rectangular back, a stiff
straight hem, no drape or gravity, a pattern that stays flat over
curvature, and unbalanced volume.

Every one of them follows from a single construction choice, and no
value of `RobeParams` removes any of them. `meshforge/robe.py` builds the
garment as a **radial height field** `r(θ, y)` on a cylindrical grid,
where `gravity_hull` takes `np.maximum.accumulate` of the body's radius
*from the top edge downward*.

1. **A height field cannot have a shoulder.** It is single-valued in `r`;
   cloth lying roughly horizontally over the top of a shoulder is a
   vertical cliff in that map. The garment can only ever stop in a rim,
   so a sleeve can only ever butt against it.
2. **A running maximum cannot come back in.** It inherits the widest
   thing each ray passes — on this owl the folded wing and the held-out
   tablet, at 0.44–0.48 of the height — and carries that width to the
   hem. Measured on the last robe build: at the hem the body is 0.704
   wide and the cloth **1.615**, i.e. 0.456 of air per side on a body
   1.899 tall. A lampshade.

The clearance report read "0 vertices inside, 0.10 of air at the hem"
throughout. It was measuring the right thing and the garment was still
wrong — a reminder that a passing check is not a look.

## What replaced it

`meshforge/drape.py` builds the same garment the way a garment is built:
a flat pattern, cut, and hung on the body under gravity.

### measure_body

`robe`'s scan is reused unchanged — it was the part of that module that
was right. Per `(angle, height)` cell: the torso's own radius
(`torso_envelope`, mirrored from the −X half), what each ray met
(`classify_cells`), and the neckline curve under the collar band's
measured lower rim (`_neckline`). The neckline's *radius* is the larger
of the mirrored envelope and the first crossing actually measured, so
the ring the whole garment hangs from never starts inside a shoulder the
mirror did not know about.

### draft

A flat piece, periodic around, with two shapings a hull cannot express:

* **It opens.** The circumference runs from the neckline's own
  (measured: 2.78 on the owl, including its dip at the front) to the
  widest **torso** girth plus `chest_ease` over a short yoke — the
  widest torso girth, not the widest anything, which is the bug above.
* **It comes back in.** Below the yoke the circumference tapers to
  `hem_ease` over the same girth. Cloth cut straight from the chest to
  the hem hangs as a lampshade whatever the physics says: there is
  nothing under it holding that width, so the width has to be cut out of
  the pattern. That taper is what a side seam does.
* **Every column is cut to its own fall.** The collar rim dips to 0.377
  of the height at the front (under the bib) and rises to 0.517 at the
  back. A piece cut to one length reaches the floor at the front —
  measured, it did. Shaping the pattern is what makes a hem level.

Surplus lives in the circumference, never in the length: cloth hanging
free below the widest point has nothing to absorb extra length, so a
long cut is a long garment while a wide cut is a folded one.

### drape

Position-based dynamics on the pattern's own grid.

* **Constraints**: warp, weft, shear and bend, every rest length read
  from the two vertices' *pattern* coordinates — so a vertex the cut
  snapped onto an opening's edge carries the right amount of cloth.
* **Tethers** (long-range attachments, Kim et al. 2012): every vertex is
  kept within its own pattern distance of the neckline vertex it hangs
  from. Without them a pinned sheet falls: with a Jacobi solve tension
  propagates about one row per iteration, so an 84-row skirt drops for
  twenty steps before it learns it is attached. The first build fell to
  y/H = −2.3.
* **Collision** against dense surface samples with their face normals,
  with the offset **graded down the cloth** — `collide_offset` at the
  neckline where the garment is held, `collide_offset_hem` at the hem
  where it hangs free.
* **A radial bound** for what swings. Colliding against a *union of
  posed meshes* does not work: a sample belonging to one pose sits
  inside another pose's shell, its normal points into the union, and the
  cloth shreds — it did, visibly, on the first attempt. A scalar bound
  ("no cloth closer to the axis than this, here") has no such ambiguity.
  Built from the legs, feet, tail and wings over the `hop`, `idle`,
  `wave`, `tablet_show` and head-follow poses.
* **push_out** at the end, over all kept vertices including the pinned
  neckline, choosing the push direction from the 8 nearest samples
  rather than the closest one — in a crease the nearest sample's normal
  points along the crease and a point never gets out.

### openings

Cut in **pattern** space, not in space: the edge then follows the grain
and stays put when the cloth moves.

The test for what to cut around is not distance from the torso. The
folded wing's upper arm is 0.1 from the chest and cloth lies on it,
because there is no air under it; a hand held out in front has air
behind it, so cloth goes behind it. `held_out_vertices` reads that off
the scan `classify_cells` already did: a wing vertex beyond the fused
surface's radius is held out. Two passes — drape once ignoring what is
held out, cut where it came through, drape again warm-started.

A limb already wearing a sleeve is not cut around twice: held-out
vertices within `sleeve_covers` of the sleeve are dropped, and the
sleeve itself becomes both an obstacle for the cut and a collider for
the cloth, so the tunic lies *on* the sleeve's root instead of through
it.

### the chart

The UV chart **is** the pattern: `u` around the piece, `v` down it. The
weave is painted by chart position, so the warp runs along the grain,
the weft around, and both bend wherever the cloth bends — which is what
"pattern conformity" means and what a position-baked cylindrical chart
could not do. `blocks_tall` defaults to whatever keeps a strip pixel the
same length of cloth across the piece as down it. Gold selvedge along
every cut edge — neckline, hem, armhole, hand slit — read from the same
signed-distance field the cut used.

## Numbers (master build, `--kente --beads`)

| | before (hull) | after (drafted) |
|---|---|---|
| garment width / owl height | 0.851 | **0.638** |
| air per side at y/H 0.15 / 0.25 | 0.456 / 0.348 | **0.157 / 0.113** |
| hem height around the figure | 0.110–0.128 (σ 0.006) | **0.125–0.182 of H (σ 0.015)** |
| triangles (tunic) | 17,414 | 21,802 (guide 6,410) |
| min air at rest / vertices inside | 0.008 / 0 | **0.007 / 0** |
| worst through the six clips + head probe | 0.004 / 0 | 0.0004 / **7** |
| tunic vertices inside the sleeve | 53 | **0** |
| sleeve root ring left uncovered | 145 of 196 | **66 of 196**, median 16 mm under |

The seven remaining vertices are on the head-follow probe (neck yaw ±25°,
a runtime cursor behaviour, not a baked clip) at 0.06 % of the garment.

## Three defects the pictures found and the numbers did not

Each of these passed every clearance check and looked wrong on screen.

* **The selvedge was four times its width.** The opening's distance field
  was counted in *cells* and scaled by the vertical cell, but the cells
  are 16 mm around against 9 mm down; and the field itself was smoothed,
  which flattens its gradient near curvature. Both are fixed by giving
  `distance_transform_edt` its `sampling` and by rounding the *mask*
  instead of the field.
* **Offcuts.** A cut that isolates a patch leaves a fragment held by
  nothing but its tethers; it swings free and, being inside the selvedge
  everywhere, renders as a gold sail over the tablet. `drop_offcuts`
  swallows anything under `min_panel_fraction` of the cloth, and
  `trim_slivers` anything narrower than `min_panel_width`.
* **The armhole was in the wrong place.** Cutting where a held-out limb is
  *near* the cloth is right; cutting where it stands *in front of* the
  cloth is not — an arm held across the chest wants cloth behind it, and
  a hole the size of the arm leaves the chest bare. `open_reach` is the
  distance at which a limb counts as coming through rather than passing
  in front.

A fourth suspected defect was not one: a gold "sail" under the tablet
that survived every fix turned out to be a gold block of the *weave*,
foreshortened at a grazing camera. Flat-shading the garment settled it in
one render — the same lesson as [[flat-shade-to-attribute-mesh-damage]].

## Parameters worth knowing

`ClothParams` — `chest_ease` 0.20 and `hem_ease` 0.06 set the volume;
`length_slack` 0.02 (surplus belongs in the circumference); `yoke_frac`
0.075; `collide_offset` 0.014 → `collide_offset_hem` 0.022;
`swing_margin` 0.018; `edge_band` 0.018; `strips_around` 24 (a multiple
of the weave's `strip_cycle`); grid 152 × 84, chart 2048 × 1024. The
guide build scales the grid and chart with `target_faces`.

CLI: `--kente-style tunic` (the default), with `--cloth-hem`,
`--cloth-ease`, `--cloth-hem-ease`, `--cloth-res`, `--cloth-tex`,
`--cloth-band`, `--cloth-iterations`, `--cloth-strips`. The old garment
is still reachable as `--kente-style robe`, and the decals as `vest` /
`sash`.

## Left deliberately

* **The right side has no cuff.** The folded wing is fused to the flank,
  so a tube cannot pass around it; the cloth covers the shoulder and the
  forearm and tablet come out of a bound slit. A short cap sleeve built
  from the wing's own cross-sections would read as a second sleeve.
* **Self-collision is off** (`self_collide` 0). Nothing visibly
  interpenetrates at these ease values; a fuller cut would need it.
* **Folds are baked at rest** and skinned with the body. Re-draping per
  pose is the next order of realism and is not worth its cost here.
* Nothing is promoted: `viewer/assets/owl.glb` and the Afromaha guide
  copy still carry the hull garment.
