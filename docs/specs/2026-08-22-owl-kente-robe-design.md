# AI-CCORE owl — the kente robe (a garment, not a decal)

**Date:** 2026-08-22
**Track:** design (replaces the *vest* and *sash* decals of
`2026-08-21-owl-kente-textile-design.md` as the way the owl wears kente;
the weave generator and the decal code from that spec stay and are reused)
**Builds on:** the OwlV1 rig (`2026-08-21-owl-full-body-rig-design.md`),
`meshforge.textile` (the procedural weave), `meshforge.kente.bake_weave_maps`
(position-baked weave + tangent-space relief, now shared).

## Problem

The first kente pass painted the weave onto a copy of the belly's own
triangles, 6 mm off the skin (`kente.build_kente_decal`). Rendered, that is
body paint: it follows every bump of the belly, ends at the region map's
triangle-by-triangle label boundary, and hugs the body like a wetsuit. The
standard it had to meet and did not: **clothing is ample; there is air
between the subject and the cloth; kente in particular is a big, loose
cloth that usually reads as a robe.** The previous spec's own "left open"
list names the fix — "a real garment *mesh* ... would read better, at
real cost" — and that is what this is.

A second, smaller problem found on the way in: `--kente` (the vest) did
not actually build. `validate_rig` rejected the vest primitive on the real
owl (`normals unit length — range 0.000..1.000`): the folded-cross-section
defect that the sash had already diagnosed and repaired with
`_repair_zero_normals` was never applied to the vest. Fixed in passing;
the vest and sash remain available as `--kente-style vest|sash`.

## What a worn cloth does (the model behind the geometry)

Three facts about hanging cloth, each of which the decal violates and the
robe is built from:

1. **It hangs from where it is held.** Here that is just under the collar
   at the front and back; under the raised wing on the −X side (the cloth
   passes *under* the wing that waves, the way a wrapper passes under an
   arm); and over the folded wing on the +X side (the wing is pressed
   against the body — one solid with it on the welded mesh — so the cloth
   falls over it like a cloak over an arm, and the hand with the tablet
   comes out in front).
2. **It falls vertically past the widest point.** Below the belly's
   widest ring the cloth does not follow the body back in toward the legs;
   it hangs straight down from that ring. Geometrically: the cloth's radius
   at any height is the *cumulative maximum from the top edge down* of the
   body's radius (`robe.gravity_hull`), not the body's radius at that
   height. This single rule is what makes the garment ample and is why the
   legs end up inside a bell of air.
3. **It has slack that grows with the fall.** Snug where it is held
   (`clearance_top` 0.02, ~1 % of the owl's height), a hand's width at the
   hem (`clearance_hem` 0.10, ~5 %), with extra slack over anything it is
   draped across that moves under it (`drape_clearance` 0.05 — the folded
   wing swings its shoulder out by ~0.05 on `tablet_show`). Vertical pleats
   (`pleats` 10, `pleat_depth` 0.035 at the hem) deepen with the fall and
   undulate the hem.

## Architecture (as built)

```
meshforge/robe.py
    RobeParams           every length in mesh units or *_frac of the bbox height
    radial_scan          one ray per (theta, height) cell from a vertical axis on the
                         body's symmetry plane; every surface crossing, nearest first,
                         with the region label of the face it came from
    torso_envelope       the body's own radius per cell: first crossing if labelled
                         torso, taken from the −X half and mirrored (+X is a solid block
                         of folded wing + tablet + mislabelled leg; the body under it
                         cannot be measured but it is the same body)
    classify_cells       what the first crossing is — torso / drape (fused: wing_right,
                         tail, and anything torso-labelled but far outside the mirrored
                         body) / stop (wing_left) — plus every later solid as a radial
                         interval the cloth may pass behind but not through
    top_edge             the collar's measured lower rim minus a gap, lowered under every
                         stop cell and under every held-out solid the cloth would cut
                         through, smoothed (erode → gaussian → min with the raw
                         constraint), iterated to a fixed point
    gravity_hull         cumulative max, from each column's own top edge down, of
                         max(mirrored envelope, what the ray actually met)
    build_kente_robe     hull + clearance + pleats → (theta, t) grid, seam column
                         duplicated for one rectangular UV chart, normals from the
                         welded grid, weave baked by position (kente.bake_weave_maps)
    transfer_body_weights  dense skin weights from the k nearest chest/body/neck
                         vertices, restricted to the torso joints
    clearance / posed_clearance   the standard, measured: distance to the body surface
                         and vertices-inside count, at rest and through every clip
meshforge/owl_pipeline.py   --kente now builds the robe (--kente-style robe|vest|sash),
                         scales its grid/chart with --faces/--tex, logs the clearance
                         report into <out>.report.json
meshforge/bake.py        rasterize_face_ids / bake_position_map accept (width, height)
meshforge/kente.py       bake_weave_maps shared by vest, sash and robe; the vest repairs
                         zero-length normals like the sash already did
```

## Measured on the owl (1.37 × 1.90 × 1.18)

The top edge, as bbox-height fractions: 0.36 at the front (the bib's lower
rim is 0.382); 0.28 across the −X side (the raised wing's lowest fused row
is 0.31; the margin is `obstacle_margin_frac` 0.03); 0.25 in a notch at
+35..+55° where the hand and tablet come out (the forearm crosses exactly
where the cloth climbs from the belly onto the folded wing, so the cloth
must dip under it); 0.48–0.50 over the folded wing and across the back
(the collar is at 0.52 there). Hem at 0.12 — above the feet (0.077) and
the anklet beads (0.093).

| Build | Robe | GLB | Clearance at rest (min / p05 / median) | Worst through clips |
|---|---|---|---|---|
| master (120k faces, 2048 tex) | 18,144 tris, 2048×512 PNG | 8.72 MB (7.75 without) | 0.015 / 0.029 / 0.116 | 0.008 (hop, the hem beside the folded wing's tip as the leg it is mislabelled into tucks), 0 vertices inside |
| guide (14k faces, 512 tex) | 2,208 tris, 1024×256 PNG | 1.55 MB (1.14 without) | 0.014 / 0.035 / 0.118 | 0.015 (idle), 0 vertices inside |

"Clearance" is the distance to the nearest of the 3 M hi-res surface
samples (rest) or to 150 k samples of the posed body (clips). "Inside" is
a vertex behind the outward normal of the face its nearest surface sample
lies on, *confirmed* by ray parity (`trimesh.contains`) — the two tests
fail in different places (a concave crease; the mesh's few boundary
edges), so a vertex counts only when both agree; the unconfirmed suspects
are reported too. Both are in the build log and in `<out>.report.json`
under `robe_clearance`.

Pixel verification (real Three.js, `tools/owl_shots.py`): five rest views,
plus `wave=0.55`, `hop=0.4`, `tablet_show=0.5..0.7`. Images next to the
handoff note: `2026-08-22-kente-before-after.png`, `-kente-robe-views.png`,
`-kente-robe-posed.png`, `-kente-robe-guide-build.png`, and
`-kente-robe-parameter-map.png` (the (theta, height) cell map: torso /
drape / stop / held, with the raw collar cap and the final top edge).

## Bugs found on the way (all on the real owl, none visible on the synthetic test body)

1. **Label contamination on the +X side.** `regions.classify`'s crop boxes
   stop at 0.25 of the height, so the folded wing's tip below that is
   `leg_right`/`foot_right`; the legs are not even symmetric (x −0.03 and
   +0.39). A torso envelope that trusted the labels put the cloth through
   the wing. Fixed by mirroring the −X half and treating any
   torso-labelled surface more than `foreign_tolerance` (0.08) outside the
   mirrored body as an appendage the cloth drapes over.
2. **The mirrored envelope is a floor, not the truth.** The body is up to
   0.07 wider on the +X side of the back; 76 top-edge vertices sat inside
   the chest until the hull took the maximum of the envelope and what each
   ray actually met.
3. **Grazing crossings.** Rays skimming the belly's embossed circuit trace
   exit and re-enter the same solid a millimetre apart; treated as
   held-out solids, they punched two dips into the front panel. Crossings
   closer than `surface_fuzz` (0.02) are one solid.
4. **Cumulative erosion.** Re-smoothing the already-smoothed top edge on
   every fixed-point iteration eroded the front plateau from the collar to
   the hip in six passes. The smoothing now runs from the raw constraint
   with an accumulated block height.
5. **A hem that followed the wing.** Body vertices next to the wing seam
   carry blended `wing_right` weight; a hem that inherited it followed the
   folded wing into the (static, mislabelled) wing tip on `tablet_show` —
   14 vertices. Garment weights are restricted to `body`/`chest`/`neck`.
6. **The folded wing's shoulder moves.** With the cloth snug over the
   wing, `tablet_show` swung 56 top-edge vertices into the shoulder.
   `drape_clearance` adds slack wherever the hull comes from a draped-over
   surface.

7. **Two centres.** `regions.measure_collar` describes the rim per angle
   around the collar's own centre (in bbox fractions); the scan measures
   angles around the body's symmetry axis, 0.12 deeper. Reading the cap at
   the scan angle shifted it by a few degrees of rim. `_collar_cap` now
   converts each scan ray's collar-height hit into the collar's own angle.
8. **The inside test was not geometric.** The first `clearance` compared a
   vertex against the nearest body *vertex*'s averaged normal; on a coarse
   mesh that is a corner normal pointing diagonally away, and a point
   hovering outside a face read as inside (4 of 5 on a unit box). Replaced
   by the nearest dense *surface sample* and its face normal, confirmed by
   ray parity — which in turn caught the sample test's own false positive
   in the crease between the tucked leg and the rump on the hop.
9. **What was tested was not what shipped.** The blocking test ran on the
   cloth radius without pleats or drape slack, and the posed check ran on
   the dense transferred weights rather than the top-4 set written to the
   GLB. Both now use what the final geometry and file actually carry.
10. **A margin measured from the wrong row.** An obstacle first seen on a
    scan row may start anywhere in the interval below it; on the coarse
    synthetic test body the cloth ended 4 mm under the fin. The block
    height now subtracts one row spacing as well as the margin.

(7–10 came out of an adversarial review by a luna worker after the first
complete build; its other notes — parameter validation, a fixed-point
iteration cap that reported nothing, `strips_around` having to be a
multiple of `strip_cycle` for the weave to meet itself at the seam,
`retopo.bake_texture` assuming a square map — are fixed too.)

The pattern is the same as the previous spec's: each of these only shows
up with the real asset and a real check (the clearance numbers and the
renders), never with the synthetic geometry the unit tests run on — so the
pipeline measures the garment it just built and prints the result every
time.

## Verification

`tests/meshforge/test_robe.py` (24): the scan's crossing order and labels;
the mirrored envelope excludes the fused bump; drape vs stop
classification; the robe stands off the body everywhere with zero vertices
inside; the hem hangs outside the torso's widest ring (not the legs); the
cloth ends under the fin and climbs over the bulge; primitive validity
(indices, UVs in [0, 1], unit outward normals, double-sided material,
texture size, seam closed and seamlessly shaded); colorway-only texels;
optional normal map; more clearance ⇒ a measurably wider robe; weight
transfer (blend, allowed joints, empty pool); clearance's inside count;
nearest-keyframe posing; posed clearance moving garment and body together;
and, from the review: the inside test on a unit box, surface samples
following a posed body, a held-out block the cloth must dip under versus
one far enough away to hang behind, convergence reporting, parameter
validation, and `torso_labels` being honoured. `test_kente.py` (24) and
`test_bake.py` (17) still pass after the refactors. Full `tests/meshforge`
suite: see the handoff note.

## Left open

* **The guide build's collar cap.** At 14k faces `collar.measure_band_frame`
  falls back (`bins_used: 5`) and the coarse collar rim is lower at the
  back, so the guide robe's back panel tops out at 0.39 instead of 0.50.
  Pre-existing; it only moves where the robe stops, not whether it clears.
* **Not promoted.** `viewer/assets/owl.glb` and `owl-guide.glb` are
  untouched; the robed builds are in `out/owl_kente/` (gitignored). Promote
  with the commands in the handoff note once the look is signed off.
* **Relief (normal map).** `--kente-normal` bakes the woven relief for the
  robe exactly as for the decals; off by default. The green-channel sign
  was checked against the shipped renderer by a luna worker reading
  `tools/three_bundle.js` (and re-grepped here): Three.js r160's shader
  bitangent follows increasing v and its GLTFLoader negates
  `normalScale.y` when there is no TANGENT attribute, so the correct
  encoding is B = *decreasing* v (the glTF spec's "+Y is up").
  `kente._face_tangent_basis` used to return `cross(N, T)`, which is that
  direction only on a mirrored chart (the robe's is; xatlas charts may not
  be) — it now picks the sign per face from the UV winding. What remains
  unverified is the *look* of the relief on the owl under `--kente-normal`;
  nobody has rendered it yet.
* **Colorway / pleat / hem parameters** are exposed (`--kente-colorway`,
  `--robe-hem`, `--robe-clearance`, `--robe-pleats`, `--robe-res`,
  `--robe-tex`) but only the defaults have been looked at.

## Addendum (2026-08-22, evening): sleeves, and both shoulders covered

Superseding the "armhole around the raised wing" of the stage-1 tunic:

1. Both wings' fused roots are **draped over** like the tail; a drape-over
   solid starting within `drape_merge_gap` (0.13) of the first solid is
   the same shoulder and is draped over too (the folded wing's upper arm
   pressed to the chest). Measured from the first solid only; held solids
   of one label on one ray are one obstacle (palm + tablet).
2. **`meshforge/sleeve.py`** wraps the raised wing's free part from the
   wing's own cross-sections (principal axis from the `wing_left` pivot;
   slices + rays from the section centroid; radius = min(wing + clearance,
   body − margin); clearance root 0.03 → wide 0.06 → cuff 0.025; cut where
   fused; cuff at `length_frac` 0.72 of the free wing). One chart, strips
   along the wing, gold cuff. Weights from the nearest wing/chest skin,
   smoothed over the grid.
3. The tunic is cut around the sleeve where the sleeve stands outside the
   tunic's actual surface (`obstacle_points`); every cut is decided against
   the real cloth radius ± `obstacle_near`; holes are signed-distance
   fields of the dilated mask (`HoleField`).
4. Measured per build: clearance of both garments at rest and through the
   clips + a head-follow probe; the sleeve root under the tunic
   (`garment_overlap`, `posed_overlap`); tunic vertices inside the sleeve.

Bugs this round, each caught by a number or a render, never by a test
passing: the sleeve's cut cells clamped to the far exit of the body and
dragged neighbouring vertices inside the egg (cut cells now take the
smallest kept radius around them); grid columns at cell edges averaged a
capped cell with an uncapped one (columns now sit on the scan's rays); the
star-curve hole inflated the diagonal slit to the neckline (fields); the
conservative slit band cut where the hand was 0.09 off the cloth (actual
radius); chained drape merging swallowed the tablet (first solid only);
the tunic cut around the bare wing while the fatter sleeve passed through
it (sleeve vertices as obstacles); vertex-based separation on a coarse fin
(surface samples).
