# The kente wrap: cloth hung from a measured support loop, tied over the raised wing

*2026-08-23. Approved for planning 2026-08-23. Supersedes the garment half of
`2026-08-22-owl-kente-tunic-design.md`; the measurement layer (radial scan,
torso envelope, `measure_body`, colliders, radial bounds, the chart-as-pattern
bake) and the harness `tools/garment_diag.py` carry over unchanged.*

Demonstration: **2026-08-26**. Everything below is scoped to that date; what is
not demo-critical is named as such.

## 1. Why a wrap, and why now

The v9 tunic fixed four measured defects (seam, pattern cut for the bare torso,
armscye into the neckline, uncovered sleeve root) and still reads as *wrinkles
added to a shell*. The second research volume names the cause precisely:

* **Class F — wrong animated baseline.** The tunic is skinned from the nearest
  body vertices, not from what holds it up. Its hem follows the torso while the
  legs and the folded wing move under it; the collar turns inside a neckline
  that was kept off the `neck` joint.
* **Class I — folds on an independent carrier.** Every fold is baked at rest and
  convected rigidly; nothing the owl does causes any of them.

No tunic parameter removes either. The wrap is the pivot the user chose: a
single cloth tied over the **raised (−X, `wing_left`) shoulder**, the
**tablet (+X, `wing_right`) shoulder and arm bare**, hem above the ankle beads.
Target look: `out/owl_kente/concepts/target-wrap-front.png` — note the image is
**mirrored** relative to the owl (its raised wing is on the viewer's right; ours
is −X). `target-wrap-back-v2.png` is the back; `target-wrap-right-side.png` the
bare side.

Two constraints from the user shape everything: *strong enough* for the demo
distance, and *not coupled to the model* — the owl underneath will change
before deploy, so the garment is rebuilt from measurements, never from stored
coordinates.

## 2. The garment

Three parts, one chart, one material:

| part | what it is | held by |
|---|---|---|
| **sheet** | the wrapped cloth: a tube below a *diagonal* support loop — high at the raised shoulder, low under the tablet arm, rising again across the back | pinned along the support loop; falls to the hem |
| **knot** | a designed bunch on the front face of the raised wing's root, just under the collar rim; the sheet's columns gather into its base | rigid to `chest` |
| **tail** | the free end: a strip hanging down the back, behind the raised wing | pinned on the back, `tail_offset_deg` behind the root ring's back edge, so it hangs clear of the wing's base (pinned at the knot's back end it hung through the base and the wave swept it); falls free |

The chest bib and the collar band stay fully visible above the loop; the
folded tablet wing lies *over* the sheet's top edge at the +X flank, as in the
reference. There is **no sleeve**: the knot and tail cover the wing root, which
is what the sleeve was for.

## 3. Measurement — what the build reads off the body

All in fractions of the bbox height `H` (1.899 on the current owl, `y0` −0.950)
and the torso angle θ (0° = +Z face, +90° = +X tablet side, −90° = −X raised
side). Measured on the v9 build for reference; the build re-measures:

| quantity | source | current value |
|---|---|---|
| collar lower rim `rim(θ)` | `drape._neckline` | 0.377 front → 0.517 back |
| raised wing root ring | `wing_left` vertices with blended weights | centroid θ −121°, y 0.441; ring spans θ −156..−102, y 0.31..0.51 |
| raised wing pivot | `wing_left` joint | θ −127°, y 0.409 |
| tablet wing lower boundary `low(θ)` | p05 of `wing_right` y per 15° bin | 0.238 at θ 45–60 (under the tablet); 0.205–0.227 on the flank θ 75–150 |
| tablet wing span | `wing_right` region | θ 38..158 |
| hip / chest | `body`, `chest` joints | y 0.215 / 0.300 |
| ankle beads | `beads` report `world_y` + bead radius | ring top y ≈ 0.102 |
| leg / foot sweep | `_swing_bounds` over hop, idle, wave, tablet_show + head probe | existing |

**Support loop `L(θ)`** (closed, one value per column of the sheet):

* **knot anchor** `θ_k` = front edge of the root ring (measured −86° on the
  14k guide mesh, −66° on the 120k master — the ring's 95th-percentile
  half-width depends on resolution) ; `y_k` = `rim(θ_k) − knot_drop` (0.03:
  at 0.015 the turning collar band reached it) — the knot tucks under the
  band, the band is never covered. The whole loop is clamped under
  `rim(θ) − knot_drop`: a straight diagonal crossed the band's dip at the
  front (56 vertices above the rim, measured).
* **knot base arc**: `θ_k ± gather_half` (25°) at `y_k`; the knot mesh sits on
  it; sheet columns inside the arc gather into it.
* **front diagonal**: straight in (θ, y) from the arc's front end to the
  **under-arm point** `(θ_a, low(θ_a) − arm_margin)` with `θ_a` the front
  edge of the `wing_right` region (≈ 38°, where the fused wing starts) and
  `arm_margin` 0.025 (at 0.012 the pin sat in the wing's lowest feathers and
  tore the top row, ratio 3.9). `low(θ)` is the 2nd percentile of the wing's
  height per cell **over the posed bodies too** (12 samples per clip): the
  wing dips behind its pivot when the tablet comes up.
* **flank**: `low(θ) − arm_margin` from `θ_a` to the wing's back edge `θ_b`
  (≈ 150°) — the cloth edge is wherever the fused wing stops, so it follows
  the wing's lower contour down from ≈ 0.27 at the front to ≈ 0.20 at the
  back. `low(θ)` is taken per column and smoothed (σ = 1 column) so the loop
  has no kinks.
* **back diagonal**: straight from `(θ_b, L(θ_b))` through θ = 180° to the arc's
  back end at `(θ_k − gather_half, y_k)`. Its last ~35° run over the root
  ring's back half, covering it.

**Hem**: `hem_y` = 0.135 (≥ bead ring top + 0.02; 0.125 landed 0.003 under
the floor), the same target everywhere — per-column length is cut from
`L(θ)` to `hem_y`, so the hem is level.

**Loop radius**: the *torso envelope* at `(θ, L(θ))` — never `wrap_r`, which
merges the raised wing's base into the body and put the pins 0.45 off the
body there (a shelf that `draft` then turned into a box: it cuts every row
for the widest thing any bound holds in it). The pins are pushed out of the
whole mesh before anything hangs, so where the root is solid they lie on it.
The cloth passes *under* the raised wing, against the body.

**Knot frame**: origin on the body surface at `(θ_k, y_k)`, normal = surface
normal there, up = the rim's tangent. Everything about the knot is placed in
this frame; nothing about it is a stored coordinate.

Parameters (`WrapParams`, new): `knot_drop`, `gather_half`, `gather_ratio`
(cloth width per arc length inside the base arc, 1.3), `arm_margin`, `hem_y`,
`tail_width` (0.22), `tail_length_to` (0.15, hem level + 0.025), `ease` (0.08
chest, 0.04 hem), bend/stretch/tether values inherited from `ClothParams`.

## 4. Construction (`meshforge/drape.py`, wrap mode)

The tunic machinery is reused with one generalisation: **the pinned top row is
the support loop instead of the neckline.**

1. `measure_body` as today, plus `low(θ)` and the root ring → a `Support`
   (θ, y, pin weight per column).
2. `draft(body, params, support)`: circumference per row from the measured
   perimeter + ease (existing), **length per column from `L(θ)` to `hem_y`**
   (existing per-column mechanism; the neckline was a special case). Columns
   under the base arc are drafted `gather_ratio` wider than the arc they pin
   to — surplus folds radiate from the knot; that is where the reference's
   folds come from, not from noise.
3. `drape` exactly as today: warp/weft/shear/bend constraints from pattern
   coordinates, tethers to the pinned row, `Collider` on the body (bind pose,
   no sleeve), `RadialBound` for legs/feet/tail/raised wing over the clips,
   seam weld, `push_out`. `bend` stays 0.25 so folds are pattern-scale, not
   grid-scale (the v3 lesson).
4. **tail**: a second, non-periodic grid (≈ 30 × 80), top row pinned on the
   back just behind the root ring's back edge (`tail_offset_deg`), `tail_width`
   wide, draped with the same colliders and bounds so it hangs behind the
   wing. Tethers to its own top row. *(Amended 2026-08-23: pinned at the
   knot's back end it hung through the wing's base.)*
5. **knot** — two implementations behind one interface
   `KnotPart(mesh, base_arc, frame)`:
   * **K1 procedural** (demo baseline): seven twisted tubes swept from the base
     arc into a flattened bulge (≈ 0.10 × 0.07 × 0.06 H) with two short
     leaf-shaped ends; ≈ 1.5k triangles; textured from the kente chart with a
     gold selvedge.
   * **K2 TRELLIS** (upgrade, day 1, optional): one reference image of a tied
     kente bunch on a plain ground → AI-GATEWAY `trellis-2` → raw GLB →
     `meshforge.clean` (weld, debris) → decimate ≤ 3k → orient (PCA + named
     axis) → scale base width to the measured arc → place in the knot frame →
     **retexture** (project the kente chart; TRELLIS' own albedo is the
     fallback). This is the "serious post-processing"; it is scoped as
     infra lift IL-3 below and is not on the demo-critical path.
6. Chart: the sheet's UV *is* its pattern (u around, v down), the tail's is its
   own strip, the knot's a small island; one 2048 × 1024 texture, selvedge on
   every free edge (hem, tail edges, the sheet's top edge where it shows under
   the tablet wing).

Openings: **none**. No armhole, no hand slit, no neckline — the tablet arm is
above the loop, the raised wing is above the knot. `held_out_vertices` and
`cut_openings` are not called in wrap mode.

## 5. Skinning from the support graph (the class-F fix)

`transfer_body_weights` (nearest-vertex pooling) is **not used** for the wrap.

* knot: `chest` 1.0.
* sheet top row: the body's own skin weights at the pinned surface point,
  **restricted to {chest, body} and renormalised** — the loop moves with the
  torso that holds it, never with the wing it passes under.
* every column: `w(y) = lerp(w_top, body, s)`, `s = clamp((y_top − y) /
  (y_top − y_hip))` with `y_hip` the `body` joint height — cloth below the hip
  is 100 % `body`.
* tail: its top row takes the knot's weights, then the same blend downward.
* `smooth_grid_weights` along the grid as today; top-4 influences.
* **forbidden joints**: `neck head cap tassel eye_* lid_* wing_* leg_* foot_*
  tail` carry exactly zero garment weight (gate G2). `tail` is the owl's
  feather tail; the optional `wrap_tail` joint of IL-2 is a new joint and is
  the one exception, only when IL-2 is picked.

The `hop` clip's root translation lives on `body`; the legs tuck inside the
skirt under their swing bound. In `tablet_show` the tablet wing lifts off the
sheet's edge (a gap opens for a beat); the sheet does not follow the wing.

## 6. Acceptance gates — defined before the first build

From the two research volumes (gates A–H, D0–D7), reduced to what the harness
can measure today. The build prints the table; a gate that fails blocks
promotion, not the build.

| gate | measure | threshold |
|---|---|---|
| G1 worn | top-row drift from its posed support point, all clips + head probe | ≤ 0.010 H |
| G2 support only | garment weight mass on forbidden joints | = 0 |
| G3 no penetration | vertices inside the body, per clip | ≤ 20 of ≈ 12k; none visible in the gray sheet |
| G4 hem | min hem y over θ ≥ bead top + 0.02; σ of hem y | ≥ 0.122; ≤ 0.015 H |
| G5 root covered | ray test from the **torso axis** through the root-ring vertices **below the tie** (`y ≤ y_k + ½ knot_size`; the shoulder above the tie is bare in the reference), hits knot ∪ tail ∪ sheet | ≥ 90 % (v9: 78 %) |
| G6 tail clear | tail–`wing_left` clearance over `wave` | ≥ 0 |
| G7 stretch | weft/warp l/l0 median, p90 | [0.97, 1.05], ≤ 1.12 |
| G8 collar visible | garment vertices above the collar band's measured lower rim at their angle (the coarse band dict over-counted), knot excepted | 0 |
| G9 sim-off sheet | gray renders: idle · wave · hop · nod · tablet_show · head-follow, three phases each, + turntable | human gate (user) |
| G10 budget | guide GLB size / garment triangles | ≤ 1.6 MB / ≤ 8k |

G9 is produced **gray first**, textured second, and is what the user judges —
at the kiosk's distance, behind the existing `guideModel` flag.

## 7. Motion

**Layer 1 (demo): animated guide only.** With §5 the wrap is transported by the
owl's own animation, which is what the second volume says the primary motion
must be. It passes its Test A by construction: simulation is off.

**Layer 2 (after the demo): runtime residual** — infra lift IL-1 below. Until it
exists, the tail and hem have no secondary motion; that is stated, not hidden.

## 8. Infra lifts — named and scoped

| id | name | where | scope | when |
|---|---|---|---|---|
| **IL-1** | **Guide Cloth Runtime** | `Afromaha-Visit2D/src/renderer/src/guide/owlRig.ts` | a verlet/PBD residual over a coarse proxy of the sheet's skirt and the tail, stepped between `mixer.update` and `render`, in the character's local frame, max-distance field 0 at the loop → large at hem/tail, progressive-release tuning (Test G); proxy→render binding in the guide GLB | **post-demo**, ≈ 2 days + tuning, kiosk fps risk |
| **IL-2** | **Tail pendulum bone** | meshforge skeleton (`wrap_tail` joint, tail weights blend to it) + ≈ 60 lines in `owlRig.ts` (spring–damper on the chest's acceleration, sets the joint quaternion after `mixer.update`) | the cheapest honest residual: one rigid DOF | **day-2 stretch** if G9 is signed off |
| **IL-3** | **TRELLIS prop post-processing** | `meshforge/prop.py` (new): clean → decimate → orient → scale → frame → retexture | needed only for knot K2 | day 1, ≈ ½ day, in parallel via `summon_luna` (< 20 min tasks) |

**No infra lift is required for the demo-critical path** (sheet + K1 + §5
skinning + gates): it is all inside `meshforge/drape.py`, `tools/garment_diag.py`
and the existing export.

## 9. Schedule to the demonstration

| day | deliverable |
|---|---|
| **D0 — 08-23** | spec approved → plan → wrap mode (support loop, gather, tail), support skinning, K1, gates G1–G8 in the harness. Evening: **gray G9 sheet in Safari**. |
| **D1 — 08-24** | textured build, chart and selvedge, guide build, kiosk flag `guideModel=owl-guide-kente-wrap`; user judges at distance. K2 attempt in parallel if approved. |
| **D2 — 08-25** | fixes from the judgement; IL-2 if approved; promote to `viewer/assets` and the kiosk's `owl-guide.glb` **only on sign-off**; rehearsal. |
| **D3 — 08-26** | demonstration. |

## 10. Left deliberately

* Self-collision stays off; the gathered folds may touch at the knot.
* The sheet's two real ends are not modelled — the loop is a closed tube;
  overlap is implied by the knot and tail.
* No per-pose re-drape; no baked secondary motion (the research says that is
  the wrong fix).
* `--kente-style tunic/robe/vest/sash` remain reachable; nothing is deleted.

## 11. Decisions the user owns

Filed as tickets with a pick column in
`docs/superpowers/tickets/2026-08-23-owl-kente-wrap-tickets.md`: W-K2
(TRELLIS knot), W-IL2 (tail pendulum bone), W-TAIL (tail length), W-PROMOTE,
IL-1, IL-3, and the three cleanup tickets C-1..C-3. Only W-0, the
demo-critical path, is picked.
