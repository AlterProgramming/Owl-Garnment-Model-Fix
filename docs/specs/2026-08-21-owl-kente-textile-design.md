# AI-CCORE owl — a procedural kente textile system

> **2026-08-22:** the *garment* half of this spec (the vest and sash
> decals) is superseded by `2026-08-22-owl-kente-robe-design.md` — a real
> robe mesh that stands off the body. The weave generator, the colorways
> and `kente.bake_weave_maps` below are what the robe reuses; the decals
> stay reachable as `--kente-style vest|sash`.

**Date:** 2026-08-21
**Track:** design (concept art approved by eye, this spec covers the
production asset that replaces it)
**Builds on:** `2026-08-21-owl-full-body-rig-design.md` (the OwlV1
skeleton and the `owl_collar_text` decal, which is the closest existing
precedent for "paint a garment onto specific geometry, share the body's
skin weights, ship it as its own material") and the Afromaha-Visit2D
kiosk's `docs/superpowers/specs/2026-08-21-owl-guide-design.md` (the guide
that will eventually wear this).

## Problem

Afromaha-Visit2D's map covers Ghana, and its own content already
describes kente accurately: strips hand-woven by Asante and Ewe makers,
patterns that can carry proverbs and social meaning. The ask is to dress
the owl guide in a Kente cloth outfit with beads, "directly inspired by
the topics we are discussing in this map" — not a generic plaid.

The first pass at this used the ChatGPT image-gen MCP connector
(text-to-image only, no reference-image input) to draw the whole owl
wearing kente. Two problems surfaced once we looked past the first good
result:

1. **It's not a texture.** A product-photo-style render of a stole has
   perspective, a drop shadow, and cropped fringe. None of that tiles or
   UV-maps onto a mesh — it's mood-board art, not an asset.
2. **It's not reusable.** Every variation costs a slow (30–90 s),
   non-deterministic round trip through someone else's diffusion model,
   and there's no way to ask for "the same cloth, different colorway"
   without redrawing the whole owl and hoping the proportions don't
   drift.

The reference (`AlterProgramming/Agentic-Production-Studio`,
`05_MODEL_FIRST_VISUAL_DOCTRINE.md`) names this directly: *"a generated
image is a view of a retained scene, not the terminal artifact."*
ObjectForge's Scope 3 applies a persistent, parametric *design language*
across many objects instead of one-off renders. This spec is that idea
applied to fabric: kente is not arbitrary art, it is a describable
structure (narrow woven strips, each strip a repeating sequence of
weft-faced bands and warp-float motifs), so it can be **generated**, not
drawn.

## Architecture (as built)

```
meshforge/textile.py
    Colorway          — named palette (ground colors, accent thread, motif color)
    WeaveParams        — strip width, block height, strip-recipe cycle, motif set, seed
    weave_color_at()   — the weave logic, generalized to arbitrary (not
                         grid-ordered) pixel-unit coordinates
    generate_weave()   — dense-grid wrapper over weave_color_at; tiles by
                         construction (periodic in x with period
                         strip_px * strip_cycle, in y with period block_px)
    weave_height_at()  — the same band structure as a grayscale relief
                         height, for the normal-map bake
    to_image()          — meshforge.paint's convention, PNG out

meshforge/kente.py
    build_kente_decal(): extract regions.classify's "chest"+"body" faces
    as their own sub-mesh → Laplacian-smooth it → retopo.unwrap (xatlas
    chart+pack, low per-triangle distortion) → offset outward along its
    own normals → bake color AND (optionally) a tangent-space normal map
    by real 3D position (bake_position_map + cylindrical_coords), not by
    sampling a flat image through the raw chart UVs → new MaterialSpec
    ("owl_kente") → PrimitiveSpec added to write_rigged_glb's list, rigid
    to the body's own skin weights at the decal's source vertices — same
    wiring pattern owl_pipeline.py already used for collar_text.

meshforge/beads.py
    build_symmetric_bead_rings(): measure both legs' circumference from
    regions.classify's own leg_left/leg_right vertices near the ankle,
    use the smaller (less label-contaminated) measurement for BOTH
    anklets, place bead_count small icospheres in alternating trade-bead
    colors around each, rigid-bound (not blended) to the leg joint —
    beads on a string don't deform with the skin under them.
```

`generate_weave` is deliberately geometry-free and mesh-free: it is
tested and iterated on its own, at zero marginal cost per variation,
before any triangle is touched. Colorway swaps, motif swaps, and strip
proportions are parameters, not new AI calls.

## Kente structure, as implemented

A strip is `strip_px` wide. `strip_cycle` distinct strip *recipes* repeat
across the width — real kente strips are sewn from cloth that alternates
design between neighbouring strips, not one flat repeat, and this is
where that variety comes from. Each recipe is derived from
`seed + strip_index` so two colorways or two seeds never collide.

Each recipe repeats vertically every `block_px` pixels, as fixed-fraction
bands (all pixel-integer math, no floating-point seams):

| band | fraction | content |
|---|---|---|
| solid | 0.16 | one ground color |
| motif | 0.44 | checker / **nkyimkyim** / cross / **babadua**, ground vs. motif color |
| accent | 0.06 | the thin light thread line real kente uses between blocks |
| solid | 0.16 | a second ground color |
| accent | 0.06 | thread line |
| solid | 0.12 | back to the first ground color |

Motifs are named after real Asante/Ewe weft-float patterns where a
description was actually verified (Wikipedia's Kente cloth entry;
essenceoftheroadart.online's motif glossary) — not assumed from memory:
**nkyimkyim** is a zigzag ("life's twists"), which the first version here
already matched; **babadua** is "tight horizontal bars stacked like
bamboo joints," which the first version did *not* match — it drew
vertical stripes under that name, a different, unnamed pattern. Fixed to
draw actual horizontal bars. `checker` and `cross` are common kente
weft-float shapes that weren't matched to a specific named motif in what
was checked, so they stay generic rather than claim a name that wasn't
verified.

## Colorways

Two named palettes, both grounded in the map's own Ghana chapter copy
(`countries.ts`: "Patterns and finished cloth can carry ideas about
courage, leadership, community, and history") and standard kente color
symbolism:

- `ASANTE_GOLD` — gold (wealth, status), deep red (spiritual/sacrifice),
  forest green (growth), white/gold accent thread.
- `EWE_ADANUDO` — indigo (peace), red, gold, white/gold accent thread.

## Real bugs found on the owl (not the synthetic test geometry)

Every one of these was invisible on `tests/meshforge/test_kente.py`'s
synthetic torso and only showed up rendering the actual owl — the
pattern across all four is the same: verify against the real asset with
a real renderer, not just unit tests against convenient geometry.

1. **Pinched, warped pattern.** A first version used a hand-rolled
   cylindrical (theta, height) UV projection, copying `collar.py`'s
   coordinate system — correct for the neck (close to a true cylinder),
   badly wrong for the belly (a doubly-curved, egg-shaped surface).
   Fixed by using `retopo.unwrap` (xatlas chart+pack) instead of inventing
   new UV math.
2. **Torn, fragmented pattern across chart seams.** Fixing (1) with
   xatlas, then sampling a flat pre-rendered weave image through its raw
   output UVs, produced something worse: `meshforge.bake`'s own module
   docstring says why — a jigsaw of many small charts means authoring in
   UV space directly is "hopeless." Fixed by baking color from real 3D
   position (`bake_position_map` + `cylindrical_coords`) instead of
   sampling a picture.
3. **The belly's circuit-trace emboss (part of the AI-CCORE brand mark,
   visible even in the flat reference art) showing straight through the
   "fabric."** The decal is an offset duplicate of the real surface, so
   it faithfully traced the bumps underneath. Root-caused by rendering
   the same close-up with no kente decal at all — the "wrinkle" artifact
   was there regardless, so it couldn't be a kente-specific bug. Fixed
   two ways: Laplacian-smoothing the decal's own copy of the geometry
   before offsetting (`_laplacian_smooth`, same principle
   `collar.flatten_collar` uses to remove *its* unwanted raised detail),
   and increasing `offset` from an arbitrary `0.0015` to `0.006` —
   matching `collar.py`'s own precedent — to physically clear the
   original, unmodified body geometry's bump height.
4. **Smeared, wrong-colored streaks in the (optional) woven-relief normal
   map.** Two real sub-bugs in the tangent-space math were found and
   fixed (a piecewise-constant height field spiking its derivative at
   band edges; a fixed global tangent vector that goes degenerate as the
   belly's curvature swings around) — neither changed the rendered
   artifact at all, which is what exposed the actual bug: pulling the
   *baked* normal texture out of the GLB (not the lit render) and looking
   at it flat showed the real problem. `bake.dilate_into_padding` blends
   *all* neighbours reached in a ring — fine for color (a blended fabric
   color still reads as "some plausible color"), wrong for a vector field
   (a blended direction from two unrelated xatlas charts is physically
   meaningless). Fixed with `_nearest_valid_fill`
   (`scipy.ndimage.distance_transform_edt`) — exact nearest-neighbor
   propagation, the same "extend the border pixels" technique
   texture-baking tools (e.g. Substance Painter's padding/dilation) use,
   without the averaging.
5. **Beaded anklets measuring visibly different sizes left vs. right.**
   `regions.classify` labels noticeably more geometry `leg_right` than
   `leg_left` at every height checked (2-3x the radius, consistently —
   not a few contaminating outliers to filter out), the same asymmetry
   `collar.py` already documents for `wing_right` vs `wing_left` and
   attributes to the tablet-holding wing folding down near that side.
   `build_symmetric_bead_rings` measures both legs and uses the smaller
   (less contaminated) reading for both anklets, keeping each leg's own
   measured position.
6. **The stole silhouette (`build_kente_sash_decal`) landed on the back of
   the owl, invisible from the front it's actually viewed from.** Its
   shoulder/hip anchors were filtered by left/right and high/low but
   never by front/back, and since `chest`/`body` wrap the full 360
   degrees, both anchors landed with negative z (the back). Fixed by
   adding a front-hemisphere filter using `region_map.collar["centre_xz"]`
   — which surfaced a second, independent bug in the same code: that
   value is in *bbox-fraction* space (`regions.measure_collar` computes it
   from `fractions`, not world coordinates), and the very first version of
   this anchor logic was already comparing it directly against raw vertex
   positions, silently comparing the wrong units for the left/right split
   too. Fixed by comparing against the vertices' own fraction array
   instead of converting the constant to world space.
7. **Zero-length vertex normals on the sash, in three separate, genuinely
   different ways, only the last of which was the real defect.** (a) A
   distance-threshold band selection is far more likely than the vest's
   whole-region one to catch a truly degenerate, near-zero-area sliver
   triangle right at its cutoff edge — fixed with `_drop_degenerate_faces`
   (the same `nondegenerate_faces()` cleanup `retopo.weld`/`retopo.decimate`
   already use, done explicitly rather than via trimesh's in-place methods
   so the mapping back to the original mesh's vertices stays correct).
   (b) Laplacian smoothing on a band this narrow can itself introduce a
   *new* degenerate triangle that wasn't in the original selection — fixed
   by running the same cleanup again after smoothing. (c) Neither fix
   actually resolved the validator failure, which is what proved neither
   diagnosis was the real one: the defect wasn't a degenerate face at all —
   `trimesh.vertex_normals` area-weight-averages a vertex's adjacent face
   normals, and smoothing had folded the band's cross-section enough that
   two individually valid, adjacent faces ended up with nearly opposite
   normals, cancelling to near-zero in their shared vertex's average.
   `nondegenerate_faces()` can never see this, because neither face is
   degenerate on its own. Fixed with `_repair_zero_normals`, which detects
   a near-zero-length normal directly and substitutes a safe
   outward-from-centroid fallback instead of chasing another upstream
   face-level cause.

## Verification

`tests/meshforge/test_textile.py` (11), `test_kente.py` (24), and
`test_beads.py` (10) check the properties that matter for a generated
*asset* rather than a picture: exact tiling, determinism, an opaque
alpha channel, that every color is one the colorway declared, low-distortion
UVs (not a bbox projection), the offset/smoothing geometry behavior, the
tangent-space normal map's basic validity, `_nearest_valid_fill`'s
nearest-not-blended contract on a synthetic two-island case, the symmetric
bead radius fix (and oval-bead shape) on synthetic geometry with
deliberately asymmetric/known parameters, and the sash's degenerate-face
and zero-normal fixes each reproduced directly on an engineered synthetic
case (not just re-run against the real owl and hoped). All 246 tests in
`tests/meshforge/` pass (no regressions to `collar.py`, `rigexport.py`, or
anything else touched along the way — `MaterialSpec.normal_image`/
`normal_scale` are additive, default `None`).

## Left open

* **Sash coverage.** The stole now validates, sits on the front, and
  renders cleanly, but the default shoulder/hip anchors put most of it
  behind the AI-CCORE collar and the tablet-holding wing — genuinely
  working, but visually subtler than the vest. One tuning pass (moving
  the shoulder anchor below the collar's own territory,
  `shoulder_height_percentile` 85 → 55) improved it; further placement
  tuning is still open, not a bug to fix.
* **Side visibility.** Both the vest and the sash are surface decals, not
  simulated or even rigid cloth with volume — from the side the vest is
  mostly occluded by the wing, which no amount of UV or texture work
  fixes. A real garment *mesh* (per the "separate object, not a decal"
  research this session did before building the vest) would read better
  there, at real cost.
* **Not promoted.** Every build in this spec's scope wrote to a scratch
  path; nothing here has been copied into `viewer/assets/owl.glb` or
  handed to the Afromaha-Visit2D kiosk's `owl-guide.glb` build. That,
  plus the guide-voice/interaction wiring the original ask also
  mentioned, is still open.
