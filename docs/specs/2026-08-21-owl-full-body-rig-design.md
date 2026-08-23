# AI-CCORE owl — full-body rig, re-baked texture, scarf text, eyes v2

**Date:** 2026-08-21
**Track:** architecture (brainstorming → spec → implementation in the same
autonomous session; the user granted free budget and left explicit
direction, so the approval gate was replaced by a restated goal they
could veto)
**Builds on:** `2026-08-20-meshforge-rig-pipeline-design.md` (stage B/C
wing rig, 6-joint skin) and
`2026-08-20-rigging-skinning-animation-guide.md` (the research this
spec implements: region map → region-constrained automatic weights →
progressive pose validation → real-render acceptance).

## Problem

The shipped owl (artifact "Owl Wave Rig", `viewer/assets/owl.glb`) is a
6-joint skin: one waving wing, two sphere eyes, two blink lids. Measured
defects, by pixels (headless Chromium screenshot of the artifact):

- Colors are washed out / pink. Root cause: `COLOR_0` was written with
  sRGB values but glTF vertex colors are linear, so every viewer
  brightens and desaturates them; the 2-median + 4-Laplacian denoise
  also blurs the texture detail.
- The scarf reads "ALCOPE" (TRELLIS hallucinated the text) instead of
  "AI-CCORE".
- Eyes are flat white spheres with a black pupil and a pink skin-colored
  lid patch — uncanny next to the reference art's huge dark-brown irises.
- Nothing else moves: no head, neck, legs, tail, cap, second wing.
- Framing in the artifact cuts the head off.

## Inputs (measured this session)

| Asset | Where | What |
|---|---|---|
| `owl-mascot.glb` | `AI-GATEWAY/3d-generated/` | 75,001 verts / 150,000 tris, closed surface (0 boundary edges), vertex colors, bounds x[-0.684,0.683] y[-0.950,0.948] z[-0.589,0.587]. Front = +Z, raised wing = −X. Holds a tablet in the +X wing. |
| `AI-CCORE Owl.glb` | `~/Downloads/` | 286,656 verts / 514,660 tris, **open patchwork** (761 components, 64k boundary edges — gaps are real, welding at 1e-3 does nothing), but carries the real 2048² baseColor / metallicRoughness / normal JPEGs with a fragmented per-patch UV atlas. Same bounds as the mascot. |
| `owl-3dshading.png`, `owl-no-cap.png` | `~/Downloads`, `viewer/assets` | Reference art: red owl, cream face/belly, orange beak/feet, red collar with white "AI-CCORE", big dark-brown eyes with two catchlights, graduation cap with red tassel. |

Decision (revised after measuring): **geometry and color both from the
hi-res export.** Its "patchwork" is only UV/normal vertex splitting — a
position-only weld (`merge_vertices(merge_tex=True, merge_norm=True)`)
closes it to 12 boundary edges / 9 components, and quadric decimation to
~120k triangles gives a closed 60k-vertex base with more surface detail
than the mascot. Color is transferred by nearest-surface-sample lookup
(3M texture-colored samples of the original surface), which sidesteps the
fragmented atlas entirely.

## Architecture

```
AI-CCORE Owl.glb (hi-res, textured)
     ├─► retopo.weld (position-only) + drop debris ─► retopo.decimate (~120k tris)
     ├─► retopo.SurfaceColorSampler (3M texture-colored surface samples)
     ├─► regions.classify (bbox-fraction + color rules + component cleanup → label per vertex)
     ├─► collar.flatten_collar (relax the embossed "ALCOPE" ridges)
     ├─► retopo.unwrap (xatlas → split verts + UVs, source_index to pre-split verts)
     ├─► retopo.bake_texture: bake.position_map (UV face-id raster → texel 3D position)
     │                        → sampler.query → island padding
     ├─► paint: collar flat fill, eye-disc blanking + socket shadow, mild grade
     ├─► regions.build_skeleton (pivots measured from the labelled mesh) + eyes2 pivots
     ├─► weights.harmonic (region-constrained Laplacian solve, top-4, normalized)
     ├─► eyes2.build_eye_set (iris/pupil/catchlights sphere, cornea shell, upper+lower lids)
     ├─► collar.build_collar_decal ("AI-CCORE" decal primitive, shares collar weights)
     ├─► clips.default_clips (idle, wave, blink, nod, hop, tablet_show)
     ├─► rigexport.write_rigged_glb (textured + vertex-colored primitives on one skin,
     │                               17-joint hierarchy, 6 animations, linear COLOR_0)
     ├─► validate_rig (IBM×bind≈I, weights, indices, channels, texture present)
     └─► previews (FK-posed software renders) + tools/owl_shots.py (headless Chromium pixels)
```

Output: `viewer/assets/owl.glb`, consumed by `viewer/owl_guide.js`
(Babylon, corner widget) and by the republished artifact (Three.js,
inline bundle with RoomEnvironment).

## Skeleton contract (OwlV1)

Names are **viewer-perspective** (screen-left = mesh −X), matching the
existing `eye_left`/`eye_right` joints. All joints have identity rest
rotation; a joint's local translation is its pivot minus its parent's
pivot, so inverse-bind = translate(−pivot) and local rotation axes equal
world axes.

```
body                     root; pivot at torso centre (y≈0.30 of bbox height)
├─ chest                 breathing / lean; pivot y≈0.42
│  ├─ neck               runtime cursor-follow; pivot = scarf band centre
│  │  └─ head            baked nod/tilt; pivot = top of the scarf ring
│  │     ├─ cap          rigid; pivot = cap base centre
│  │     │  └─ tassel    pendulum; pivot = the cap corner the tassel hangs from
│  │     ├─ eye_left, eye_right      runtime gaze
│  │     └─ lid_left, lid_right      blink clip (+ runtime follow of gaze pitch)
│  ├─ wing_left          raised wing (−X), inner half; pivot = seam-ring centroid (shoulder)
│  │  └─ wing_left_tip   outer 54 % of the wing; pivot = the inner/outer seam centroid
│  └─ wing_right         folded wing + tablet (+X); pivot = its shoulder
├─ leg_left  → foot_left     pivots at hip ring / ankle
├─ leg_right → foot_right
└─ tail                  pivot at tail root (back-bottom)
```

Regions are rigid cores; blending happens only in a graph-distance band
around each seam (default 3 rings), where harmonic weights are solved.
The waving wing is the exception and is deliberately soft: 8 rings at the
shoulder and 10 at the elbow, so 3277 of its 5548 vertices are shared
between `wing_left` and `wing_left_tip`. The second joint is what makes
the wing bend at all — weight painting cannot, because a single joint
applies one rotation to every vertex it owns, however gently the seam is
feathered. Measured through the wave, the wing deviates from a best-fit
rigid transform by 1.3-3.2 % of its diagonal (max 12 %); with the tip
track removed the same poses give 0.6-1.1 %.

## Clips (all baked into the GLB)

| Clip | Joints | Notes |
|---|---|---|
| `idle` | chest, body(translation), tail, tassel, wing_right | 4 s loop; breathing ±1.2°, bob ±4 mm, tail wag, tassel swing, tablet micro-sway |
| `wave` | wing_left, wing_left_tip | 1.6 s one-shot; raise + 3 oscillations + settle. The tip repeats the base's motion 75 ms late and scaled 1.15x, plus a curl, so the outer wing trails and overshoots |
| `blink` | lid_left, lid_right | 0.26 s one-shot, triggered at random intervals by the runtime |
| `nod` | head | 0.9 s one-shot |
| `hop` | body(translation), leg_*, foot_* | 0.7 s one-shot |
| `tablet_show` | wing_right | 1.4 s one-shot; lifts the tablet toward the camera and back |

Clips drive disjoint joint sets so they can play simultaneously without
blending logic in either runtime; `neck` and the eyes are runtime-only.

## Texture / paint

- Atlas 2048², baseColor JPEG q≈88, sRGB (tagged automatically as
  `baseColorTexture`). Material metallic 0, roughness 0.75.
- Scarf, revised after measurement: the band is described by a
  `collar.BandFrame` — low-order polynomials fitted to the *red facing's*
  upper and lower rims in fine angular bins. The region map's own collar
  description (percentiles of every dark vertex, 10° bins, linear
  interpolation) is kept for splitting head/neck/chest but is not usable
  for typesetting: measured on this asset it over-reads the band height
  by ~2× (0.097 bbox-fraction against a real facing of 0.044–0.067) and
  its interpolation zigzags, which drifts lettering ~0.45 band-heights
  relative to the band across the span.
- TRELLIS *embossed* the hallucinated letters as geometry, so the band is
  first fitted flat against a base surface fitted to its non-letter (red)
  vertices, then `despike_band` clamps ridges that survive that, then the
  texture is flat-filled with the band's median red **between the fitted
  rims** (not by region label — the label boundary is ragged) with a dark
  piping ring just outside each rim.
- "AI-CCORE" is a separate decal primitive sharing the collar's skin
  weights: the collar's own front triangles offset 5 mm along the normal,
  UVs running rim-to-rim in v and across the patch's own angular range in
  u, and an RGBA image sized to the band's true arc/height aspect so cap
  height and text width are set as direct fractions of the band. The
  image carries an opaque panel of the band colour behind the letters,
  feathered to nothing at the patch's UV limits, which hides the residual
  chips the flatten cannot reach. Crisp at any zoom, independent of atlas
  texel density.
- Debugging note: bake a grid into the decal in place of the text to see
  a parameterization fault. The chevron kink in the first frame was
  invisible in lettering and unmistakable in a grid.
- Eyes: the painted eye discs are blanked to the local face cream with a
  soft AO ring; new geometry sits exactly on the measured disc centres
  with the measured disc radius.
- `COLOR_0` on vertex-colored primitives is converted sRGB→linear before
  export (fixes the washed-out look for eyes/lids).

## Eyes v2

Per eye: (1) eyeball sphere, vertex-colored: warm sclera, dark-brown iris
(radial gradient, darker limbal ring), black pupil, two painted
catchlights (upper-left large, lower-right small) in the bind gaze
direction; (2) cornea shell, transparent PBR (alpha blend, roughness
0.08) for real specular; (3) upper lid cap at rest covering ~12 % of the
eye, lower lid static sliver, both colored by sampling the surrounding
face texels, with a thin darker lash line at the rim. Runtime: smooth
pursuit with saccadic jumps, micro-saccades, convergence, blink every
2–6 s with occasional doubles, head (neck) carries ~35 % of the gaze
angle, upper lid follows gaze pitch.

## Validation

- `meshforge/validate.py` (extended): every joint index in range,
  weights finite/non-negative/sum≈1, no NaN, hierarchy acyclic,
  `world_bind[j] × inverse_bind[j] ≈ I` from the *exported* transforms,
  every animation channel targets a real node with monotonic times and
  unit quaternions, a baseColorTexture is present.
- Pose sweep previews (software rasterizer with FK): rest, each joint
  at ±small and max, combined idle+wave frame.
- Pixel acceptance: `tools/owl_shots.py` drives the real Babylon widget
  and the Three artifact in headless Chromium and writes screenshots;
  they are looked at, and compared against the artifact's "before" shot.

## Out of scope

Blender/bpy; dual-quaternion skinning and pose correctives (no observed
failure yet that needs them); physics; re-generating the mesh with
TRELLIS.
