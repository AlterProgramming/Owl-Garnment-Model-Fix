# meshforge

Rig pipeline for generated (TRELLIS-style) meshes. Two entry points:

* `python3 -m meshforge <raw.glb> --out <rigged.glb>` — the original
  stage-B/C wing pipeline (clean → segment one part → blended skin →
  eyes v1 → wave clip). Kept for the unit tests and as a generic tool.
* `python3 -m meshforge.owl_pipeline --hires "<AI-CCORE Owl.glb>" --out viewer/assets/owl.glb --cache .owl_cache --previews out/owl_previews`
  — the AI-CCORE owl recipe: full-body OwlV1 skeleton, re-baked texture,
  AI-CCORE collar decal, eyes v2, six baked clips. Design and measured
  decisions: `docs/superpowers/specs/2026-08-21-owl-full-body-rig-design.md`.
  Add `--kente` for the kente robe (and `--beads` for the anklets):
  `docs/superpowers/specs/2026-08-22-owl-kente-robe-design.md`.

## Owl pipeline modules

| Module | Role |
|---|---|
| `retopo.py` | position-only weld, debris filter, quadric decimation, xatlas unwrap, surface-sample color transfer, texture bake |
| `bake.py` | UV face-id raster → texel world position/normal ("position map"), gutter dilation, cylindrical coordinates, text rendering |
| `regions.py` | OwlV1 region map (bbox-fraction + color rules + island cleanup), collar rim measurement, skeleton pivots |
| `weights.py` | region-constrained harmonic skin weights (rigid cores, blended seam bands, top-4) |
| `fk.py` | forward kinematics + LBS for previews and validation, same joint convention as the exporter |
| `collar.py` | measure the band's rims as smooth curves (`BandFrame`), flatten the embossed lettering, despike what survives, build the "AI-CCORE" decal |
| `textile.py` | procedural kente weave: colorways, strip recipes, named motifs, exact tiling, relief height |
| `robe.py` | the kente **tunic**: ray-scan the body in (angle, height) cells, hang a cloth from under the collar that falls past the widest point (gravity hull + clearance + pleats), drape it over both wings' fused roots, cut a slit around the hand + tablet and an armscye around the sleeve (signed-distance holes); measures its own clearance at rest and through the clips |
| `sleeve.py` | the kente **sleeve** on the wing that waves: the wing's own frame and cross-sections (slices + rays from the centroid), wide past the shoulder and snug at the cuff, cut where the wing is fused, weights from the wing's skin; hides its root under the tunic (`robe.garment_overlap`) |
| `kente.py` | the older skin-tight kente *decals* (`vest`, `sash`) and `bake_weave_maps`, the position-baked weave + relief shared with the robe |
| `beads.py` | beaded anklets, rigid to the leg joints |
| `paint.py` | 3D-aware texture ops: collar fill, eye-disc detection/blanking, grade |
| `eyes2.py` | stylized eyeball + cornea + lids, sized from the painted eyes |
| `clips.py` | idle / wave / blink / nod / hop / tablet_show keyframe clips |
| `rigexport.py` | multi-primitive, multi-material, multi-clip skinned GLB writer (linear COLOR_0, embedded textures) |
| `validate_rig.py` | structural validator for that layout (`python3 -m meshforge.validate_rig file.glb`) |
| `fastpreview.py` | 1 s flat-shaded software renders for region/weight/pose debugging |

Pixel verification through a real renderer:

```bash
python3 tools/owl_shots.py viewer/assets/owl.glb --out shots/ \
    --view front:0:5 --view side:90:0 --clip idle=1.0 --clip wave=0.8
```

(`tools/three_bundle.js` is a self-contained Three.js r160 IIFE build —
GLTFLoader, OrbitControls, RoomEnvironment, SkeletonUtils — used by the
harness and inlined into the artifact page.)

## OwlV1 contract (what a runtime can rely on)

Joints (viewer-perspective left/right, identity rest rotations, local
translation = pivot − parent pivot):

```
body → chest → neck → head → {cap → tassel, eye_left, eye_right, lid_left, lid_right}
       chest → wing_left (raised, −X) → wing_left_tip
       chest → wing_right (folded + tablet, +X)
body → leg_left → foot_left, leg_right → foot_right, tail
```

`wing_left_tip` owns the outer 54 % of the waving wing. One joint cannot
make a wing bend, however the weights are painted — every vertex it owns
takes the same rotation — so the wave drives the tip with the base's
motion delayed 75 ms and scaled, and the blend bands are wide (8 rings at
the shoulder, 10 at the elbow; 3277 of the wing's 5548 vertices are
shared between the two). Measured on the built asset, the wing's
deviation from a best-fit rigid transform through the wave is 1.3-3.2 %
of its own diagonal (max 12 %), against 0.6-1.1 % with the tip track
removed.

Clips: `idle` (loop, 4 s), `wave`, `blink`, `nod`, `hop`, `tablet_show`
(one-shots). They target disjoint joints, so a runtime may play them
simultaneously. `neck` and the eyes are runtime-driven (cursor follow);
each eye joint's glTF `extras.gaze_forward` is the local axis its pupil
is painted on, each lid's `extras.blink_axis` / `blink_close_deg` is the
closing rotation. Asset-level `extras.meshforge` lists joints, clips and
eye radii.

## Consumers

* `viewer/owl_guide.js` — Babylon corner widget (idle/blink/wave/nod/hop/tablet,
  cursor gaze + head follow). Verify with `python3 tools/owl_widget_shots.py`.
* `artifact/` — the published Three.js page. `python3 artifact/build.py`
  reassembles it; `python3 artifact/verify.py` checks it by pixels.

## Build sizes

| Build | Command | GLB | Triangles | Used by |
|---|---|---|---|---|
| master | defaults | 7.75 MB | 148,557 | reference, stills |
| page | `--faces 88000 --tex 1536` | 5.4 MB | 116,180 | the artifact (base64-embedded) |
| guide | `--faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256` | 1.14 MB | 18,782 | `viewer/assets/owl-guide.glb`, the corner widget |
| master + `--kente --beads` | as master | 8.79 MB | +17,414 (tunic) +2,880 (sleeve) | `out/owl_kente/owl-kente-sleeve-wip.glb` (not promoted) |
| guide + `--kente --beads` | as guide | ~1.6 MB | +2,184 (tunic) +432 (sleeve), auto-scaled | `out/owl_kente/owl-guide-kente-sleeve-wip.glb` (not promoted) |

The robe's grid (`--robe-res cols,rows`, default 144,64) and chart
(`--robe-tex WxH`, default 2048x512) scale with `--faces`/`--tex` unless
given explicitly; `--robe-hem`, `--robe-clearance`, `--robe-pleats` and
`--kente-colorway` are the look; `--sleeve-length` (0–1 of the free wing,
default 0.72), `--sleeve-clearance` (air at the widest, default 0.06) and
`--no-sleeve` shape the sleeve. Every `--kente` build prints and records
(`<out>.report.json` → `robe_clearance`, `sleeve_clearance`) each garment's minimum distance to
the body and its count of vertices inside it, at rest and through all six
clips — the standard the robe is built to (ample, air between cloth and
body) is measured, not assumed.

The master GLB is geometry-bound, not texture-bound: 6.6 MB of vertex
buffers against 1.15 MB of textures. Decimation is therefore the only
lever that matters for weight; at the widget's ~200 px the guide build is
indistinguishable from the master side by side. `--eye-subdiv` is worth
knowing about — the eye assemblies are authored at a fixed tessellation
and at one point carried more triangles (14,592) than the whole decimated
body (13,964).

```bash
python3 -m meshforge.owl_pipeline --hires "<AI-CCORE Owl.glb>" \
    --out build/owl_web.glb --cache .owl_cache --faces 88000 --tex 1536
python3 artifact/build.py --glb build/owl_web.glb && python3 artifact/verify.py
```

## The collar band frame

`regions.measure_collar` finds the band from every *dark* vertex in a
broad height window, in 10 degree bins. That is fine for splitting
head/neck/chest and wrong for typesetting: it also catches the tablet,
the mouth and the wing piping (reported band 0.097 bbox-fraction tall
against a real facing of 0.044-0.067), and interpolating noisy per-bin
percentiles zigzags. Lettering laid on it drifted about 0.45 band-heights
relative to the band across the span.

`collar.measure_band_frame` fits low-order polynomials to the *red
facing's* rims in fine bins instead. The decal patch, the decal UVs, the
texture fill and the piping ring all read from that one frame, so they
agree by construction. A debug grid baked into the decal in place of the
text is the fastest way to see a parameterization problem — the chevron
kink in the old frame was invisible in the lettering but obvious in a
grid.

Tests: `python3 -m pytest tests/meshforge -q`.
