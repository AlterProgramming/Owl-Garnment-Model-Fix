# Coupled garment repair experiment

This branch is a controlled repair of the specific motion failure visible in the latest garment: **the owl appears to move underneath a pre-shaped clothing shell**.

It does not treat that as a wrinkle-quality problem.

## What was wrong in the code, not just the render

The repository already contains the key evidence.

1. `wrap.py::support_weights` explicitly restricted the sheet to `body` and `chest`, then faded every column to 100% `body` at the hip.
2. `gates.py::G2` made *all wing weight a failure condition*.
3. `measure_support` therefore compensated geometrically for moving wing roots by pushing the support loop down/out of their swept volume.
4. The PBD drape is baked into the GLB. There is no runtime cloth solver in the exported model.
5. The clips do animate `wing_left`, `wing_left_tip`, and `wing_right` substantially.

Those facts mean the runtime carrier can be internally consistent and collision-free while the wings animate beneath it. The old acceptance system rewarded that separation.

The repo's own tunic path already contains the counterexample: upper cloth over fused wing roots is given wing-root skin weights so that it rides the moving root. The wrap removed that mechanism in the name of a simplified support graph.

## Repair hypothesis

The GLB needs a useful **animated garment guide** before any baked folds are considered.

For this static-export pipeline the practical approximation is:

```
shipped garment pose
    = support/body transport
    + local body/wing-root guide influence
    + baked residual drape
```

The local guide is not applied uniformly. `meshforge/coupling.py::guide_weights` computes two fields:

- **vertical release**: strong animation authority near the top/support, fading to zero before the hem;
- **contact likelihood**: stronger local guide where the baked cloth already lies close to the character surface.

The local skin is restricted to:

```
body, chest, wing_left, wing_left_tip, wing_right
```

Head, neck, eyes, legs, feet, tail, cap, and tassel remain forbidden. This is not nearest-skin transfer without semantics.

The lower garment still releases back to the old support/body baseline. The intent is a carrier that is unmistakably *worn* without turning the kente into a body decal.

## Static drape changes

The experiment also reduces cues that were trying to manufacture "clothiness" in the bind pose:

| parameter | control | coupled experiment | reason |
|---|---:|---:|---|
| `gather_ratio` | 1.20 | 1.065 | stop baking a large bunch of motionless folds |
| `gather_half_deg` | 25 | 34 | broader, calmer support transition |
| `length_slack` | 0.020 | 0.006 | less pre-authored wrinkle surplus |
| `chest_ease` | 0.080 | 0.045 | reduce shell spacing |
| `hem_ease` | 0.030 | 0.012 | reduce lampshade flare |
| `hem_min_frac` | 0.55 | 0.50 | permit a narrower lower silhouette |
| `collide_offset` | 0.014 | 0.011 | reduce visible air at supported cloth |
| `collide_offset_hem` | 0.022 | 0.017 | reduce global inflated spacing |
| `sweep_band` | 0.12 H | 0.045 H | stop redesigning the whole support line around a wing that the carrier can now follow |
| `sweep_drop_max` | 0.08 H | 0.025 H | same reason |
| `root_bound` | on | off | do not pre-flare the cloth for a moving root; validate the shipped skin instead |

These are hypotheses, not beauty-render claims. They are isolated behind a separate entry point so the existing build remains an A/B control.

## Gate changes

Two old gates encoded the wrong contract.

### G1

Old: compare every top-row point only to a `body/chest` torso anchor.

New experiment: compare it to its **local animated guide**, restricted to the legitimate guide joints above.

### G2

Old: any wing mass fails.

New experiment: wing-root mass is allowed; mass on unrelated joints still fails exactly.

No penetration, hem, root coverage, tail clearance, stretch, and collar gates remain unchanged for the first experiment. If the visual result proves that G8 is still forcing the garment too low, it should be revised only with a concrete collar-visibility mask rather than deleted globally.

## Build A/B

Control:

```bash
PYTHONPATH=. python3 -m meshforge.owl_pipeline \
  --hires "assets/AI-CCORE Owl.glb" \
  --out build/owl-kente-wrap-control.glb \
  --cache .owl_cache --kente --kente-style wrap --beads --cloth-res 76,42
```

Coupled experiment:

```bash
PYTHONPATH=. python3 -m meshforge.owl_pipeline_coupled \
  --hires "assets/AI-CCORE Owl.glb" \
  --out build/owl-kente-coupled.glb \
  --cache .owl_cache --kente --kente-style wrap --beads --cloth-res 76,42
```

Do **not** pass the old explicit `--wrap-gather`, `--cloth-ease`, etc. while evaluating this first experiment; doing so overrides the experimental defaults.

## Motion diagnostic

Run the new probe on both final GLBs:

```bash
PYTHONPATH=. python3 tools/wrap_coupling_diag.py build/owl-kente-wrap-control.glb \
  --out probes/coupling-control.json
PYTHONPATH=. python3 tools/wrap_coupling_diag.py build/owl-kente-coupled.glb \
  --out probes/coupling-coupled.json
```

The probe uses the **shipped skin weights and shipped animation clips**. For each frame it compares the displacement of a wrap vertex with the displacement of the nearest animated body surface beneath it.

Primary metrics:

- `carrier_static_fraction`: nearby body moved meaningfully while garment moved <25% as much. This directly operationalizes “owl moves under shell.” Lower is better in the upper garment.
- `cosine_to_body`: directional agreement between garment and nearby body displacement. Higher is better where the garment is supported/contacting.
- `magnitude_ratio`: whether garment receives a comparable amount of primary motion.
- `wing_weight_mass`: confirms the upper carrier actually receives wing-root guide motion while the hem does not.
- `unrelated_weight_mass_total`: must remain zero.

A single worst frame is reported for quick review, but the complete sampled time series is retained because phase/coupling failures can be hidden by aggregating a clip.

## Approval order for this branch

1. **Mechanical A/B:** new coupling tests pass; unrelated mass remains zero.
2. **Skinned-only motion:** wave/tablet/chest animation no longer reads as the owl swimming inside the carrier.
3. **Gray turntable:** silhouette is calmer than control; no reliance on kente print or baked wrinkles.
4. **Pose sheets:** inspect `wave`, `tablet_show`, `idle`, `hop` at several times, especially 90°/135° views.
5. **Existing safety gates:** penetration/stretch/root/collar results are reviewed, not blindly optimized.
6. **Texture:** only after the gray carrier is accepted.

A numerical gate passing is not promotion. The branch exists because the previous system demonstrated that a garment can pass several geometric gates while still failing the human motion read.

## If this experiment still looks like a shell

Do not add folds.

The next escalation is to split the carrier into a stronger skinned guide region and a separate lower residual proxy, or to add the runtime local-frame residual described in the earlier IL-1 proposal. At that point the diagnostic above tells us whether the remaining failure is broad motion transfer or genuinely cloth dynamics.
