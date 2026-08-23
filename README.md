# Owl garment / model — handoff for a fix

Everything needed to rebuild the AI‑CCORE owl mascot with its kente garment, plus the source model,
the current builds, and the renders that show what is wrong with them.

## The state, plainly

**The garment does not look right.** In the viewer it reads as a bunched green‑and‑red striped skirt around
the owl's hips, not as a wrapped kente cloth. See `renders/sheet_shipped_turntable.png` and
`renders/sheet_before_after.png`.

The measured reason: the cloth's support line is squeezed into a narrow band of the body.

| quantity | value | why |
|---|---|---|
| tie height | 0.432 H | must stay under the AI‑CCORE collar rim at 0.462 H |
| top edge, front | 0.327 H | must pass under the tablet wing's lower contour |
| top edge, flank | 0.226 H | must pass under the raised wing's swept root |
| hem | 0.143–0.174 H | above the ankle beads (they start at 0.10 H) |

So the "shoulder‑to‑hip diagonal" is a 0.1 H drop on a 1.9 H body — a diagonal on paper, a horizontal band at
belly height on screen, with bare owl above it. Both wings attach low and wide on this model and the chest is
already occupied by the collar sash, which is what forces the cloth down.

The gate that blocks the obvious fix is **G8, "no cloth above the collar band's rim"**: a toga has to cross the
shoulder, which here means crossing the sash. That was set as an acceptance criterion up front and was never
revisited.

## What the code does

`meshforge/` is a from‑scratch pipeline: hi‑res GLB in, rigged and textured GLB out.

```
owl_pipeline.py    the whole build: mesh prep, regions, unwrap, texture bake, eyes, collar decal,
                   skin weights, garment, beads, clips, glTF export, validation, gate table
drape.py           position‑based cloth: pattern drafting, constraints, tethers, colliders,
                   radial bounds, the PBD solve, the woven texture bake
wrap.py            THIS is the kente wrap: measures a support loop off the body, drafts and drapes
                   the sheet and the hanging tail, places the knot, skins all three to the torso
knot.py            the tied bunch at the shoulder (procedural)
gates.py           G1–G8 as numbers with thresholds, printed by every build
robe.py, sleeve.py, textile.py, regions.py, fk.py, rigexport.py, clips.py, beads.py, ...
```

The cloth solver already has **gravity, damping, friction, a settling pass and cloth‑cloth separation**
(`ClothParams.gravity -9.0`, `damping 0.86`, `friction 0.55`, `settle_iterations 60`, `self_collide` — which
**defaults to 0.0, i.e. off**, and was off for every build here). What sits on top of it in `wrap.py` is a lot of
bespoke scaffolding — a measured support loop, radial bounds, per‑column tethers and cut lengths — and that
scaffolding, not the simulation, is what decides the silhouette. If you want cloth that simply falls on the
model under gravity and settles, the solver is there; the wrap's fitting machinery is what would have to go.

## Gates on the current build

Master 6/8 — G1 0.0015, G2 0, G3 6, G4 0.1427 (σ 0.0075), **G5 0.419** (wants ≥ 0.90), G6 0, **G7 p90 1.16**
(wants ≤ 1.12), G8 0. Guide (14k faces) 5/8: the same code reads G3 = 57 there. G3's count is a chaotic
statistic — identical code gives 6 and 57 on the two resolutions, and small parameter changes swing it 6 ↔ 67
without changing anything visible (`probes/sheet_wave_peak.png`).

Full write‑up, including what was measured and rejected: `docs/handoff/2026-08-22-kente-robe-handoff.md`.
Design spec, plan and the ticket board with open decisions: `docs/specs/`, `docs/plans/`, `docs/tickets/`.

## Rebuild it

Python 3.11. `numpy scipy trimesh pygltflib pillow xatlas playwright pytest` (playwright only for renders:
`python3 -m playwright install chromium`).

```bash
# master (~1 min with a warm cache; the cache directory is created on first run)
PYTHONPATH=. python3 -m meshforge.owl_pipeline --hires "assets/AI-CCORE Owl.glb" \
    --out build/owl-kente-wrap.glb --cache .owl_cache --kente --kente-style wrap --beads --cloth-res 76,42

# kiosk guide (14k faces, 1.68 MB)
PYTHONPATH=. python3 -m meshforge.owl_pipeline --hires "assets/AI-CCORE Owl.glb" \
    --out build/owl-guide-kente-wrap.glb --cache .owl_cache \
    --faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256 --kente --kente-style wrap --beads

# diagnosis sheets (gray + textured + close-ups; ~4 min, headless chromium)
PYTHONPATH=. python3 tools/garment_diag.py build/owl-kente-wrap.glb --out diag \
    --report build/owl-kente-wrap.report.json --stage renders

# a self-contained HTML viewer for a GLB
PYTHONPATH=. python3 artifact/build.py --glb build/owl-kente-wrap.glb --out owl-kente-wrap.html

# tests (361)
PYTHONPATH=. python3 -m pytest tests/meshforge -q
```

Other garment styles the pipeline still supports: `--kente-style robe`, `--kente-style tunic` (both abandoned
earlier; the tunic was rejected as "wrinkles added to a shell").

## Contents

```
assets/AI-CCORE Owl.glb     the source model (20 MB, 120k faces)
build/                      the current master and kiosk builds, their reports and build logs
meshforge/ tools/ tests/    the pipeline, the diagnostics, the tests
artifact/                   the self-contained HTML viewer builder
docs/                       spec, implementation plan, ticket board, handoff notes
renders/                    the sheets that show the current state, plus LOOK.md (a written verdict)
probes/                     instrumented measurements of gates G5 and G7, with their scripts
```
