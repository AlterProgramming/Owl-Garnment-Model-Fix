# Owl full-body rig — done (2026-08-21)

Everything in the 2026-08-21 brief shipped. Nothing is committed; `git status`
lists every new file.

## What exists now

| Piece | Where | Notes |
|---|---|---|
| Rigged owl | `viewer/assets/owl.glb` (7.75 MB) | 18-joint OwlV1 skeleton, region-constrained harmonic weights, re-baked 2048² texture, measured collar band + "AI-CCORE" printed panel, textured eyes with cornea + lids, 6 baked clips |
| Web build | rebuilt on demand, `--faces 88000 --tex 1536` (5.4 MB) | what the artifact embeds |
| Guide build | `viewer/assets/owl-guide.glb` (1.14 MB, 18,782 tris) | what the corner widget loads; `--faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256` |
| Pipeline | `meshforge/` (see its README) | `python3 -m meshforge.owl_pipeline --hires "~/Downloads/AI-CCORE Owl.glb" --out viewer/assets/owl.glb --cache .owl_cache --previews out/owl_previews` |
| Babylon widget | `viewer/owl_guide.js` + `owl_guide.css` + `owl_guide_test.html` | idle loop, random blinks (20 % doubles), wave on hover/click, nod, hop on double-click, tablet_show on long idle, cursor gaze + head follow, reduced-motion path |
| Artifact | https://claude.ai/code/artifact/0ea05e3c-04ef-4e64-a6b3-878c9c1fe0ba | source in `artifact/` — `python3 artifact/build.py` reassembles `owl_artifact.html` (8.08 MB) from `template.html` + `tools/three_bundle.js` + the GLB |
| Tests | `python3 -m pytest tests -q` | 215 pass |

## Verification (pixels, not flags)

- `python3 tools/owl_shots.py <glb> --out shots --view front:0:4 --clip wave=0.55` — poses any clip at an exact time through real Three.js.
- `python3 tools/owl_widget_shots.py --out shots/widget` — drives the Babylon widget headlessly; asserts wave/hop/gaze actually move pixels. Last run (on the guide build): wave 7.0, hop 18.0, blink 8.2, tablet 11.8, gaze L↔R 11.7, zero console errors.
- `python3 artifact/verify.py` — same for the artifact page, with region-limited metrics (a whole-frame mean hides a one-wing motion in a 16:10 stage). Last run: wave_wing 10.1, blink_face 18.7, gaze_face 25.0, hop 15.3, skeleton 1.6, zero console errors.

Widget sheets: `2026-08-21-widget-wave.png` (the wave frame by frame), `-widget-clips.png` (every clip + gaze), `-widget-in-page.png` (in place on `viewer/index.html`).

Screenshots kept next to this file: `*-owl-v4-full.png`, `*-owl-v4-blink.png`,
`*-widget-clips.png`, `*-artifact-clips.png`, `*-artifact-gaze.png`.

## Round two: the collar lettering (same day)

The first build's text followed the *region map's* collar description, and
that description is built for splitting head/neck/chest, not for
typesetting. It reads the band from every dark vertex in a broad height
window in 10° bins — which also catches the tablet, the mouth and the wing
piping — then interpolates the per-bin percentiles linearly. Measured
consequences: the reported band was 0.097 bbox-fraction tall against a real
red facing of 0.044–0.067, and the text's centreline drifted ~0.45
band-heights relative to the band across the span. Baking a debug grid into
the decal in place of the text showed the rest: a chevron kink across the
middle, invisible in lettering but obvious in a grid.

`collar.measure_band_frame` now fits low-order polynomials to the *red
facing's* rims in fine bins. Everything that touches the band reads that one
frame — the decal patch, the decal UVs, the texture fill, and a new dark
piping ring — so they agree by construction. Four further things fell out of
having it:

* the decal image is now sized to the band's true arc/height aspect and
  spans the patch's own angular range, so `cap_frac` / `width_frac` are read
  straight off the reference art and the final E stopped riding onto the wing;
* the texture fill follows the fitted rims instead of the "neck" label, which
  removed the red spikes into the chest and the white notch punched through
  the middle of the band;
* `despike_band` clamps the ridges `flatten_collar` could not reach;
* the lettering sits on an opaque panel of the band's own colour, feathered
  to nothing at the patch's UV limits, floating 0.005 above the surface —
  which covers the chips and ridges that remain, and is what a screen-printed
  scarf looks like anyway.

Before/after next to this file: `2026-08-21-collar-before.png` /
`-after.png`, and the two debug grids `-grid-before.png` / `-grid-after.png`.

## Round three: the collar tear, the wing, and page weight

**The tear was mine, not the generator's.** I had told the user the bib's
ragged top edge was source geometry. Flat-shading the mesh at each pipeline
stage — no texture, uniform grey — showed the decimated mesh clean and the
*flattened* one torn. `flatten_collar` was selecting its movable set from
the coarse dark-rim bounds, which over-read the band's height ~2x, so it
reached past the top rim into the chest and pulled those vertices down onto
the band's radius. It now takes the fitted `BandFrame`, measured before
flattening, and skips wing/tablet-labelled vertices (which it was shattering
at the +X end). Flat-shading each stage is the technique worth keeping: it
separates "the generator did this" from "we did this", and neither the
textured render nor any assertion showed the difference.

**The waving wing bends now.** It was one joint, so the whole 0.62-long wing
took one rotation and read as a paddle. It is split at 46 % of its length
into `wing_left` + `wing_left_tip`, blended wide (8 and 10 rings — 3277 of
5548 wing vertices are shared), and the wave drives the tip with the base's
motion delayed 75 ms and scaled 1.15x plus a curl. Deviation from a best-fit
rigid transform through the wave: 1.3-3.2 % of the wing's diagonal, max 12 %,
against 0.6-1.1 % with the tip track removed. Note for next time: widening
the blend bands alone could not have done this, and it is worth saying so
plainly rather than tuning weights and hoping.

**Page weight.** The master GLB is geometry-bound (6.6 MB of vertex buffers
against 1.15 MB of textures), so decimation is the only lever that matters.
The corner widget now loads a 1.14 MB / 18,782-triangle guide build instead
of the 7.75 MB master; at ~200 px the two are indistinguishable side by side.
`--eye-subdiv` was added because the eye assemblies were authored at a fixed
tessellation and carried more triangles (14,592) than the whole decimated
body (13,964). Babylon itself is the other half of the bill: 1.77 MB gzipped
for `babylon.js` plus 0.13 MB for the loaders, against 184 KB gzipped for the
Three.js bundle the artifact uses.

## Round four: the wave was upside down

Asked to *show* the widget rather than measure it, a frame-by-frame contact
sheet of the wave (`2026-08-21-widget-wave.png`) made it obvious that the
wing went **down** — it tucked against the chest for the whole oscillation
and came back up at the end. The sign in `wave_clip` had been inverted since
the clip was written: positive Z swings the −X wing out and up, negative Z
tucks it in (`2026-08-21-wing-z-sign.png` is the ±30° test that settles it).

Both pixel harnesses passed it every single run, because a wing moving the
wrong way still moves plenty of pixels. Neither measures *direction*. This is
the third variant of the same lesson in this file: a number that goes up is
not the same as a thing that is right. `wave_wing` on the artifact went 4.0 →
10.1 once the sign was fixed, which is also the tell that the old value was
low for a reason nobody had asked about.

Also from this round: `__owlGuide.poseClip(name, u)` now exists on the widget
(the artifact already had one), and it stops and resets every other one-shot
first — scrubbing blink and then scrubbing idle had been leaving the lids shut
in the "look left" capture. And the widget's framing was loosened (scale 1.78
→ 1.66, lifted 5.5 %) because the circular panel was cropping his feet and the
"GUIDE" caption sat on his toes.

## Bugs found and fixed by looking at pixels

1. `COLOR_0` was sRGB where glTF wants linear — the whole owl read washed-out pink.
2. TRELLIS *embossed* the hallucinated collar word as geometry; repainting alone left ghost relief. Fixed by fitting the band's base surface to its non-letter (red) vertices and re-projecting.
3. Babylon's glTF loader puts `scaling.z = -1` on `__root__`; assigning `scaling` wholesale mirrored the owl (scarf backwards, wings swapped).
4. Camera orbit angle was 180° out, so the widget framed his back.
5. Lids drawn inside the eye aperture read as cream blobs; parked outside it, and the shell enlarged so a blink fully closes.
6. Both pixel harnesses initially "passed" on ambient motion alone — the sway and the idle clip had to be frozen, and clips scrubbed to an exact time, before the numbers meant anything.
7. The collar band the lettering was laid on was measured by a dark-vertex heuristic that over-read its height by ~2x and zigzagged; see "Round two" above. A debug grid baked into the decal is what made it visible.
8. That same bad measurement was also tearing the bib open where it meets the chest — attributed to the generator until a flat-shaded render of each pipeline stage showed the decimated mesh intact and the flattened one damaged.
9. The wave rotated the wing the wrong way for its whole life. Every diff-based check passed it, because they measure magnitude, not direction. A contact sheet of consecutive frames is what caught it.

## Left deliberately

- Nothing is committed.
- Faint relief remains at the extreme ends of the collar band, outside the decal span and so not under the printed panel; not visible head-on.
- The bib's top edge still has a slightly stepped silhouette against the white chest at 120k faces; that part *is* source geometry (it survives at 300k faces and is present in the untouched generator output). The tear that used to sit above the lettering was ours and is fixed.
- `wing_right` is still a single joint; the tablet clip would benefit from the same treatment but the tablet must stay rigid, so it was left alone.
- `meshforge/eyes.py` (v1) and the old `meshforge/cli.py` wing pipeline are untouched — the tests still cover them.

## Round five: in the room (same day, evening)

The guide build is now inside `Afromaha-Visit2D` as a character, not a
widget: `src/renderer/src/guide/` there holds a Three.js port of
`viewer/owl_guide.js` (same gaze/neck/blink logic, no Babylon — the kiosk CSP
allows no CDN and the Three bundle is a tenth of the weight), an old-game
dialog box, a voice channel, and reactions wired to the map's stores. Design
and verification: `Afromaha-Visit2D/docs/superpowers/specs/2026-08-21-owl-guide-design.md`.
`viewer/owl_guide.js` stays as the Babylon reference harness; the Afromaha
copy is the one visitors see.
