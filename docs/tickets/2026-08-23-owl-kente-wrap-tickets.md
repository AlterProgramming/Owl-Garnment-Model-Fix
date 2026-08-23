# Owl kente wrap — tickets

*Opened 2026-08-23 for the 2026-08-26 demonstration. Spec:
`docs/superpowers/specs/2026-08-23-owl-kente-wrap-design.md`.*

**How to pick:** change `[ ]` to `[x]` in the **pick** column (or say the id in
chat). Nothing with an open pick box is started. `cost` is wall-clock with
parallel luna tasks where noted.

| id | pick | title | scope | when | cost |
|---|---|---|---|---|---|
| **W-0** | [x] | Wrap, demo-critical path | support loop, gather, tail, K1 knot, support-graph skinning, gates G1–G8, gray G9 sheet, textured build, guide build, kiosk flag | D0–D1 | ≈ 1.5 days |
| **W-K2** | [ ] | TRELLIS knot | reference image → `trellis-2` → IL-3 post-processing → side-by-side with K1 at kiosk distance | D1, parallel | ≈ ½ day (luna) |
| **W-IL2** | [ ] | Tail pendulum bone | `wrap_tail` joint in the meshforge skeleton + spring–damper in `Afromaha-Visit2D … owlRig.ts`; one rigid DOF of secondary motion | D2 stretch, only after G9 sign-off | ≈ ½ day |
| **W-TAIL** | [ ] | Tail length | default: to hem level + 0.025 H (`tail_length_to` 0.15). Tick to keep; write another value here to change it: `____` | D0 | 0 |
| **W-G5** | [ ] | G5's threshold vs the design | the gate asks for cloth over 90 % of the raised wing's root ring; the wrap passes *under* that wing by construction, so it tops out near 0.53. Options: (a) re-scope the ring to the part below the tie, (b) lower the threshold to ~0.55 with the visibility evidence, (c) hold 0.90 and redesign the tie to cross over the wing root. Evidence: `out/owl_kente/probes/G5.md`, `out/owl_kente/diag_wrap/renders/sheet_g5gap.png` | D1 | 0 (a) / 0 (b) / ≈ ½ day (c) |
| **W-G3** | [ ] | G3 counts what it should measure | the wave's contact with the raised wing's base is a *count* of vertices, and that count swings between 6 and 67 on parameter changes that alter nothing anyone can see (measured: master 6, guide 57 with the same code; arm_margin 0.025→0.035 takes the master 6→50 and the guide 57→15). Proposal: G3 measures the worst penetration **depth** in H (and, optionally, whether any of it is unoccluded), with the count kept as detail. Evidence: `out/owl_kente/probes/sheet_wave_peak.png` — at the flagged frame nothing shows from any angle behind the owl | D1 | ≈ 1 h |
| **W-PROMOTE** | [ ] | Promote the wrap | copy the guide build to `viewer/assets/owl.glb` and the kiosk's `owl-guide.glb`; flip the flag default | D2, on sign-off only | ≈ 10 min |
| **IL-1** | [ ] | Guide Cloth Runtime | runtime residual PBD in `owlRig.ts` (local frame, max-distance field, proxy→render binding, progressive release) | **post-demo** | ≈ 2 days + tuning |
| **IL-3** | [ ] | TRELLIS prop post-processing | `meshforge/prop.py`: clean → decimate → orient → scale → frame → retexture; generic for any generated prop | with W-K2 | ≈ ½ day |
| **C-1** | [ ] | Delete `shots/` | 85 MB of earlier render sheets | any | 0 |
| **C-2** | [ ] | Delete root-level scratch | `P1.npy env.npy held.npy scan_th.npy scan_y.npy field.png` | any | 0 |
| **C-3** | [ ] | Delete earlier garment builds | `out/owl_kente/` robe / tunic-wip / sleeve-wip and their guide versions (v9 and production stay) | any | 0 |
| **C-4** | [ ] | Delete tonight's probe scratch | `out/owl_kente/probes/` is 172 MB of experiment builds (`taut*`, `fit*`, `cap`, `nr_*`, `sw_*`, `g_*`, `noroot`, `nobounds`). The scripts, the `.npz` dumps, `G5.md`/`G7.md` and `sheet_wave_peak.png` are the evidence and should stay; the `.glb`s can go | any | 0 |

## Log

* 2026-08-23 — board opened; W-0 picked by the user ("proceed with planning").
* 2026-08-23/24 night — W-0 built end to end: `meshforge/wrap.py`, `knot.py`, `gates.py`, pipeline
  `--kente-style wrap`, 31 new tests. Master `out/owl_kente/owl-kente-wrap.glb` (gates 5/8: G3 wave,
  G5, G7 fail — see handoff), guide 1.70 MB copied to the kiosk behind `guideModel=owl-guide-kente-wrap`
  (dev app running with it). Gray sheets + `LOOK.md` in `out/owl_kente/diag_wrap/`, viewer
  `owl-kente-wrap.html` — opened in Safari for the morning. Awaiting the user's G9 verdict; nothing
  promoted, nothing committed. W-TAIL built at the default (hem level + 0.025).
* 2026-08-23 03:xx — night shift 2, still inside W-0. Four luna workers (one knot, three instrumented gate
  probes; the `summon_luna_workflow*` tools are not registered in this session, so they were four concurrent
  `summon_luna` calls). Two probes hit the MCP idle cap at 30 min — G7's finished its artifacts anyway, G3's was
  killed and done by hand. Findings and changes:
  * **G3 fixed at master resolution, 58 → 6.** `measure_support` now clears the loop under where the raised
    wing's *root band* sweeps on the clips (`WrapParams.sweep_band/sweep_blend_deg/sweep_drop_max`), blended in
    outside the knot arc so the tie itself does not move. `wave` turns `wing_left` through 22° lift / 20° swing
    and the cloth may only be skinned to the torso, so the loop has to sit under where that root *goes*.
  * **G4 stays level** (0.1427 H, sigma 0.0075 — its best reading yet). G5 falls 0.53 → 0.42: covering the wing
    root and clearing its sweep are the same lever pulled in opposite directions (see W-G5).
  * **The taut cut** (`ClothParams.taut_cap`, `WrapParams.taut_cut`, `drape.taut_fall`) is implemented, tested and
    **off**: cutting each column to the path it actually travels instead of its vertical drop relieved the warp
    stretch by 0.01–0.02 and cost the hem (the surplus lands under the raised wing, where the wave then catches
    it). Kept because a different body may need it; the finding is in the handoff.
  * A cut-hang-measure-recut fit pass was tried and **removed**: it does not converge on this body (hem sigma
    0.052 → 0.047 over two passes, stretch unmoved).
  * **The knot** is now a cinched bunch — an icosphere pinched at the waist with soft flutes, a flat band wrapped
    round it, two short curled ends and three folds tucked under (1016 faces, was 2132). Three earlier shapes
    (sausage saucer, trefoil hoop, clenched claw) were rejected on sight; previews `knot_k1b/c/d.png`.
  * `tools/garment_diag.py` aims its close-ups at the parts the build measured for itself instead of the
    tunic-era crop list.
  * Master rebuilt: **6/8 gates** (G5 and G7 fail, both ticketed). Guide rebuilt (1.68 MB) and copied to the
    kiosk behind the same flag. New sheets in `out/owl_kente/diag_wrap2/`.
