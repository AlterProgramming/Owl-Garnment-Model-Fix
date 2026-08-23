# Owl Kente Wrap (W-0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the kente tunic with a wrapped cloth tied over the raised wing — sheet hung from a measured diagonal support loop, designed knot, hanging tail — skinned from its support graph, with the acceptance gates computed by the build, rendered gray for the user's judgement, then textured, built for the kiosk guide and put behind the existing `guideModel` flag.

**Architecture:** `meshforge/drape.py` keeps every cloth primitive (draft, constraints, tethers, PBD drape, colliders, radial bounds, chart bake); it gains a column→angle map so cloth can *gather* and a non-periodic mode so a strip can hang. New module `meshforge/wrap.py` measures the support loop off the body, drafts/drapes the sheet and tail, places the knot from `meshforge/knot.py`, and skins all three from the support graph (`support_weights`). New module `meshforge/gates.py` computes gates G1–G8 as numbers with thresholds; `owl_pipeline.py` grows `--kente-style wrap`, prints the gate table and writes it into the report. `tools/garment_diag.py` renders the gray G9 sheets for any `owl_kente*` material.

**Tech Stack:** Python 3.12, numpy, scipy, trimesh, PIL; pytest (`tests/meshforge`); the existing PBD solver in `meshforge/drape.py`; `tools/owl_shots.py` (playwright + Three.js) for renders; `artifact/build.py` for the Safari HTML viewer; Afromaha-Visit2D (Electron/Vite) for the kiosk flag.

**Spec:** `docs/superpowers/specs/2026-08-23-owl-kente-wrap-design.md` (read it first; the tickets board is `docs/superpowers/tickets/2026-08-23-owl-kente-wrap-tickets.md`).

## Global Constraints

- Frame: +Z = face, +Y up, raised wing `wing_left` at −X, tablet wing `wing_right` at +X. Angles θ about the torso axis: 0 = +Z, +90° = +X. Heights as fractions of the bbox height H (`(y - y0) / H`).
- Demo date 2026-08-26; this plan is the demo-critical path only (ticket W-0). No sleeve in wrap mode. No openings are cut.
- Garment weights live on `chest` and `body` only. Forbidden joints (exact zero weight): `neck head cap tassel eye_left eye_right lid_left lid_right wing_left wing_left_tip wing_right leg_left leg_right foot_left foot_right tail`.
- Gate thresholds (spec §6): G1 ≤ 0.010 H; G2 = 0; G3 ≤ 20 vertices inside per clip; G4 min hem ≥ bead-ring top + 0.02 H and σ ≤ 0.015 H; G5 ≥ 90 % of root-ring rays covered; G6 tail inside count over `wave` = 0; G7 warp/weft ratio median in [0.97, 1.05], p90 ≤ 1.12; G8 = 0 garment vertices in the collar band (knot excepted); G10 guide GLB ≤ 1.6 MB.
- Materials: sheet + knot `owl_kente_wrap`; tail `owl_kente_wrap_tail`. Body `owl_body`, decal `owl_collar_text`, beads `owl_beads` unchanged.
- Existing tests must keep passing: `python3 -m pytest tests/meshforge -q` (319 before this plan; `test_drape.py` 29 of them).
- **Commits:** the working tree carries uncommitted work from earlier sessions and the user commits on request only. Each task's "Commit" step means: run the task's tests, then *stop* — do not `git commit` unless the user has said to. Never `git add -A`.
- Nothing is promoted to `viewer/assets/owl.glb` or the kiosk's `owl-guide.glb` (ticket W-PROMOTE is unpicked). The wrap goes to the kiosk only as `owl-guide-kente-wrap.glb` behind the flag.
- The hi-res source is `"$HOME/Downloads/AI-CCORE Owl.glb"`; the cache `.owl_cache` makes master builds ~3–5 min. Builds write under `out/owl_kente/` (gitignored).
- Paths: always absolute or from the repo root `/Users/AI-CCORE/altageris/AICCORE/3d_development`.

## File structure

| file | responsibility |
|---|---|
| `meshforge/drape.py` (modify) | `Pattern.theta_cols`; `column_angles`; `draft(..., gather=)`; `initial_positions` reads `theta_cols`; `periodic=` on `_pairs`, `seam_columns`, `constraint_sets`, `build_tethers`, `drape`, `_grid_normals`; `_swing_bounds(..., with_neck=)` |
| `meshforge/wrap.py` (new) | `WrapParams`, `Support`, `root_ring`, `measure_support`, `grid_faces`, `build_kente_wrap` → `KenteWrap`, `support_weights`, `forbidden_mass` |
| `meshforge/knot.py` (new) | K1 procedural knot: `build_knot` → `Knot(primitive, base_points, info)` |
| `meshforge/gates.py` (new) | `Gate`, `gate_worn` … `gate_collar_visible`, `run_gates`, `format_table` |
| `meshforge/owl_pipeline.py` (modify) | `--kente-style wrap`, wrap flags, support skinning, gates in the log and report |
| `tools/garment_diag.py` (modify) | any `owl_kente*` material is garment for the gray variants |
| `tests/meshforge/test_drape.py` (modify) | gather / strip tests |
| `tests/meshforge/test_wrap.py`, `test_knot.py`, `test_gates.py` (new) | unit tests on the synthetic body from `tests/meshforge/test_robe.py::make_body` |
| `docs/superpowers/handoff/2026-08-22-kente-robe-handoff.md` (modify) | wrap section; commands |

---

### Task 1: Gather and strips in `drape.py`

**Files:**
- Modify: `meshforge/drape.py` (`Pattern` ~line 362, `draft` ~377, `initial_positions` ~467, `_pairs` ~494, `seam_columns` ~513, `constraint_sets` ~524, `build_tethers` ~561, `drape` ~612, `_swing_bounds` ~705, `_grid_normals` ~1042)
- Test: `tests/meshforge/test_drape.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `column_angles(theta, gather, cols) -> np.ndarray (cols,)`; `Pattern.theta_cols: np.ndarray | None`; `draft(body, params, bounds=(), gather=None)`; `_pairs(rows, cols, keep, di, dj, periodic=True)`; `seam_columns(rows, cols, keep, periodic=True)`; `constraint_sets(..., params, periodic=True)`; `build_tethers(..., rows, cols, periodic=True)`; `drape(..., bounds=(), periodic=True)`; `_grid_normals(verts, faces, used, cols, n, periodic=True)`; `_swing_bounds(mesh, labels, body, params, sweep_vertices, with_neck=True)`. Defaults reproduce today's behaviour exactly.

- [ ] **Step 1: Write the failing tests** — append to `tests/meshforge/test_drape.py`:

```python
# --------------------------------------------------------------------------
# gather (cloth crowding into a knot) and strips (a tail that is not a tube)
# --------------------------------------------------------------------------

def test_column_angles_are_uniform_without_gather_and_crowd_with_it():
    theta = -np.pi + (np.arange(48) + 0.5) * (2 * np.pi / 48)
    cols = 49
    uni = D.column_angles(theta, None, cols)
    assert np.allclose(np.diff(uni), 2 * np.pi / 48)
    g = np.ones(48)
    g[(theta > -2.2) & (theta < -1.5)] = 2.0          # twice the cloth per radian there
    th = D.column_angles(theta, g, cols)
    assert th[0] == pytest.approx(-np.pi) and th[-1] == pytest.approx(np.pi)
    assert np.all(np.diff(th) > 0)
    inside = ((th > -2.2) & (th < -1.5)).sum()
    assert inside > 1.6 * ((uni > -2.2) & (uni < -1.5)).sum()


def test_draft_with_gather_holds_more_cloth_at_the_top(measured):
    p = small_params()
    plain = D.draft(measured, p)
    assert plain.theta_cols is not None
    assert np.all(np.diff(plain.theta_cols) > 0)                       # spaced by the ring's own length, not by angle
    assert plain.theta_cols[0] == pytest.approx(-np.pi) and plain.theta_cols[-1] == pytest.approx(np.pi)
    g = np.ones(len(measured.theta))
    g[np.abs(measured.theta + np.pi / 2) < 0.35] = 1.3
    gathered = D.draft(measured, p, gather=g)
    assert gathered.circumference[0] > plain.circumference[0] * 1.02
    assert np.all(np.diff(gathered.theta_cols) > 0)


def test_a_strip_has_no_constraint_across_its_edges():
    rows, cols = 6, 8
    keep = np.ones(rows * cols, dtype=bool)
    a, b = D._pairs(rows, cols, keep, 0, 1, periodic=False)
    assert len(a) == rows * (cols - 1)
    assert not np.any(a % cols == cols - 1)
    src, dup = D.seam_columns(rows, cols, keep, periodic=False)
    assert len(src) == 0 and len(dup) == 0
    phi = np.tile(np.linspace(0, 1, cols), rows)
    v = np.repeat(np.linspace(0, 1, rows), cols)
    pinned = np.zeros(rows * cols, dtype=bool)
    pinned[:cols] = True
    t = D.build_tethers(phi, v, np.array([0.5, 0.5]), np.array([0.0, 1.0]), keep, pinned, rows, cols, periodic=False)
    # the bottom-right vertex hangs from the top-right pin, not from the top-left one across a seam
    last = rows * cols - 1
    assert t.anchor[t.vertex == last][0] == cols - 1


def test_a_strip_drapes_as_a_flat_sheet(body):
    mesh, _ = body
    rows, cols = 10, 6
    width, length = 0.3, 0.5
    phi = np.tile(np.linspace(0, 1, cols), (rows, 1))
    v = np.linspace(0, length, rows)[:, None] * np.ones((1, cols))
    pat = D.Pattern(phi=phi, v=v, circumference=np.array([width, width]), v_profile=np.array([0.0, length]),
                    length=length, fall=np.full(cols, length), grid_shape=(rows, cols), theta_cols=None)
    P0 = np.stack([phi.ravel() * width + 2.0, 1.0 - v.ravel(), np.zeros(rows * cols)], axis=1)   # away from the body
    keep = np.ones(rows * cols, dtype=bool)
    pinned = np.zeros(rows * cols, dtype=bool)
    pinned[:cols] = True
    p = small_params(iterations=30, settle_iterations=5)
    sets = D.constraint_sets(phi.ravel(), v.ravel(), pat.circumference, pat.v_profile, keep, rows, cols, p, periodic=False)
    tet = D.build_tethers(phi.ravel(), v.ravel(), pat.circumference, pat.v_profile, keep, pinned, rows, cols, periodic=False)
    P, _ = D.drape(P0, pat, sets, [D.Collider(mesh, n=20_000)], pinned, keep, p, tethers=tet, periodic=False)
    top_w = np.linalg.norm(P[cols - 1] - P[0])
    bottom_w = np.linalg.norm(P[-1] - P[-cols])
    assert abs(top_w - width) < 1e-6                         # pins untouched
    assert 0.7 * width < bottom_w < 1.3 * width              # the free edge is neither rolled into a tube nor torn
    assert P[-cols:, 1].max() < P[:cols, 1].min()            # it hangs
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/meshforge/test_drape.py -q -k "column_angles or gather or strip" 2>&1 | tail -5`
Expected: FAIL — `AttributeError: module 'meshforge.drape' has no attribute 'column_angles'` / `TypeError: ... unexpected keyword argument 'periodic'`.

- [ ] **Step 3: Implement** — in `meshforge/drape.py`:

(a) `Pattern`: add the last field

```python
    wrap_perimeter: dict = {}  # the measured perimeter the cloth wraps, per scan row, and what sized the piece
    theta_cols: np.ndarray | None = None   # (cols,) world angle each column hangs at; None = uniform, last = seam copy
```

(b) above `draft`, add

```python
def column_angles(theta: np.ndarray, gather: np.ndarray | None, cols: int) -> np.ndarray:
    """World angle of each of `cols` pattern columns (the last is the seam
    copy of the first). Uniform without `gather`; with it — a relative cloth
    density per scan angle, 1 where the cloth hangs plain and more where it
    is gathered into a knot — the columns crowd where the density is high,
    so the same pattern width covers less of the loop there and the surplus
    has to fold. This is where a wrap's folds come from: cloth, not noise."""
    if gather is None:
        return -np.pi + 2 * np.pi * np.arange(cols) / (cols - 1)
    g = np.asarray(gather, dtype=np.float64)
    if g.shape != np.shape(theta) or np.any(g < 1.0 - 1e-9):
        raise ValueError("column_angles: gather must be one density >= 1 per scan angle")
    edges = np.concatenate([[-np.pi], 0.5 * (theta[1:] + theta[:-1]), [np.pi]])
    cum = np.concatenate([[0.0], np.cumsum(np.diff(edges) * g)])
    return np.interp(np.linspace(0.0, cum[-1], cols), cum, edges)
```

(c) `draft`: signature `def draft(body: BodyMeasure, params: ClothParams, bounds: Sequence["RadialBound"] = (), gather: np.ndarray | None = None) -> Pattern:`; replace the `c_neck` line and the `th =` line:

```python
    seg = np.linalg.norm(np.diff(np.vstack([nk, nk[:1]]), axis=0), axis=1)        # (n_theta,) ring segment lengths
    g_mid = np.ones_like(seg) if gather is None else 0.5 * (np.asarray(gather) + np.roll(np.asarray(gather), -1))
    c_neck = float((seg * g_mid).sum())          # the cloth along the ring: its length, times the gather where there is one
    ...
    phi = np.arange(cols) / params.n_u
    th = column_angles(body.theta, gather, cols)
```

and pass `theta_cols=th` in the returned `Pattern(...)`.

(d) `initial_positions`: replace `th = -np.pi + 2 * np.pi * pattern.phi[0]` with

```python
    th = pattern.theta_cols if pattern.theta_cols is not None else -np.pi + 2 * np.pi * pattern.phi[0]
```

(e) `_pairs(rows, cols, keep, di, dj, periodic=True)`:

```python
    n = cols - 1 if periodic else cols
    r, c = np.meshgrid(np.arange(rows), np.arange(n), indexing="ij")
    r2 = r + di
    c2 = (c + dj) % n if periodic else c + dj
    ok = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < n)
```

(f) `seam_columns(rows, cols, keep, periodic=True)`: first line `if not periodic: z = np.zeros(0, dtype=np.int64); return z, z`.

(g) `constraint_sets(..., params, periodic=True)`: pass `periodic` into `_pairs`. `build_tethers(..., rows, cols, periodic=True)`: `d = np.abs(_wrap(phi[:, None] - a_phi[None, :])) if periodic else np.abs(phi[:, None] - a_phi[None, :])`.

(h) `drape(..., bounds=(), periodic=True)`: `seam_src, seam_dup = seam_columns(rows, cols, keep, periodic)`.

(i) `_grid_normals(verts, faces, used, cols, n, periodic=True)`: wrap the weld line: `if periodic: weld[cols - 1::cols] = np.arange(n)[0::cols]`.

(j) `_swing_bounds(..., sweep_vertices, with_neck=True)`: guard the neck block with `if with_neck and neck.any():`.

- [ ] **Step 4: Run the whole drape suite**

Run: `python3 -m pytest tests/meshforge/test_drape.py -q 2>&1 | tail -3`
Expected: 33 passed (29 old + 4 new).

- [ ] **Step 5: Full meshforge suite, then stop (no commit unless asked)**

Run: `python3 -m pytest tests/meshforge -q 2>&1 | tail -2`
Expected: all pass (323).

---

### Task 2: `WrapParams`, `Support`, `measure_support`

**Files:**
- Create: `meshforge/wrap.py`
- Test: `tests/meshforge/test_wrap.py`

**Interfaces:**
- Consumes: `drape.BodyMeasure` (`theta`, `y`, `wrap_r`, `neck_y`, `axis_xz`, `size_y`, `y0`, `y_hem`), `regions.RegionMap.labels`.
- Produces:

```python
@dataclasses.dataclass(frozen=True)
class WrapParams:
    knot_drop: float = 0.015          # H: the knot's top under the collar rim
    gather_half_deg: float = 20.0     # half-width of the knot base arc
    gather_ratio: float = 1.3         # cloth per loop length inside the arc
    arm_margin: float = 0.012         # H: loop below the tablet wing's lower boundary
    hem_frac: float = 0.125           # H: hem height
    loop_smooth_cols: float = 1.0     # gaussian sigma, scan columns
    root_band: float = 0.03           # H: wing vertices this close to the torso are the root ring
    knot_size: float = 0.07           # H: bulge height
    knot_strands: int = 7
    tail_width_frac: float = 0.22     # H
    tail_end_frac: float = 0.15       # H: where the tail ends
    tail_span_deg: float = 25.0       # angular width of the tail's pin arc
    tail_offset_deg: float = 30.0     # gap between the knot arc's back end and the tail's pin arc
    tail_n_u: int = 30
    tail_n_v: int = 80
    tail_texture_size: tuple[int, int] = (1024, 512)
    tail_strips: int = 6
    image_format: str = "PNG"
    raised_wing: str = "wing_left"
    tablet_wing: str = "wing_right"
    material_name: str = "owl_kente_wrap"
    tail_material_name: str = "owl_kente_wrap_tail"

class Support(NamedTuple):
    theta: np.ndarray      # (n_theta,) scan angles
    y: np.ndarray          # (n_theta,) loop height, world
    r: np.ndarray          # (n_theta,) loop radius off the body, world
    gather: np.ndarray     # (n_theta,) cloth density, 1 outside the knot arc
    theta_k: float         # knot centre, radians
    y_k: float             # knot height, world
    arc: tuple[float, float]   # (theta_lo, theta_hi) of the knot base arc, radians, may straddle -pi
    frame: dict            # origin (3,), normal (3,), tangent (3,), up (3,)
    root_ring: np.ndarray  # (m, 3)
    low: np.ndarray        # (n_theta,) tablet wing lower boundary, world y; nan where there is no wing
    info: dict

def root_ring(mesh, labels, wing: str, band: float) -> np.ndarray
def measure_support(mesh, region_map, body: BodyMeasure, params: WrapParams) -> Support
def loop_at(support: Support, theta: np.ndarray) -> np.ndarray   # (k, 3) world points on the loop
```

- [ ] **Step 1: Write the failing tests** — create `tests/meshforge/test_wrap.py`:

```python
"""Tests for meshforge.wrap — the kente cloth tied over the raised wing.

Synthetic body from test_robe.make_body: egg torso, a bulge on +X
(`wing_right`, the tablet wing), a fin standing off -X (`wing_left`,
the raised wing), two legs. The loop must be measured off that body the
same way it is measured off the owl: nothing here is a stored coordinate.
"""
import dataclasses

import numpy as np
import pytest

from meshforge import drape as D
from meshforge import wrap as W
from meshforge.textile import ASANTE_GOLD, WeaveParams
from tests.meshforge.test_drape import small_params
from tests.meshforge.test_robe import make_body

WEAVE = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=3, block_px=12, seed=0)
FAST_WRAP = W.WrapParams(tail_n_u=10, tail_n_v=16, tail_texture_size=(128, 64), knot_strands=4)


@pytest.fixture(scope="module")
def body():
    return make_body()


@pytest.fixture(scope="module")
def measured(body):
    mesh, rm = body
    return D.measure_body(mesh, rm, small_params(hem_frac=0.125), n_theta=48, n_rows=40)


@pytest.fixture(scope="module")
def support(body, measured):
    mesh, rm = body
    return W.measure_support(mesh, rm, measured, FAST_WRAP)


def _angle(P, axis_xz):
    return np.arctan2(P[:, 0] - axis_xz[0], P[:, 2] - axis_xz[1])


def test_root_ring_is_where_the_wing_meets_the_torso(body):
    mesh, rm = body
    ring = W.root_ring(mesh, np.asarray(rm.labels), "wing_left", 0.05)
    assert len(ring) >= 8
    assert ring[:, 0].max() < -0.3                      # on the -X side
    torso = mesh.vertices[np.isin(rm.labels, ["chest", "body", "neck"])]
    from scipy.spatial import cKDTree
    assert cKDTree(torso).query(ring)[0].max() <= 0.05 + 1e-9


def test_knot_sits_on_the_raised_side_under_the_rim(support, measured):
    assert np.sin(support.theta_k) < 0                 # -X
    rim = np.interp(support.theta_k, measured.theta, measured.neck_y, period=2 * np.pi)
    assert support.y_k < rim
    assert support.y_k > measured.y_hem
    assert np.linalg.norm(support.frame["normal"]) == pytest.approx(1.0)
    assert abs(np.dot(support.frame["normal"], support.frame["tangent"])) < 1e-6


def test_loop_dips_under_the_tablet_wing_and_rises_across_the_back(support, measured, body):
    mesh, rm = body
    low = support.low
    valid = np.isfinite(low)
    assert valid.any()
    th_a = support.info["theta_a_deg"]
    assert 0 < th_a < 180                               # the tablet wing is on +X
    i_a = int(np.argmin(np.abs(np.degrees(support.theta) - th_a)))
    assert support.y[i_a] < low[i_a]                    # the edge passes under the fused wing's contour there
    assert np.all(support.y[valid] <= low[valid])       # ... and everywhere along the flank
    # from the knot forward the loop only descends (the front diagonal)
    d = (support.theta - support.theta_k) % (2 * np.pi)
    d_a = (np.radians(th_a) - support.theta_k) % (2 * np.pi)
    front = (d > np.radians(FAST_WRAP.gather_half_deg) + 0.1) & (d < d_a - 0.1)
    order = np.argsort(d[front])
    assert np.all(np.diff(support.y[front][order]) <= 1e-6)
    assert np.all(support.y <= measured.neck_y.max()) and np.all(support.y >= measured.y_hem)


def test_loop_radius_is_off_the_body_everywhere(support, measured):
    r_body = np.array([np.interp(y, measured.y, measured.wrap_r[:, i]) for i, y in enumerate(support.y)])
    ok = np.isfinite(r_body)
    assert ok.sum() > len(ok) // 2
    assert np.all(support.r[ok] >= r_body[ok] - 1e-9)


def test_gather_is_one_outside_the_arc_and_the_ratio_inside(support):
    g = support.gather
    assert g.min() == pytest.approx(1.0)
    assert g.max() == pytest.approx(FAST_WRAP.gather_ratio, abs=1e-6)
    d = (support.theta - support.theta_k) % (2 * np.pi)
    far = (d > np.pi / 2) & (d < 3 * np.pi / 2)
    assert np.allclose(g[far], 1.0)
    i_k = int(np.argmin(np.abs(((support.theta - support.theta_k + np.pi) % (2 * np.pi)) - np.pi)))
    assert g[i_k] == pytest.approx(FAST_WRAP.gather_ratio, abs=1e-6)


def test_loop_at_returns_points_on_the_loop(support):
    P = W.loop_at(support, support.theta[:5])
    assert P.shape == (5, 3)
    assert np.allclose(P[:, 1], support.y[:5])
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/meshforge/test_wrap.py -q 2>&1 | tail -3`
Expected: FAIL — `ModuleNotFoundError: No module named 'meshforge.wrap'`.

- [ ] **Step 3: Implement `meshforge/wrap.py` (part 1)**

```python
"""The kente wrap: cloth hung from a measured support loop and tied over
the raised wing.

Where the tunic hung from the neckline, the wrap hangs from a *diagonal*
loop: high at the raised wing's shoulder (the knot), low under the tablet
wing (the cloth passes under the arm), rising again across the back. Every
point of that loop is read off the body at build time — collar rim, wing
root ring, the fused wing's lower contour — so the garment is rebuilt,
not re-fitted, when the owl underneath changes.

  measure_support   the loop L(theta), its radius off the body, the knot
                    frame and the gather density.
  build_kente_wrap  sheet (a tube hung from the loop, gathered into the
                    knot), tail (a strip pinned behind the knot), knot
                    (meshforge.knot), baked onto the pattern chart.
  support_weights   skin weights from the support graph: what holds the
                    cloth up, never what happens to be nearest.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from meshforge import drape as drapemod
from meshforge import robe as robemod
from meshforge.drape import BodyMeasure, ClothParams, Collider, Pattern, RadialBound
from meshforge.regions import RegionMap
from meshforge.rigexport import MaterialSpec, PrimitiveSpec
from meshforge.textile import WeaveParams

TWO_PI = 2.0 * np.pi


@dataclasses.dataclass(frozen=True)
class WrapParams:
    """Fractions of the body height H unless stated. Tuned on the AI-CCORE owl."""
    knot_drop: float = 0.015
    gather_half_deg: float = 20.0
    gather_ratio: float = 1.3
    arm_margin: float = 0.012
    hem_frac: float = 0.125
    loop_smooth_cols: float = 1.0
    root_band: float = 0.03
    knot_size: float = 0.07
    knot_strands: int = 7
    tail_width_frac: float = 0.22
    tail_end_frac: float = 0.15
    tail_span_deg: float = 25.0
    tail_offset_deg: float = 30.0
    tail_n_u: int = 30
    tail_n_v: int = 80
    tail_texture_size: tuple[int, int] = (1024, 512)
    tail_strips: int = 6
    image_format: str = "PNG"
    raised_wing: str = "wing_left"
    tablet_wing: str = "wing_right"
    material_name: str = "owl_kente_wrap"
    tail_material_name: str = "owl_kente_wrap_tail"


class Support(NamedTuple):
    theta: np.ndarray
    y: np.ndarray
    r: np.ndarray
    gather: np.ndarray
    theta_k: float
    y_k: float
    arc: tuple[float, float]
    frame: dict
    root_ring: np.ndarray
    low: np.ndarray
    info: dict


def _wrap_angle(a):
    return (np.asarray(a, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def _angle_of(P: np.ndarray, axis_xz: np.ndarray) -> np.ndarray:
    return np.arctan2(P[:, 0] - axis_xz[0], P[:, 2] - axis_xz[1])


def root_ring(mesh: trimesh.Trimesh, labels: np.ndarray, wing: str, band: float) -> np.ndarray:
    """The wing's vertices within `band` of the torso: where it grows out
    of the body. Found by distance, not adjacency, so a wing that is a
    separate shell in the mesh (the synthetic fin) has a root too."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    labels = np.asarray(labels)
    w = labels == wing
    if not w.any():
        raise ValueError(f"root_ring: no vertices labelled {wing!r}")
    torso = np.isin(labels, ["chest", "body", "neck"])
    d = cKDTree(V[torso]).query(V[w], workers=-1)[0]
    ring = V[w][d <= band]
    if len(ring) < 8:
        ring = V[w][np.argsort(d)[:max(8, int(0.1 * w.sum()))]]
    return ring


def _lower_boundary(mesh: trimesh.Trimesh, labels: np.ndarray, wing: str, theta: np.ndarray,
                    axis_xz: np.ndarray, min_count: int = 5) -> np.ndarray:
    """Per scan angle, the 5th percentile height of the wing's vertices in
    that angular cell — the fused wing's lower contour. nan where the wing
    is not."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    w = np.asarray(labels) == wing
    out = np.full(len(theta), np.nan)
    if not w.any():
        return out
    th = _angle_of(V[w], axis_xz)
    y = V[w][:, 1]
    edges = np.concatenate([[-np.pi], 0.5 * (theta[1:] + theta[:-1]), [np.pi]])
    cell = np.clip(np.searchsorted(edges, th) - 1, 0, len(theta) - 1)
    for i in range(len(theta)):
        m = cell == i
        if m.sum() >= min_count:
            out[i] = np.percentile(y[m], 5)
    return out


def measure_support(mesh: trimesh.Trimesh, region_map: RegionMap, body: BodyMeasure,
                    params: WrapParams) -> Support:
    """The loop the wrap hangs from, read off the body."""
    labels = np.asarray(region_map.labels)
    H = body.size_y
    theta = body.theta
    n = len(theta)

    # --- the knot: on the front edge of the raised wing's root, under the rim
    ring = root_ring(mesh, labels, params.raised_wing, params.root_band * H)
    th_ring = _angle_of(ring, body.axis_xz)
    mean = float(np.angle(np.exp(1j * th_ring).mean()))
    half = float(np.percentile(np.abs(_wrap_angle(th_ring - mean)), 95))
    cands = [mean + half, mean - half]
    theta_k = float(_wrap_angle(min(cands, key=lambda a: abs(float(_wrap_angle(a))))))   # the end nearer the face
    rim = float(np.interp(theta_k, theta, body.neck_y, period=TWO_PI))
    y_k = min(rim - params.knot_drop * H, float(ring[:, 1].max()))
    gh = np.radians(params.gather_half_deg)

    # --- the tablet wing's lower contour, and where it starts and ends
    low = _lower_boundary(mesh, labels, params.tablet_wing, theta, body.axis_xz)
    valid = np.isfinite(low)
    if not valid.any():
        raise ValueError(f"measure_support: no {params.tablet_wing!r} to pass the cloth under")
    d = (theta - theta_k) % TWO_PI                       # distance forward from the knot, [0, 2pi)
    i_a = int(np.flatnonzero(valid)[np.argmin(d[valid])])
    i_b = int(np.flatnonzero(valid)[np.argmax(d[valid])])
    d_a, d_b = float(d[i_a]), float(d[i_b])
    low_f = np.interp(d, d[valid][np.argsort(d[valid])], low[valid][np.argsort(d[valid])])   # gaps filled along d
    y_a = float(low[i_a]) - params.arm_margin * H
    y_b = float(low[i_b]) - params.arm_margin * H

    # --- the loop
    y = np.empty(n)
    for i in range(n):
        di = d[i]
        if di <= gh or di >= TWO_PI - gh:
            y[i] = y_k
        elif di < d_a:
            y[i] = y_k + (y_a - y_k) * (di - gh) / max(d_a - gh, 1e-6)
        elif di <= d_b:
            y[i] = low_f[i] - params.arm_margin * H
        else:
            y[i] = y_b + (y_k - y_b) * (di - d_b) / max((TWO_PI - gh) - d_b, 1e-6)
    y = np.clip(y, body.y_hem + 0.05 * H, float(body.neck_y.max()))
    y = gaussian_filter1d(y, sigma=max(params.loop_smooth_cols, 1e-3), mode="wrap")

    # --- its radius: the body's own at that height, plus the solver's air
    r_body = np.array([np.interp(y[i], body.y, body.wrap_r[:, i]) for i in range(n)])
    r_body = np.where(np.isfinite(r_body), r_body, np.nanmax(r_body))
    r = gaussian_filter1d(r_body, sigma=1.0, mode="wrap") + 0.014
    r = np.maximum(r, r_body)

    # --- gather density: the ratio inside the arc, smooth shoulders outside
    dd = np.abs(_wrap_angle(theta - theta_k))
    shoulder = np.radians(5.0)
    t = np.clip((dd - gh) / shoulder, 0.0, 1.0)
    gather = 1.0 + (params.gather_ratio - 1.0) * (1.0 - t * t * (3 - 2 * t))

    # --- the knot frame: surface point, its normal, the loop's tangent
    i_k = int(np.argmin(dd))
    origin = np.array([body.axis_xz[0] + r[i_k] * np.sin(theta_k), y_k, body.axis_xz[1] + r[i_k] * np.cos(theta_k)])
    _, _, fid = mesh.nearest.on_surface(origin[None, :])
    normal = np.asarray(mesh.face_normals[int(fid[0])], dtype=np.float64)
    radial = np.array([np.sin(theta_k), 0.0, np.cos(theta_k)])
    if np.dot(normal, radial) < 0:
        normal = -normal
    tangent = np.array([np.cos(theta_k), 0.0, -np.sin(theta_k)])     # along the loop, +theta direction
    tangent -= normal * np.dot(tangent, normal)
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    up = np.cross(normal, tangent)
    if up[1] < 0:
        up, tangent = -up, -tangent
    frame = {"origin": origin, "normal": normal / np.linalg.norm(normal), "tangent": tangent, "up": up}

    arc = (float(_wrap_angle(theta_k - gh)), float(_wrap_angle(theta_k + gh)))
    info = {
        "theta_k_deg": round(np.degrees(theta_k), 1), "y_k_frac": round((y_k - body.y0) / H, 3),
        "rim_frac_at_knot": round((rim - body.y0) / H, 3),
        "theta_a_deg": round(float(np.degrees(theta[i_a])), 1), "theta_b_deg": round(float(np.degrees(theta[i_b])), 1),
        "y_a_frac": round((y_a - body.y0) / H, 3), "y_b_frac": round((y_b - body.y0) / H, 3),
        "loop_frac": [round((y.min() - body.y0) / H, 3), round((y.max() - body.y0) / H, 3)],
        "root_ring_vertices": int(len(ring)),
        "root_ring_theta_deg": [round(float(np.degrees(mean - half)), 1), round(float(np.degrees(mean + half)), 1)],
    }
    return Support(theta=theta, y=y, r=r, gather=gather, theta_k=theta_k, y_k=y_k, arc=arc, frame=frame,
                   root_ring=ring, low=low, info=info)


def loop_at(support: Support, theta: np.ndarray, axis_xz: np.ndarray | None = None) -> np.ndarray:
    """World points on the loop at the given angles (periodic interpolation).
    `axis_xz` defaults to the one stored in `support.info`; callers inside
    this module pass the body's."""
    theta = np.asarray(theta, dtype=np.float64)
    ax = np.asarray(support.info.get("axis_xz", [0.0, 0.0]) if axis_xz is None else axis_xz, dtype=np.float64)
    y = np.interp(theta, support.theta, support.y, period=TWO_PI)
    r = np.interp(theta, support.theta, support.r, period=TWO_PI)
    return np.stack([ax[0] + r * np.sin(theta), y, ax[1] + r * np.cos(theta)], axis=1)
```

Add `info["axis_xz"] = [float(body.axis_xz[0]), float(body.axis_xz[1])]` inside `measure_support` before building `Support` (so `loop_at` works without the body).

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/meshforge/test_wrap.py -q 2>&1 | tail -3`
Expected: 6 passed. If `test_loop_dips_under...` fails on monotonicity, check the `gaussian_filter1d` sigma (1 column) — the diagonal is linear in d, the smoothing only rounds the corners; loosen the check to `<= 1e-3` only if the failure is at the corner columns.

- [ ] **Step 5: Full suite, then stop (no commit unless asked)**

Run: `python3 -m pytest tests/meshforge -q 2>&1 | tail -2`

---

### Task 3: The procedural knot (K1) — `meshforge/knot.py`

**Files:**
- Create: `meshforge/knot.py`
- Test: `tests/meshforge/test_knot.py`

**Interfaces:**
- Consumes: a frame dict (`origin`, `normal`, `tangent`, `up`, unit vectors), arc points (k, 3) on the loop, a `MaterialSpec` (the sheet's — the knot shares its texture).
- Produces: `build_knot(frame, arc_points, size, material, n_strands=7, stations=16, segments=6, seed=0) -> Knot`; `Knot(primitive: PrimitiveSpec, base_points: (n_strands, 3), info: dict)`. The primitive's `joints`/`weights` are left `None` for the caller.

- [ ] **Step 1: Write the failing test** — create `tests/meshforge/test_knot.py`:

```python
"""The designed knot: a bunch of twisted strands over a core, two short ends."""
import numpy as np

from meshforge import knot as K
from meshforge.rigexport import MaterialSpec

FRAME = {"origin": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0]),
         "tangent": np.array([1.0, 0.0, 0.0]), "up": np.array([0.0, 1.0, 0.0])}


def _arc(n=24, half=0.1):
    return np.stack([np.linspace(-half, half, n), np.zeros(n), np.zeros(n)], axis=1)


def test_knot_is_a_valid_primitive_on_its_frame():
    knot = K.build_knot(FRAME, _arc(), size=0.1, material=MaterialSpec(name="m"))
    p = knot.primitive
    assert p.faces.min() >= 0 and p.faces.max() < len(p.vertices)
    assert np.allclose(np.linalg.norm(p.normals, axis=1), 1.0, atol=1e-6)
    assert p.uvs.shape == (len(p.vertices), 2) and p.uvs.min() >= 0.0 and p.uvs.max() <= 1.0
    assert p.vertices[:, 2].min() > -0.03            # at most a strand radius behind the surface
    assert p.vertices[:, 2].max() < 0.25 and np.abs(p.vertices[:, 0]).max() < 0.3
    assert p.vertices[:, 1].min() > -0.2             # the ends hang about one size below the arc, no further
    assert knot.base_points.shape == (7, 3)
    assert np.allclose(knot.base_points[:, 1:], 0.0, atol=1e-9)   # the strands start on the arc
    assert 800 < len(p.faces) < 4000
    assert p.material.name == "m"


def test_knot_scales_with_size_and_strands():
    small = K.build_knot(FRAME, _arc(), size=0.05, material=MaterialSpec(name="m"), n_strands=4, stations=10)
    big = K.build_knot(FRAME, _arc(), size=0.10, material=MaterialSpec(name="m"))
    assert len(small.primitive.faces) < len(big.primitive.faces)
    assert small.primitive.vertices[:, 2].max() < big.primitive.vertices[:, 2].max()
    assert small.base_points.shape == (4, 3)


def test_tube_sweeps_a_ring_along_a_path():
    path = np.stack([np.linspace(0, 1, 5), np.zeros(5), np.zeros(5)], axis=1)
    V, F, UV = K._tube(path, np.full(5, 0.1), 6, along=(0.0, 1.0), around=(0.0, 1.0))
    assert V.shape == (30, 3) and F.shape == (48, 3) and UV.shape == (30, 2)
    assert np.allclose(np.linalg.norm(V[:6, 1:], axis=1), 0.1)      # the first ring is a circle of radius 0.1 about the path
    assert np.allclose(V[:6, 0], 0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/meshforge/test_knot.py -q 2>&1 | tail -3`
Expected: FAIL — `ModuleNotFoundError: No module named 'meshforge.knot'`.

- [ ] **Step 3: Implement `meshforge/knot.py`**

```python
"""A designed knot: the bunch where the wrap is tied over the raised wing.

K1, procedural: strands of cloth leave the base arc (where the sheet's
gathered columns enter the knot), loop over a core and return to the arc
mirrored, so the bunch reads as twisted cloth; two short flat ends hang
from the top. Everything is sized from `size` (the bulge height, a
fraction of the body height chosen by the caller) and the arc's length, and
placed in the knot frame measured off the body — nothing here is a stored
coordinate. K2 (a TRELLIS-generated prop) is the same interface, ticket
W-K2.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh

from meshforge.rigexport import MaterialSpec, PrimitiveSpec


class Knot(NamedTuple):
    primitive: PrimitiveSpec
    base_points: np.ndarray      # (n_strands, 3) where the strands leave the arc
    info: dict


def _bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _tube(path: np.ndarray, radius: np.ndarray, segments: int, along: tuple[float, float],
          around: tuple[float, float], flatten: float = 1.0):
    """Sweep a circle (an ellipse when `flatten` < 1) along `path` with
    parallel-transport frames. Returns (verts (k*segments, 3), faces, uvs)
    with the chart's u taken from `around` and v from `along`: the woven
    strips run along the strand, as they do on cloth twisted into one."""
    path = np.asarray(path, dtype=np.float64)
    k = len(path)
    T = np.gradient(path, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    N = np.zeros_like(T)
    a = np.array([0.0, 1.0, 0.0]) if abs(T[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N[0] = np.cross(T[0], a)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, k):
        n = N[i - 1] - T[i] * np.dot(T[i], N[i - 1])
        N[i] = n / max(np.linalg.norm(n), 1e-12)
    B = np.cross(T, N)
    ph = np.arange(segments) * (2 * np.pi / segments)
    ring = np.cos(ph)[:, None, None] * N[None] + flatten * np.sin(ph)[:, None, None] * B[None]   # (seg, k, 3)
    verts = (path[None] + np.asarray(radius)[None, :, None] * ring).transpose(1, 0, 2).reshape(-1, 3)
    i = np.arange(k - 1)[:, None]
    j = np.arange(segments)[None, :]
    a_ = i * segments + j
    b_ = i * segments + (j + 1) % segments
    c_ = a_ + segments
    d_ = b_ + segments
    faces = np.concatenate([np.stack([a_, b_, d_], -1).reshape(-1, 3), np.stack([a_, d_, c_], -1).reshape(-1, 3)])
    v = along[0] + (along[1] - along[0]) * np.repeat(np.linspace(0.0, 1.0, k), segments)
    u = around[0] + (around[1] - around[0]) * np.tile(np.arange(segments) / segments, k)
    return verts, faces, np.stack([u, v], axis=1)


def build_knot(frame: dict, arc_points: np.ndarray, size: float, material: MaterialSpec,
               n_strands: int = 7, stations: int = 16, segments: int = 6, seed: int = 0) -> Knot:
    rng = np.random.default_rng(seed)
    o, nrm, tan, up = (np.asarray(frame[k], dtype=np.float64) for k in ("origin", "normal", "tangent", "up"))
    arc = np.asarray(arc_points, dtype=np.float64)
    arc_len = float(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum())
    idx = np.linspace(0, len(arc) - 1, n_strands).round().astype(int)
    starts = arc[idx]
    centre = o + nrm * (0.45 * size) + up * (0.10 * size)
    parts = []
    # strands: leave the arc, over the core, back to the mirrored point
    for s in range(n_strands):
        p0, p3 = starts[s], starts[n_strands - 1 - s]
        lift = 0.9 * size * (1 + 0.25 * rng.uniform(-1, 1))
        c1 = p0 + nrm * lift + up * (0.35 * size * rng.uniform(-1, 1)) + tan * (0.10 * size * rng.uniform(-1, 1))
        c2 = p3 + nrm * lift + up * (0.35 * size * rng.uniform(-1, 1)) + tan * (0.10 * size * rng.uniform(-1, 1))
        path = _bezier(p0, c1, c2, p3, stations)
        r = 0.09 * size * (1 + 0.3 * rng.uniform(-1, 1)) * (0.6 + 0.4 * np.sin(np.linspace(0, np.pi, stations)))
        u0 = 0.30 + 0.05 * (s % 6)
        parts.append(_tube(path, r, segments, along=(0.05, 0.60), around=(u0, u0 + 0.04)))
    # the core under the strands
    core = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    local = np.asarray(core.vertices) * np.array([0.55 * arc_len, 0.45 * size, 0.5 * size])
    R = np.stack([tan, up, nrm], axis=1)
    cv = centre + local @ R.T
    cuv = np.stack([0.20 + 0.10 * (np.arctan2(local[:, 2], local[:, 0]) / (2 * np.pi) + 0.5),
                    0.10 + 0.30 * (local[:, 1] / max(size, 1e-9) + 0.5)], axis=1)
    parts.append((cv, np.asarray(core.faces), np.clip(cuv, 0.0, 1.0)))
    # two short flat ends hanging from the top
    for sgn in (-1.0, 1.0):
        p0 = centre + tan * (sgn * 0.25 * size) + up * (0.30 * size)
        p3 = p0 + nrm * (0.35 * size) - up * (1.10 * size) + tan * (sgn * 0.15 * size)
        path = _bezier(p0, p0 + nrm * (0.4 * size), p3 + up * (0.3 * size), p3, 12)
        r = 0.16 * size * np.linspace(1.0, 0.55, 12)
        u0 = 0.40 if sgn < 0 else 0.46
        parts.append(_tube(path, r, segments, along=(0.05, 0.45), around=(u0, u0 + 0.05), flatten=0.35))
    verts = np.concatenate([p[0] for p in parts])
    offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    faces = np.concatenate([p[1] + off for p, off in zip(parts, offs)])
    uvs = np.concatenate([p[2] for p in parts])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = np.asarray(m.vertex_normals, dtype=np.float64).copy()
    bad = np.linalg.norm(normals, axis=1) < 0.5
    normals[bad] = nrm
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    prim = PrimitiveSpec(name="kente_knot", vertices=verts, faces=faces, normals=normals, uvs=uvs, material=material)
    info = {"faces": int(len(faces)), "vertices": int(len(verts)), "strands": n_strands,
            "size": round(float(size), 4), "arc_length": round(arc_len, 4)}
    return Knot(primitive=prim, base_points=starts, info=info)
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/meshforge/test_knot.py -q 2>&1 | tail -3`
Expected: 3 passed. If the face count is outside (800, 4000), adjust `stations`/`segments` defaults, not the test.

- [ ] **Step 5: Look at it once** (the only visual step before the owl build)

```bash
python3 - <<'PY'
import numpy as np, trimesh
from meshforge import knot as K
from meshforge.rigexport import MaterialSpec
f = {"origin": np.zeros(3), "normal": np.array([0,0,1.]), "tangent": np.array([1.,0,0]), "up": np.array([0,1.,0])}
arc = np.stack([np.linspace(-0.1, 0.1, 24), np.zeros(24), np.zeros(24)], 1)
k = K.build_knot(f, arc, 0.1, MaterialSpec(name="m"))
trimesh.Trimesh(k.primitive.vertices, k.primitive.faces).export("out/owl_kente/knot_k1.glb")
PY
python3 tools/preview.py out/owl_kente/knot_k1.glb --view face --size 480 --no-skin --out out/owl_kente/knot_k1.png
```
Open `out/owl_kente/knot_k1.png` (Read tool): strands over a core, two ends below. Then stop (no commit unless asked).

---

### Task 4: `build_kente_wrap` — sheet, tail, knot

**Files:**
- Modify: `meshforge/wrap.py` (append)
- Test: `tests/meshforge/test_wrap.py` (append)

**Interfaces:**
- Consumes: Task 1's `draft(..., gather=)`, `Pattern.theta_cols`, `periodic=` flags, `_swing_bounds(..., with_neck=False)`; Task 2's `measure_support`, `loop_at`; Task 3's `build_knot`; `drape._bake`, `drape._grid_normals`, `drape.held_out_vertices`, `drape.Collider`, `drape.push_out`, `drape.seam_columns`.
- Produces:

```python
class KenteWrap(NamedTuple):
    sheet: PrimitiveSpec; tail: PrimitiveSpec; knot: PrimitiveSpec
    sheet_grid: tuple[int, int]; sheet_pattern: Pattern; sheet_sets: list
    tail_grid: tuple[int, int]; tail_pattern: Pattern; tail_sets: list
    support: Support; body: BodyMeasure
    pins: dict          # {"sheet": (cols, 3), "tail": (tail_cols, 3)} — the pinned top rows
    info: dict

def grid_faces(rows: int, cols: int) -> np.ndarray                      # (2*(rows-1)*(cols-1), 3)
def build_kente_wrap(mesh, region_map, weave_params, cloth_params=ClothParams(), params=WrapParams(),
                     band_frame=None, sweep_vertices=None, n_theta=144) -> KenteWrap
```

  Vertex order of `sheet` is the full grid (`row * cols + col`, no cut); of `tail` the full strip grid (`row * tail_n_u + col`). `sheet.material is knot.material`.

- [ ] **Step 1: Write the failing tests** — append to `tests/meshforge/test_wrap.py`:

```python
from meshforge import robe as R


@pytest.fixture(scope="module")
def dressed(body):
    mesh, rm = body
    return W.build_kente_wrap(mesh, rm, WEAVE, small_params(), FAST_WRAP, n_theta=48), mesh


def test_wrap_keeps_out_of_the_body(dressed):
    wrap, mesh = dressed
    for prim in (wrap.sheet, wrap.tail):
        assert R.clearance(prim.vertices, mesh)["inside_count"] == 0, prim.name


def test_sheet_hangs_from_the_loop_and_reaches_the_hem(dressed):
    wrap, _ = dressed
    rows, cols = wrap.sheet_grid
    top = wrap.sheet.vertices[:cols]
    assert np.allclose(top, wrap.pins["sheet"])
    loop = W.loop_at(wrap.support, wrap.sheet_pattern.theta_cols, wrap.body.axis_xz)
    assert np.linalg.norm(top - loop, axis=1).max() < 0.03 * wrap.body.size_y   # on the loop, pushed out of the fin's edges
    hem = wrap.sheet.vertices[-cols:]
    assert abs((hem[:, 1].mean() - wrap.body.y0) / wrap.body.size_y - FAST_WRAP.hem_frac) < 0.05


def test_tail_hangs_behind_the_knot(dressed):
    wrap, _ = dressed
    rows, cols = wrap.tail_grid
    top = wrap.tail.vertices[:cols]
    assert np.allclose(top, wrap.pins["tail"])
    assert wrap.tail.vertices[-cols:, 1].max() < wrap.support.y_k - 0.3 * wrap.tail_pattern.length
    th = np.arctan2(top[:, 0] - wrap.body.axis_xz[0], top[:, 2] - wrap.body.axis_xz[1])
    d = (th - wrap.support.theta_k) % (2 * np.pi)
    assert np.all(d > np.pi)                          # on the knot's back side


def test_wrap_primitives_are_valid(dressed):
    wrap, _ = dressed
    for prim in (wrap.sheet, wrap.tail, wrap.knot):
        assert prim.faces.min() >= 0 and prim.faces.max() < len(prim.vertices), prim.name
        assert np.allclose(np.linalg.norm(prim.normals, axis=1), 1.0, atol=1e-5), prim.name
        assert prim.uvs.min() >= -1e-9 and prim.uvs.max() <= 1 + 1e-9, prim.name
    assert wrap.sheet.material.name == "owl_kente_wrap" and wrap.knot.material is wrap.sheet.material
    assert wrap.tail.material.name == "owl_kente_wrap_tail"
    assert wrap.sheet.material.base_color_image.size == (256, 128)
    assert wrap.tail.material.base_color_image.size == (128, 64)


def test_wrap_cuts_nothing(dressed):
    wrap, _ = dressed
    rows, cols = wrap.sheet_grid
    assert len(wrap.sheet.vertices) == rows * cols
    assert len(wrap.sheet.faces) == 2 * (rows - 1) * (cols - 1)
    assert np.allclose(wrap.sheet.vertices[cols - 1::cols], wrap.sheet.vertices[0::cols])   # seam welded


def test_knot_sits_on_the_knot_frame(dressed):
    wrap, _ = dressed
    o = wrap.support.frame["origin"]
    assert np.linalg.norm(wrap.knot.vertices.mean(axis=0) - o) < 0.3 * wrap.body.size_y
    assert wrap.info["knot"]["strands"] == FAST_WRAP.knot_strands


def test_grid_faces_tile_the_grid():
    F = W.grid_faces(3, 4)
    assert F.shape == (12, 3) and F.min() == 0 and F.max() == 11
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/meshforge/test_wrap.py -q 2>&1 | tail -3`
Expected: FAIL — `AttributeError: module 'meshforge.wrap' has no attribute 'build_kente_wrap'`.

- [ ] **Step 3: Implement** — append to `meshforge/wrap.py`:

```python
# --------------------------------------------------------------------------
# the garment
# --------------------------------------------------------------------------

class KenteWrap(NamedTuple):
    sheet: PrimitiveSpec
    tail: PrimitiveSpec
    knot: PrimitiveSpec
    sheet_grid: tuple[int, int]
    sheet_pattern: Pattern
    sheet_sets: list
    tail_grid: tuple[int, int]
    tail_pattern: Pattern
    tail_sets: list
    support: Support
    body: BodyMeasure
    pins: dict
    info: dict


def grid_faces(rows: int, cols: int) -> np.ndarray:
    """Two triangles per cell of a (rows, cols) vertex grid, row-major."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).ravel()
    b, cc, d = a + 1, a + cols, a + cols + 1
    return np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, cc], axis=1)])


def _strip_pattern(width: float, length: float, rows: int, cols: int) -> Pattern:
    phi = np.tile(np.linspace(0.0, 1.0, cols)[None, :], (rows, 1))
    v = np.linspace(0.0, length, rows)[:, None] * np.ones((1, cols))
    return Pattern(phi=phi, v=v, circumference=np.array([width, width]), v_profile=np.array([0.0, length]),
                   length=length, fall=np.full(cols, length), grid_shape=(rows, cols), theta_cols=None)


def _face_out(P: np.ndarray, faces: np.ndarray, normals: np.ndarray, body: BodyMeasure):
    radial = P - np.array([body.axis_xz[0], 0.0, body.axis_xz[1]])
    radial[:, 1] = 0.0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    if float(np.mean(np.sum(normals * radial, axis=1))) < 0:
        return faces[:, ::-1], -normals
    return faces, normals


def build_kente_wrap(mesh: trimesh.Trimesh, region_map: RegionMap, weave_params: WeaveParams,
                     cloth_params: ClothParams = ClothParams(), params: WrapParams = WrapParams(),
                     band_frame=None, sweep_vertices: Sequence[np.ndarray] | None = None,
                     n_theta: int = 144) -> KenteWrap:
    """Measure the loop, hang the sheet from it (gathered into the knot),
    hang the tail behind the knot, place the knot, bake the weave."""
    for name, strips in (("strips_around", cloth_params.strips_around), ("tail_strips", params.tail_strips)):
        if strips % weave_params.strip_cycle != 0:
            raise ValueError(f"build_kente_wrap: {name} ({strips}) must be a multiple of the weave's strip_cycle "
                             f"({weave_params.strip_cycle})")
    labels = np.asarray(region_map.labels)
    cloth = dataclasses.replace(cloth_params, hem_frac=params.hem_frac, material_name=params.material_name)
    body = drapemod.measure_body(mesh, region_map, cloth, band_frame=band_frame, n_theta=n_theta)
    support = measure_support(mesh, region_map, body, params)
    H = body.size_y
    hung = body._replace(neck_y=support.y, neck_r=support.r)       # the loop is the neckline now
    bounds = drapemod._swing_bounds(mesh, labels, body, cloth, sweep_vertices, with_neck=False)
    protruding = drapemod.held_out_vertices(mesh, labels, body.scan, body.cells, body.axis_xz, cloth.hold_tolerance)
    collider = Collider(mesh, face_mask=~protruding[mesh.faces].any(axis=1))   # the cloth passes behind what is held out

    # --- the sheet: a tube hung from the loop, gathered into the knot
    pattern = drapemod.draft(hung, cloth, bounds=bounds, gather=support.gather)
    rows, cols = pattern.grid_shape
    n = rows * cols
    keep = np.ones(n, dtype=bool)
    pinned = np.zeros(n, dtype=bool)
    pinned[:cols] = True
    P0 = drapemod.initial_positions(pattern, hung)
    v_rows = pattern.v_profile
    sets = drapemod.constraint_sets(pattern.phi.ravel(), pattern.v.ravel(), pattern.circumference, v_rows,
                                    keep, rows, cols, cloth)
    tet = drapemod.build_tethers(pattern.phi.ravel(), pattern.v.ravel(), pattern.circumference, v_rows,
                                 keep, pinned, rows, cols)
    grade = lambda v: cloth.collide_offset + (cloth.collide_offset_hem - cloth.collide_offset) * np.clip(
        v / max(pattern.length, 1e-9), 0.0, 1.0) ** 1.4
    P, d1 = drapemod.drape(P0, pattern, sets, [collider], pinned, keep, cloth,
                           offset=grade(pattern.v.ravel()), tethers=tet, bounds=bounds)
    d1["pushed_out"] = drapemod.push_out(P, keep, collider, cloth.collide_offset * 0.6)
    d1["swept_clear"] = sum(b.apply(P, free=keep) for b in bounds)
    seam_src, seam_dup = drapemod.seam_columns(rows, cols, keep)
    P[seam_dup] = P[seam_src]
    faces = grid_faces(rows, cols)
    F = np.minimum(pattern.v, pattern.fall[None, :] - pattern.v)            # distance to the top edge or the hem
    uvs = np.stack([pattern.phi.ravel(), pattern.v.ravel() / max(pattern.length, 1e-9)], axis=1)
    uvs[np.arange(n) % cols == cols - 1, 0] = 1.0
    normals = drapemod._grid_normals(P, faces, np.arange(n), cols, n)
    faces, normals = _face_out(P, faces, normals, body)
    img, normal_img = drapemod._bake(P, faces, uvs, normals, pattern, F, weave_params, cloth, body)
    sheet_mat = MaterialSpec(name=params.material_name, base_color_image=img, image_format=params.image_format,
                             roughness=cloth.roughness, metallic=0.0, double_sided=True, normal_image=normal_img)
    sheet = PrimitiveSpec(name="kente_wrap", vertices=P, faces=faces, normals=normals, uvs=uvs, material=sheet_mat)

    # --- the tail: a strip pinned behind the knot, hanging down the back
    rows_t, cols_t = params.tail_n_v, params.tail_n_u
    width = params.tail_width_frac * H
    y_end = body.y0 + params.tail_end_frac * H
    length = max(support.y_k - y_end, 0.05 * H) * (1.0 + cloth.length_slack)
    tpat = _strip_pattern(width, length, rows_t, cols_t)
    gh = np.radians(params.gather_half_deg)
    th_hi = support.theta_k - gh - np.radians(params.tail_offset_deg)
    th_cols = th_hi - np.radians(params.tail_span_deg) * (1.0 - np.linspace(0.0, 1.0, cols_t))
    pins_t = loop_at(support, th_cols, body.axis_xz)
    pins_t[:, 1] = support.y_k
    radial = np.stack([np.sin(th_cols), np.zeros(cols_t), np.cos(th_cols)], axis=1)
    pins_t += radial * (0.01 * H)
    direction = radial * np.sin(np.radians(15.0)) + np.array([0.0, -1.0, 0.0]) * np.cos(np.radians(15.0))
    P0t = (pins_t[None, :, :] + tpat.v[:, :, None] * direction[None, :, :]).reshape(-1, 3)
    nt = rows_t * cols_t
    keep_t = np.ones(nt, dtype=bool)
    pinned_t = np.zeros(nt, dtype=bool)
    pinned_t[:cols_t] = True
    sets_t = drapemod.constraint_sets(tpat.phi.ravel(), tpat.v.ravel(), tpat.circumference, tpat.v_profile,
                                      keep_t, rows_t, cols_t, cloth, periodic=False)
    tet_t = drapemod.build_tethers(tpat.phi.ravel(), tpat.v.ravel(), tpat.circumference, tpat.v_profile,
                                   keep_t, pinned_t, rows_t, cols_t, periodic=False)
    Pt, d2 = drapemod.drape(P0t, tpat, sets_t, [collider], pinned_t, keep_t, cloth,
                            offset=cloth.collide_offset_hem, tethers=tet_t, periodic=False)
    d2["pushed_out"] = drapemod.push_out(Pt, keep_t, collider, cloth.collide_offset_hem * 0.6)
    faces_t = grid_faces(rows_t, cols_t)
    Ft = np.minimum(np.minimum(tpat.v, length - tpat.v), np.minimum(tpat.phi, 1.0 - tpat.phi) * width)
    uvs_t = np.stack([tpat.phi.ravel(), tpat.v.ravel() / length], axis=1)
    normals_t = drapemod._grid_normals(Pt, faces_t, np.arange(nt), cols_t, nt, periodic=False)
    faces_t, normals_t = _face_out(Pt, faces_t, normals_t, body)
    cloth_t = dataclasses.replace(cloth, texture_size=params.tail_texture_size, strips_around=params.tail_strips,
                                  material_name=params.tail_material_name)
    img_t, normal_t = drapemod._bake(Pt, faces_t, uvs_t, normals_t, tpat, Ft, weave_params, cloth_t, body)
    tail_mat = MaterialSpec(name=params.tail_material_name, base_color_image=img_t, image_format=params.image_format,
                            roughness=cloth.roughness, metallic=0.0, double_sided=True, normal_image=normal_t)
    tail = PrimitiveSpec(name="kente_wrap_tail", vertices=Pt, faces=faces_t, normals=normals_t, uvs=uvs_t, material=tail_mat)

    # --- the knot, on the base arc
    from meshforge.knot import build_knot
    lo, hi = support.arc
    arc_th = lo + ((hi - lo) % TWO_PI) * np.linspace(0.0, 1.0, 24)
    knot = build_knot(support.frame, loop_at(support, arc_th, body.axis_xz), size=params.knot_size * H,
                      material=sheet_mat, n_strands=params.knot_strands)

    hem = P[-cols:]
    info = {
        "support": support.info,
        "sheet": {"vertices": int(n), "faces": int(len(faces)), "grid": [rows, cols],
                  "pattern": {"circumference": [round(float(pattern.circumference[0]), 3), round(float(pattern.circumference[-1]), 3)],
                              "chest": round(float(pattern.circumference.max()), 3), "wrap": pattern.wrap_perimeter,
                              "profile": {"v": [round(float(x), 4) for x in pattern.v_profile],
                                          "circ": [round(float(x), 4) for x in pattern.circumference]},
                              "length": round(float(pattern.length), 3)},
                  "hem_frac_target": params.hem_frac,
                  "hem_frac_actual": [round(float((hem[:, 1].min() - body.y0) / H), 3), round(float((hem[:, 1].max() - body.y0) / H), 3)],
                  "width_by_height": drapemod._width_profile(P, body), "drape": d1, "image": img.size},
        "tail": {"vertices": int(nt), "faces": int(len(faces_t)), "grid": [rows_t, cols_t], "width": round(width, 3),
                 "length": round(float(length), 3), "pin_theta_deg": [round(float(np.degrees(th_cols[0])), 1), round(float(np.degrees(th_cols[-1])), 1)],
                 "end_frac_actual": round(float((Pt[:, 1].min() - body.y0) / H), 3), "drape": d2, "image": img_t.size},
        "knot": knot.info,
        "colorway": weave_params.colorway.name,
    }
    return KenteWrap(sheet=sheet, tail=tail, knot=knot.primitive, sheet_grid=(rows, cols), sheet_pattern=pattern,
                     sheet_sets=sets, tail_grid=(rows_t, cols_t), tail_pattern=tpat, tail_sets=sets_t,
                     support=support, body=body, pins={"sheet": P[:cols].copy(), "tail": pins_t.copy()}, info=info)
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/meshforge/test_wrap.py -q 2>&1 | tail -5`
Expected: 13 passed. Likely first failures and what they mean:
- `inside_count > 0` on the tail: the tail's initial positions started inside the fin; raise the radial lift (`pins_t += radial * 0.01 H`) to `0.02 * H` and re-run — never loosen the test.
- hem off by > 0.05: `measure_body`'s `y_hem` came from `cloth.hem_frac` — check `cloth = dataclasses.replace(cloth_params, hem_frac=params.hem_frac, ...)` ran before `measure_body`.
- `_bake` raising on the tail's `blocks`: `strips_around=6` with `strip_cycle=3` is a valid multiple; if `px_per_u` is tiny at the 128 px test texture, the weave is coarse but must not raise.

- [ ] **Step 5: Full suite, then stop (no commit unless asked)**

Run: `python3 -m pytest tests/meshforge -q 2>&1 | tail -2`

---

### Task 5: Skinning from the support graph — `support_weights`

**Files:**
- Modify: `meshforge/wrap.py` (append)
- Test: `tests/meshforge/test_wrap.py` (append)

**Interfaces:**
- Consumes: `robe.smooth_grid_weights(weights, grid_index, grid_shape, sigma, periodic)`.
- Produces:

```python
def support_weights(verts, grid_index, grid_shape, pins, body_V, body_W, joint_names, y_hip,
                    pin_weights=None, periodic=True, sigma=1.0) -> np.ndarray     # (n, n_joints), rows sum to 1
def forbidden_mass(W, joint_names, allowed=("body", "chest")) -> float
```

- [ ] **Step 1: Write the failing tests** — append to `tests/meshforge/test_wrap.py`:

```python
def _label_weights(rm):
    labels = np.asarray(rm.labels)
    names = sorted(set(labels.tolist()))
    return names, (labels[:, None] == np.array(names)[None, :]).astype(np.float64)


def test_support_weights_live_on_chest_and_body_and_end_on_body(dressed, body):
    wrap, mesh = dressed
    _, rm = body
    names, bw = _label_weights(rm)
    rows, cols = wrap.sheet_grid
    y_hip = wrap.body.y0 + 0.25 * wrap.body.size_y
    Wn = W.support_weights(wrap.sheet.vertices, np.arange(rows * cols), wrap.sheet_grid, wrap.pins["sheet"],
                           mesh.vertices, bw, names, y_hip)
    assert Wn.shape == (rows * cols, len(names))
    assert np.allclose(Wn.sum(axis=1), 1.0)
    assert W.forbidden_mass(Wn, names) == 0.0
    assert np.allclose(Wn[-cols:, names.index("body")], 1.0)               # the hem is body-driven
    top = Wn[:cols]
    assert np.allclose(top[:, names.index("chest")] + top[:, names.index("body")], 1.0)


def test_support_weights_take_the_given_pin_weights_for_a_strip(dressed, body):
    wrap, mesh = dressed
    _, rm = body
    names, bw = _label_weights(rm)
    rows, cols = wrap.tail_grid
    chest = np.zeros((cols, len(names)))
    chest[:, names.index("chest")] = 1.0
    Wt = W.support_weights(wrap.tail.vertices, np.arange(rows * cols), wrap.tail_grid, wrap.pins["tail"],
                           mesh.vertices, bw, names, y_hip=wrap.body.y0 + 0.25 * wrap.body.size_y,
                           pin_weights=chest, periodic=False)
    assert np.allclose(Wt[:cols, names.index("chest")], 1.0, atol=0.05)    # smoothing only blurs one row
    assert W.forbidden_mass(Wt, names) == 0.0
    assert Wt[:, names.index("body")].max() > 0.5                            # it does hand over to body further down


def test_forbidden_mass_counts_everything_off_the_torso():
    names = ["body", "chest", "wing_left"]
    Wn = np.array([[0.5, 0.5, 0.0], [0.2, 0.7, 0.1]])
    assert W.forbidden_mass(Wn, names) == pytest.approx(0.1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/meshforge/test_wrap.py -q -k "support_weights or forbidden" 2>&1 | tail -3`
Expected: FAIL — `AttributeError ... 'support_weights'`.

- [ ] **Step 3: Implement** — append to `meshforge/wrap.py`:

```python
# --------------------------------------------------------------------------
# skinning from the support graph
# --------------------------------------------------------------------------

def support_weights(verts: np.ndarray, grid_index: np.ndarray, grid_shape: tuple[int, int], pins: np.ndarray,
                    body_V: np.ndarray, body_W: np.ndarray, joint_names: Sequence[str], y_hip: float,
                    pin_weights: np.ndarray | None = None, periodic: bool = True, sigma: float = 1.0) -> np.ndarray:
    """Skin weights for a hung piece of cloth from what holds it up.

    The pinned row takes the body's own skin at its pin point, restricted to
    the torso joints (`chest`, `body`) and renormalised — the loop moves
    with the torso that carries it, never with the wing it passes under.
    Every column blends from its pin's weights to `body` by height, fully
    `body` below the hip: the skirt rides the root translation (the hop)
    and nothing else. No joint the cloth is not supported by gets any
    weight, whatever is nearest — which is the class-F failure of nearest-
    vertex transfer (the research volume's wrong animated baseline)."""
    joint_names = list(joint_names)
    jb, jc = joint_names.index("body"), joint_names.index("chest")
    rows, cols = grid_shape
    col = np.asarray(grid_index) % cols
    if pin_weights is None:
        near = cKDTree(np.asarray(body_V)).query(np.asarray(pins), workers=-1)[1]
        Wp = np.zeros((cols, len(joint_names)))
        Wp[:, jb] = body_W[near, jb]
        Wp[:, jc] = body_W[near, jc]
        s = Wp.sum(axis=1)
        Wp[s < 1e-6, jc] = 1.0
        Wp /= Wp.sum(axis=1, keepdims=True)
    else:
        Wp = np.asarray(pin_weights, dtype=np.float64)
        if Wp.shape != (cols, len(joint_names)):
            raise ValueError(f"support_weights: pin_weights must be ({cols}, {len(joint_names)}), got {Wp.shape}")
    y_top = np.asarray(pins)[:, 1]
    s = np.clip((y_top[col] - verts[:, 1]) / np.maximum(y_top[col] - y_hip, 1e-6), 0.0, 1.0)
    W = (1.0 - s)[:, None] * Wp[col]
    W[:, jb] += s
    W = robemod.smooth_grid_weights(W, np.asarray(grid_index), grid_shape, sigma=sigma, periodic=periodic)
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)


def forbidden_mass(W: np.ndarray, joint_names: Sequence[str], allowed: Sequence[str] = ("body", "chest")) -> float:
    """Total weight on joints the garment is not supported by (gate G2)."""
    cols = [i for i, nm in enumerate(joint_names) if nm not in allowed]
    return float(np.asarray(W)[:, cols].sum()) if cols else 0.0
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/meshforge/test_wrap.py -q 2>&1 | tail -3`
Expected: 16 passed.

- [ ] **Step 5: Full suite, then stop (no commit unless asked)**

Run: `python3 -m pytest tests/meshforge -q 2>&1 | tail -2`

---

### Task 6: Acceptance gates — `meshforge/gates.py`

**Files:**
- Create: `meshforge/gates.py`
- Test: `tests/meshforge/test_gates.py`

**Interfaces:**
- Consumes: `fk.skin_matrices`, `fk.world_matrices`, `fk.pose_vertices`, `robe.clip_pose_at`, `robe.posed_clearance`, `regions.collar_bounds_at`, `rigexport.AnimationSpec`/`Track`.
- Produces:

```python
FORBIDDEN: tuple[str, ...]
class Gate(NamedTuple): id: str; name: str; value: float; threshold: str; passed: bool; detail: dict
def clip_skins(joints, joint_names, clips, samples=5) -> list[tuple[str, float, dict]]
def gate_worn(pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H) -> Gate        # G1
def gate_support_only(W_all, joint_names) -> Gate                                                      # G2
def gate_penetration(posed: dict, limit=20) -> Gate                                                    # G3
def gate_hem(hem_points, y0, H, bead_top_y) -> Gate                                                    # G4
def gate_root_covered(root_ring, wing_pivot, wing_axis, garments: list[tuple[V, F]]) -> Gate           # G5
def gate_tail_clear(tail_V, tail_W, body_mesh, body_W, joints, joint_names, wave_clips) -> Gate        # G6
def gate_stretch(P, sets, stretch_k) -> Gate                                                           # G7
def gate_collar_visible(cloth_V, collar: dict, bmin, size) -> Gate                                     # G8
def run_gates(**kw) -> list[Gate]        # keyword names exactly as the pipeline passes them in Task 7
def format_table(gates: Sequence[Gate]) -> str
```

- [ ] **Step 1: Write the failing tests** — create `tests/meshforge/test_gates.py`:

```python
"""Gates are numbers with thresholds. Each test builds the smallest input
that makes one gate pass and one that makes it fail."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from meshforge import gates as G
from meshforge.drape import _Set
from meshforge.fk import Joint
from meshforge.rigexport import AnimationSpec, Track

NAMES = ["body", "chest", "wing_left"]
JOINTS = [Joint("body", None, np.array([0.0, 0.0, 0.0])), Joint("chest", "body", np.array([0.0, 0.5, 0.0])),
          Joint("wing_left", "chest", np.array([-0.3, 0.8, 0.0]))]


def _clip(joint, deg, axis="z"):
    times = np.array([0.0, 0.5, 1.0])
    q = Rotation.from_euler(axis, np.array([0.0, deg, 0.0])[:, None], degrees=True).as_quat()
    return AnimationSpec("probe", [Track(joint, "rotation", times, q)])


def _onehot(names, which):
    W = np.zeros((len(which), len(names)))
    for i, w in enumerate(which):
        W[i, names.index(w)] = 1.0
    return W


def test_g1_worn_is_zero_when_the_pins_move_with_the_torso_and_not_when_the_wing_carries_them():
    pins = np.array([[0.0, 0.9, 0.3], [0.1, 0.9, 0.3]])
    body_V = np.array([[0.0, 0.9, 0.28], [0.1, 0.9, 0.28], [-0.3, 0.9, 0.0]])
    body_W = _onehot(NAMES, ["chest", "chest", "wing_left"])
    torso = np.array([True, True, False])
    ok = G.gate_worn(pins, _onehot(NAMES, ["chest", "chest"]), body_V, body_W, torso, JOINTS, NAMES, [_clip("chest", 30)], H=2.0)
    assert ok.id == "G1" and ok.passed and ok.value < 1e-9
    bad = G.gate_worn(pins, _onehot(NAMES, ["wing_left", "wing_left"]), body_V, body_W, torso, JOINTS, NAMES, [_clip("wing_left", 30)], H=2.0)
    assert not bad.passed and bad.value > 0.01


def test_g2_support_only():
    assert G.gate_support_only(np.array([[0.5, 0.5, 0.0]]), NAMES).passed
    g = G.gate_support_only(np.array([[0.5, 0.4, 0.1]]), NAMES)
    assert not g.passed and g.value == 0.1


def test_g3_penetration_reads_the_worst_clip():
    posed = {"per_clip": {"idle": {"inside_count": 3, "min": 0.01}, "hop": {"inside_count": 25, "min": -0.02}}, "worst": {}}
    g = G.gate_penetration(posed)
    assert not g.passed and g.value == 25 and g.detail["clip"] == "hop"
    assert G.gate_penetration({"per_clip": {"idle": {"inside_count": 3, "min": 0.01}}, "worst": {}}).passed


def test_g4_hem_is_above_the_beads_and_level():
    hem = np.stack([np.zeros(10), np.full(10, 0.13) + 0.004 * np.sin(np.arange(10)), np.zeros(10)], axis=1)
    assert G.gate_hem(hem, y0=0.0, H=1.0, bead_top_y=0.10).passed
    assert not G.gate_hem(hem - [0, 0.02, 0], y0=0.0, H=1.0, bead_top_y=0.10).passed
    wavy = hem.copy()
    wavy[:, 1] += 0.05 * (np.arange(10) % 2)
    assert not G.gate_hem(wavy, y0=0.0, H=1.0, bead_top_y=0.10).passed


def test_g5_root_covered_by_a_cylinder_around_it():
    ph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([0.2 * np.cos(ph), np.zeros(24), 0.2 * np.sin(ph)], axis=1)
    cyl = trimesh.creation.cylinder(radius=0.3, height=1.0, sections=32)     # axis along z by default
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))   # now along y
    covered = G.gate_root_covered(ring, np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]), [(cyl.vertices, cyl.faces)])
    assert covered.passed and covered.value == 1.0
    bare = G.gate_root_covered(ring, np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]), [])
    assert not bare.passed and bare.value == 0.0


def test_g7_stretch():
    P = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    sets = [_Set(np.array([0, 1]), np.array([1, 2]), np.array([1.0, 1.0]), 0.92)]
    assert G.gate_stretch(P, sets, 0.92).passed
    assert not G.gate_stretch(P * 1.2, sets, 0.92).passed


def test_g8_collar_visible():
    bins = np.linspace(-np.pi, np.pi, 9)
    collar = {"theta_bins": bins, "y_lo": np.full(8, 0.80), "y_hi": np.full(8, 0.90), "centre_xz": np.array([0.5, 0.5])}
    bmin, size = np.zeros(3), np.ones(3)
    assert G.gate_collar_visible(np.array([[0.5, 0.5, 0.9]]), collar, bmin, size).passed
    g = G.gate_collar_visible(np.array([[0.5, 0.85, 0.9]]), collar, bmin, size)
    assert not g.passed and g.value == 1


def test_table_lists_every_gate_with_its_verdict():
    gates = [G.Gate("G2", "support only", 0.0, "= 0", True, {}), G.Gate("G4", "hem", 0.11, ">= 0.122", False, {"min": 0.11})]
    text = G.format_table(gates)
    assert "G2" in text and "PASS" in text and "G4" in text and "FAIL" in text
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/meshforge/test_gates.py -q 2>&1 | tail -3`
Expected: FAIL — `ModuleNotFoundError: No module named 'meshforge.gates'`.

- [ ] **Step 3: Implement `meshforge/gates.py`**

```python
"""Acceptance gates for the wrap (spec §6): numbers with thresholds, computed
by the build and printed as a table. A failing gate does not stop the build;
it blocks promotion. G9 (the gray sheet) and G10 (guide size) are not here —
one is the user's eye, the other is `ls -l`."""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from meshforge import fk
from meshforge import regions as regionsmod
from meshforge import robe as robemod

FORBIDDEN: tuple[str, ...] = ("neck", "head", "cap", "tassel", "eye_left", "eye_right", "lid_left", "lid_right",
                              "wing_left", "wing_left_tip", "wing_right", "leg_left", "leg_right",
                              "foot_left", "foot_right", "tail")


class Gate(NamedTuple):
    id: str
    name: str
    value: float
    threshold: str
    passed: bool
    detail: dict


def clip_skins(joints, joint_names, clips, samples: int = 5) -> list[tuple[str, float, dict]]:
    out = []
    for anim in clips:
        dur = max(float(np.max(t.times)) for t in anim.tracks) if anim.tracks else 0.0
        for tm in np.linspace(0.0, dur, samples):
            rot, tr = robemod.clip_pose_at(anim, tm)
            skin = fk.skin_matrices(joints, fk.world_matrices(joints, rotations=rot, translations=tr))
            out.append((anim.name, float(tm), skin))
    return out


def gate_worn(pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H) -> Gate:
    """G1: the pinned row moves with the torso skin it is pinned to. Anchors
    are the nearest *torso* vertices (the loop passes under the fused wing;
    the wing lifting off it is not the cloth coming off the owl)."""
    pins = np.asarray(pins, dtype=np.float64)
    body_V = np.asarray(body_V, dtype=np.float64)
    torso_idx = np.flatnonzero(np.asarray(torso_mask, dtype=bool))
    near = torso_idx[cKDTree(body_V[torso_idx]).query(pins, workers=-1)[1]]
    anchors, aW = body_V[near], np.asarray(body_W, dtype=np.float64)[near]
    rest = pins - anchors
    worst, where = 0.0, {}
    for name, tm, skin in clip_skins(joints, joint_names, clips):
        p = fk.pose_vertices(pins, pin_W, joint_names, skin)
        a = fk.pose_vertices(anchors, aW, joint_names, skin)
        drift = np.linalg.norm((p - a) - rest, axis=1) / H
        i = int(np.argmax(drift))
        if drift[i] > worst:
            worst, where = float(drift[i]), {"clip": name, "t": round(tm, 2), "column": i}
    return Gate("G1", "worn: pinned row drift from its torso support", round(worst, 4), "<= 0.010 H", worst <= 0.010, where)


def gate_support_only(W_all, joint_names) -> Gate:
    cols = [i for i, nm in enumerate(joint_names) if nm in FORBIDDEN]
    mass = float(np.asarray(W_all)[:, cols].sum()) if cols else 0.0
    by = {joint_names[i]: round(float(np.asarray(W_all)[:, i].sum()), 4) for i in cols if np.asarray(W_all)[:, i].sum() > 0}
    return Gate("G2", "support only: weight on forbidden joints", round(mass, 4), "= 0", mass == 0.0, by)


def gate_penetration(posed: dict, limit: int = 20) -> Gate:
    worst, clip = 0, None
    for name, c in (posed.get("per_clip") or {}).items():
        if c and int(c.get("inside_count", 0)) >= worst:
            worst, clip = int(c.get("inside_count", 0)), name
    return Gate("G3", "no penetration: cloth vertices inside the body, worst clip", worst, f"<= {limit}", worst <= limit, {"clip": clip})


def gate_hem(hem_points, y0, H, bead_top_y) -> Gate:
    yf = (np.asarray(hem_points)[:, 1] - y0) / H
    floor = (bead_top_y - y0) / H + 0.02
    sigma = float(yf.std())
    ok = float(yf.min()) >= floor and sigma <= 0.015
    return Gate("G4", "hem: above the beads and level", round(float(yf.min()), 4), f">= {floor:.3f} H, sigma <= 0.015 H", ok,
                {"min": round(float(yf.min()), 4), "max": round(float(yf.max()), 4), "sigma": round(sigma, 4), "floor": round(floor, 4)})


def gate_root_covered(root_ring, wing_pivot, wing_axis, garments) -> Gate:
    """G5: rays from the wing axis out through each root-ring vertex; the
    ring is covered where a ray meets cloth beyond the vertex."""
    ring = np.asarray(root_ring, dtype=np.float64)
    axis = np.asarray(wing_axis, dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    rel = ring - np.asarray(wing_pivot, dtype=np.float64)
    foot = np.asarray(wing_pivot, dtype=np.float64) + (rel @ axis)[:, None] * axis
    dirs = ring - foot
    dist = np.linalg.norm(dirs, axis=1)
    good = dist > 1e-6
    dirs[good] /= dist[good][:, None]
    covered = np.zeros(len(ring), dtype=bool)
    if garments and good.any():
        mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=np.asarray(V), faces=np.asarray(F), process=False) for V, F in garments])
        loc, idx, _ = mesh.ray.intersects_location(ray_origins=foot[good], ray_directions=dirs[good], multiple_hits=True)
        if len(idx):
            t = np.einsum("ij,ij->i", loc - foot[good][idx], dirs[good][idx])
            beyond = t > dist[good][idx] + 0.002
            hit = np.zeros(int(good.sum()), dtype=bool)
            np.logical_or.at(hit, idx[beyond], True)
            covered[good] = hit
    frac = float(covered.mean()) if len(ring) else 0.0
    return Gate("G5", "root covered: ring rays meeting cloth", round(frac, 3), ">= 0.90", frac >= 0.90,
                {"covered": int(covered.sum()), "ring": int(len(ring))})


def gate_tail_clear(tail_V, tail_W, body_mesh, body_W, joints, joint_names, wave_clips) -> Gate:
    posed = robemod.posed_clearance(np.asarray(tail_V), np.asarray(tail_W), body_mesh, body_W, joints, joint_names, wave_clips)
    worst = max((int(c.get("inside_count", 0)) for c in posed["per_clip"].values() if c), default=0)
    mn = min((float(c.get("min", 0.0)) for c in posed["per_clip"].values() if c), default=0.0)
    return Gate("G6", "tail clear of the waving wing", worst, "= 0 inside", worst == 0, {"min": round(mn, 4)})


def gate_stretch(P, sets, stretch_k) -> Gate:
    P = np.asarray(P, dtype=np.float64)
    ratios = []
    for s in sets:
        if abs(float(s.k) - float(stretch_k)) < 1e-9:
            L = np.linalg.norm(P[s.j] - P[s.i], axis=1)
            ratios.append(L / np.maximum(s.rest, 1e-9))
    r = np.concatenate(ratios) if ratios else np.ones(1)
    med, p90 = float(np.median(r)), float(np.percentile(r, 90))
    ok = 0.97 <= med <= 1.05 and p90 <= 1.12
    return Gate("G7", "stretch: warp/weft l/l0", round(med, 3), "median in [0.97, 1.05], p90 <= 1.12", ok,
                {"median": round(med, 3), "p90": round(p90, 3), "p99": round(float(np.percentile(r, 99)), 3)})


def gate_collar_visible(cloth_V, collar: dict, bmin, size) -> Gate:
    f = (np.asarray(cloth_V, dtype=np.float64) - np.asarray(bmin)) / np.asarray(size)
    cx, cz = collar["centre_xz"]
    theta = np.arctan2(f[:, 0] - cx, f[:, 2] - cz)
    lo, hi = regionsmod.collar_bounds_at(collar, theta)
    inside = int(((f[:, 1] >= lo) & (f[:, 1] <= hi)).sum())
    return Gate("G8", "collar visible: cloth vertices in the band", inside, "= 0", inside == 0, {})


def run_gates(*, pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H, W_all, posed,
              hem_points, y0, bead_top_y, root_ring, wing_pivot, wing_axis, garments, tail_V, tail_W,
              body_mesh, wave_clips, sheet_P, sheet_sets, stretch_k, cloth_V, collar, bmin, size) -> list[Gate]:
    return [
        gate_worn(pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H),
        gate_support_only(W_all, joint_names),
        gate_penetration(posed),
        gate_hem(hem_points, y0, H, bead_top_y),
        gate_root_covered(root_ring, wing_pivot, wing_axis, garments),
        gate_tail_clear(tail_V, tail_W, body_mesh, body_W, joints, joint_names, wave_clips),
        gate_stretch(sheet_P, sheet_sets, stretch_k),
        gate_collar_visible(cloth_V, collar, bmin, size),
    ]


def format_table(gates: Sequence[Gate]) -> str:
    rows = [("gate", "measure", "value", "threshold", "verdict")]
    for g in gates:
        rows.append((g.id, g.name, f"{g.value:g}", g.threshold, "PASS" if g.passed else "FAIL"))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    lines = [" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    failed = [g.id for g in gates if not g.passed]
    lines.append(f"{len(gates) - len(failed)}/{len(gates)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/meshforge/test_gates.py -q 2>&1 | tail -3`
Expected: 8 passed. If G5's cylinder test finds 0 hits, check that `intersects_location` returned `index_ray` as the second element (trimesh ≥ 3: `(locations, index_ray, index_tri)`).

- [ ] **Step 5: Full suite, then stop (no commit unless asked)**

Run: `python3 -m pytest tests/meshforge -q 2>&1 | tail -2`

---

### Task 7: `--kente-style wrap` in the pipeline, gates in the log and report; master build

**Files:**
- Modify: `meshforge/owl_pipeline.py` (signature ~line 60–100; kente block ~257–400; clips block ~480–495; report ~538; CLI ~608–700)

**Interfaces:**
- Consumes: Task 4 `build_kente_wrap`, `KenteWrap`; Task 5 `support_weights`; Task 6 `run_gates`, `format_table`.
- Produces: `build_owl(..., wrap_params: "WrapParams | None" = None)`; CLI `--kente-style wrap` with `--wrap-hem --wrap-gather --wrap-knot-drop --wrap-knot-size --wrap-tail-width --wrap-tail-end --wrap-tail-offset`; `report["kente"] = wrap.info`, `report["gates"] = [ {id, name, value, threshold, passed, detail}, ... ]`, `report["kente_style"] == "wrap"`. No sleeve primitive in wrap builds.

- [ ] **Step 1: Edit `build_owl`**

(a) Signature: after `sleeve_explicit: "set[str] | None" = None,` add `wrap_params: "WrapParams | None" = None,`.

(b) Next to `robe_check = None` (top of the kente section) add `wrap_gate_args = None` and `report_gates = None`.

(c) `if kente_style in ("tunic", "robe"):` → `if kente_style in ("tunic", "robe", "wrap"):`. Inside it, `if kente_style == "tunic":` (the `ClothParams` scaling block) → `if kente_style in ("tunic", "wrap"):`.

(d) `if sleeve:` (the sleeve-first block) → `if sleeve and kente_style != "wrap":`.

(e) The sweeps block `if kente_style == "tunic":` → `if kente_style in ("tunic", "wrap"):`. Replace the single `robe = drapemod.build_kente_tunic(...)` call inside it with:

```python
                if kente_style == "wrap":
                    from meshforge import wrap as wrapmod

                    wp = wrap_params or wrapmod.WrapParams()
                    if target_faces < REF_FACES // 2:
                        # the guide build: a coarser tail and knot, JPEG cloth textures
                        wp = dataclasses.replace(wp, tail_n_u=max(12, int(round(30 * scale))),
                                                 tail_n_v=max(24, int(round(80 * scale))),
                                                 tail_texture_size=(512, 256), knot_strands=5, image_format="JPEG")
                    wrap = wrapmod.build_kente_wrap(flat, neck_map, weave_params, cloth_params, wp,
                                                    band_frame=neck_frame, sweep_vertices=sweeps)
                    robe = None
                else:
                    robe = drapemod.build_kente_tunic(flat, neck_map, weave_params, cloth_params,
                                                      band_frame=neck_frame, obstacle_points=obstacles,
                                                      obstacle_mesh=sleeve_mesh, sweep_vertices=sweeps)
```

(f) Immediately before the comment `# weights: body + chest, plus each wing for the cloth that lies` insert the wrap branch, and indent the existing block — from that comment down to and including `robe_pose_args = (robe.primitive.vertices, exported_W)` — one level under `else:`:

```python
            if kente_style == "wrap":
                from meshforge import wrap as wrapmod

                # skinning from the support graph, not from the nearest skin
                y_hip = float(next(j.pivot[1] for j in joints if j.name == "body"))
                rows, cols = wrap.sheet_grid
                W_sheet = wrapmod.support_weights(wrap.sheet.vertices, np.arange(rows * cols), wrap.sheet_grid,
                                                  wrap.pins["sheet"], flat.vertices, full, joint_names, y_hip)
                rows_t, cols_t = wrap.tail_grid
                chest_one = np.zeros((cols_t, len(joint_names)))
                chest_one[:, joint_names.index("chest")] = 1.0
                W_tail = wrapmod.support_weights(wrap.tail.vertices, np.arange(rows_t * cols_t), wrap.tail_grid,
                                                 wrap.pins["tail"], flat.vertices, full, joint_names, y_hip,
                                                 pin_weights=chest_one, periodic=False)
                W_knot = np.zeros((len(wrap.knot.vertices), len(joint_names)))
                W_knot[:, joint_names.index("chest")] = 1.0
                dense = {}
                for prim, Wd in ((wrap.sheet, W_sheet), (wrap.tail, W_tail), (wrap.knot, W_knot)):
                    prim.joints, prim.weights = top4(Wd.astype(np.float32))
                    E = np.zeros_like(Wd)
                    np.put_along_axis(E, prim.joints.astype(np.int64), prim.weights, axis=1)
                    dense[prim.name] = E
                    primitives.append(prim)
                kente_info = wrap.info
                _log(f"kente wrap: support={wrap.info['support']} sheet={wrap.info['sheet']['grid']} "
                     f"tail={wrap.info['tail']['grid']} knot={wrap.info['knot']}", t0)
                cloth_V = np.vstack([wrap.sheet.vertices, wrap.tail.vertices])
                cloth_W = np.vstack([dense["kente_wrap"], dense["kente_wrap_tail"]])
                robe_check = {"rest": robemod.clearance(cloth_V, flat, surface_tree=sampler.tree)}
                _log(f"wrap clearance at rest: {robe_check['rest']}", t0)
                robe_pose_args = (cloth_V, cloth_W)
                wrap_gate_args = {"wrap": wrap, "dense": dense, "params": wp,
                                  "W_all": np.vstack([W_sheet, W_tail, W_knot])}
            else:
                # weights: body + chest, plus each wing ...   (existing block, indented)
```

(g) In step 8, after `_log(f"robe clearance through the clips: ...")` and before `if sleeve_check is not None:` add:

```python
        if wrap_gate_args is not None:
            from meshforge import gates as gatesmod

            wrap, dense, wp = wrap_gate_args["wrap"], wrap_gate_args["dense"], wrap_gate_args["params"]
            H = float(flat.bounds[1][1] - flat.bounds[0][1])
            y0 = float(flat.bounds[0][1])
            bead_top = max((r["world_y"] + r["bead_radius"] for r in (bead_info or [])), default=y0 + 0.10 * H)
            pivot = {j.name: np.asarray(j.pivot, dtype=np.float64) for j in joints}
            axis = (pivot["wing_left_tip"] - pivot[wp.raised_wing]) if "wing_left_tip" in pivot \
                else wrap.support.root_ring.mean(axis=0) - pivot[wp.raised_wing]
            gates = gatesmod.run_gates(
                pins=wrap.pins["sheet"], pin_W=dense["kente_wrap"][:wrap.sheet_grid[1]],
                body_V=flat.vertices, body_W=full, torso_mask=np.isin(region_map.labels, ["chest", "body"]),
                joints=joints, joint_names=joint_names, clips=check_clips, H=H,
                W_all=wrap_gate_args["W_all"], posed=robe_check["posed"],
                hem_points=wrap.sheet.vertices[-wrap.sheet_grid[1]:], y0=y0, bead_top_y=bead_top,
                root_ring=wrap.support.root_ring, wing_pivot=pivot[wp.raised_wing], wing_axis=axis,
                garments=[(p.vertices, p.faces) for p in (wrap.sheet, wrap.tail, wrap.knot)],
                tail_V=wrap.tail.vertices, tail_W=dense["kente_wrap_tail"], body_mesh=flat,
                wave_clips=[a for a in anims if a.name == "wave"],
                sheet_P=wrap.sheet.vertices, sheet_sets=wrap.sheet_sets, stretch_k=cloth_params.stretch,
                cloth_V=np.vstack([wrap.sheet.vertices, wrap.tail.vertices]), collar=region_map.collar,
                bmin=flat.bounds[0], size=flat.bounds[1] - flat.bounds[0])
            _log("gates:\n" + gatesmod.format_table(gates), t0)
            report_gates = [dict(g._asdict()) for g in gates]
```

(h) In `report = {...}` add `"gates": report_gates,`.

- [ ] **Step 2: Edit the CLI (`main`)**

`choices=("tunic", "robe", "vest", "sash", "wrap")`; after the `--cloth-strips` flag add:

```python
    ap.add_argument("--wrap-hem", type=float, default=None, help="wrap: hem height, fraction of the height (default 0.125)")
    ap.add_argument("--wrap-gather", type=float, default=None, help="wrap: cloth per loop length inside the knot arc (default 1.3)")
    ap.add_argument("--wrap-knot-drop", type=float, default=None, help="wrap: knot top under the collar rim, fraction of the height (default 0.015)")
    ap.add_argument("--wrap-knot-size", type=float, default=None, help="wrap: knot bulge height, fraction of the height (default 0.07)")
    ap.add_argument("--wrap-tail-width", type=float, default=None, help="wrap: tail width, fraction of the height (default 0.22)")
    ap.add_argument("--wrap-tail-end", type=float, default=None, help="wrap: where the tail ends, fraction of the height (default 0.15)")
    ap.add_argument("--wrap-tail-offset", type=float, default=None, help="wrap: degrees behind the knot arc the tail is pinned (default 30)")
```

`if args.kente and args.kente_style == "tunic":` (the `c_over` block) → `in ("tunic", "wrap")`. After the sleeve params block add:

```python
    wrap_params = None
    if args.kente and args.kente_style == "wrap":
        from meshforge.wrap import WrapParams

        w_over = {k: v for k, v in (("hem_frac", args.wrap_hem), ("gather_ratio", args.wrap_gather),
                                    ("knot_drop", args.wrap_knot_drop), ("knot_size", args.wrap_knot_size),
                                    ("tail_width_frac", args.wrap_tail_width), ("tail_end_frac", args.wrap_tail_end),
                                    ("tail_offset_deg", args.wrap_tail_offset)) if v is not None}
        wrap_params = WrapParams(**w_over)
```

and pass `wrap_params=wrap_params` in the `build_owl(...)` call.

- [ ] **Step 3: Smoke-test the wiring on the guide size first** (fast, ~1 min)

```bash
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-guide-kente-wrap.glb --cache .owl_cache \
    --faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256 \
    --kente --kente-style wrap --beads 2>&1 | tee out/owl_kente/build-guide-wrap.log | tail -30
```
Expected: `kente wrap: support={...theta_k_deg ≈ -100, y_k_frac ≈ 0.43, theta_a_deg ≈ 38 ...}`, a `gates:` table with eight rows, `validation passed`. A traceback here is a wiring bug — fix it before the master build.

- [ ] **Step 4: Master build** (~5 min)

```bash
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-kente-wrap.glb --cache .owl_cache --kente --kente-style wrap --beads \
    2>&1 | tee out/owl_kente/build-wrap.log | tail -40
python3 -c "import json; r=json.load(open('out/owl_kente/owl-kente-wrap.report.json')); [print(g['id'], 'PASS' if g['passed'] else 'FAIL', g['value'], g['threshold'], g['detail']) for g in r['gates']]"
```
Expected: a GLB, a report with `kente_style == "wrap"` and eight gates. Record the table in the handoff (Task 10). **Do not tune parameters to make a gate pass before the gray sheet has been looked at** (Task 8): a gate failing is a finding, and the user judges the sheet first.

- [ ] **Step 5: Existing tests still pass; then stop (no commit unless asked)**

Run: `python3 -m pytest tests/meshforge -q 2>&1 | tail -2`

---

### Task 8: G9 — the gray sim-off sheet, and the textured viewer in Safari

**Files:**
- Modify: `tools/garment_diag.py` (`make_variants` ~line 672–720)

**Interfaces:**
- Produces: gray variants and render sheets for any GLB whose garment materials start with `owl_kente`; `out/owl_kente/diag_wrap/renders/sheet_garment_gray_posed.png`, `sheet_garment_gray_turntable.png`, `sheet_garment_gray_high_low.png`, `sheet_garment_gray_zooms.png`, `sheet_all_gray_*`, `sheet_naked_gray_*`, `sheet_shipped_turntable.png`; `out/owl_kente/owl-kente-wrap.html`.

- [ ] **Step 1: Make the variants material-prefix based** — in `make_variants`, replace the three tests `m.name in ("owl_kente_tunic", "owl_kente_sleeve")` with `m.name.startswith("owl_kente")` (two in the gray loops, one in the `drop` set). Also change the posed-sheet `label` in `stage_renders` to end with `; front right back left` only (drop nothing else). `stage_metrics`/`stage_maps` stay tunic-only; `--stage renders` never calls them.

- [ ] **Step 2: Render** (~4 min; playwright)

```bash
python3 tools/garment_diag.py out/owl_kente/owl-kente-wrap.glb --out out/owl_kente/diag_wrap \
    --report out/owl_kente/owl-kente-wrap.report.json --stage renders 2>&1 | tail -12
```
Expected: nine `sheet_*.png` paths printed.

- [ ] **Step 3: Look before the user does** — Read `out/owl_kente/diag_wrap/renders/sheet_garment_gray_posed.png` and `sheet_garment_gray_turntable.png`. Check against the spec's picture: knot under the collar on the −X shoulder, diagonal edge across the chest, the tablet wing over the cloth's edge, tail behind the raised wing, hem above the beads, nothing through the collar, legs inside the skirt on the hop. Write what you see (three sentences, defects first) into `out/owl_kente/diag_wrap/LOOK.md` together with the gate table.

- [ ] **Step 4: The textured viewer, in Safari**

```bash
python3 artifact/build.py --glb out/owl_kente/owl-kente-wrap.glb --out out/owl_kente/owl-kente-wrap.html
open -a Safari out/owl_kente/diag_wrap/renders/sheet_garment_gray_posed.png \
    out/owl_kente/diag_wrap/renders/sheet_garment_gray_turntable.png out/owl_kente/owl-kente-wrap.html
```

- [ ] **Step 5: Stop here for the user's judgement (G9 is theirs).** Report the gate table and `LOOK.md` verbatim. Parameter tuning (`--wrap-*`, `--cloth-*`) happens only after they have looked, and each re-build re-runs Steps 2–4.

---

### Task 9: Guide build, G10, and the kiosk flag

**Files:**
- Create: `/Users/AI-CCORE/altageris/AICCORE/Afromaha-Visit2D/src/renderer/public/models/guide/owl-guide-kente-wrap.glb` (copied build output)
- No kiosk code changes: `guideModel.ts` already resolves `?guideModel=<stem>` / `localStorage['boma.guideModel']` / `VITE_GUIDE_MODEL`.

- [ ] **Step 1: Guide build** (the Task 7 Step 3 command, re-run after any tuning)

```bash
python3 -m meshforge.owl_pipeline --hires "$HOME/Downloads/AI-CCORE Owl.glb" \
    --out out/owl_kente/owl-guide-kente-wrap.glb --cache .owl_cache \
    --faces 14000 --tex 512 --jpeg-quality 82 --eye-subdiv 2 --decal-height 256 \
    --kente --kente-style wrap --beads 2>&1 | tee out/owl_kente/build-guide-wrap.log | tail -20
ls -l out/owl_kente/owl-guide-kente-wrap.glb
python3 -c "import json; r=json.load(open('out/owl_kente/owl-guide-kente-wrap.report.json')); print('garment tris', r['kente']['sheet']['faces']+r['kente']['tail']['faces']+r['kente']['knot']['faces']); [print(g['id'], g['passed'], g['value']) for g in r['gates']]"
```
Expected (G10): size ≤ 1,600,000 bytes; garment triangles ≤ 8,000. If over: `--cloth-res 80,44 --cloth-tex 1024x512` first, then `--wrap-tail-width` no — the tail texture is already 512×256 in guide builds; if still over, lower `--tex` to 448 before touching geometry.

- [ ] **Step 2: Copy into the kiosk behind the flag**

```bash
cp out/owl_kente/owl-guide-kente-wrap.glb /Users/AI-CCORE/altageris/AICCORE/Afromaha-Visit2D/src/renderer/public/models/guide/owl-guide-kente-wrap.glb
cd /Users/AI-CCORE/altageris/AICCORE/Afromaha-Visit2D && npm test 2>&1 | tail -5
```
Expected: the guide tests (`guideModel.test.ts` among them) pass; `owl-guide.glb` untouched (`git status` shows only the new file under `public/models/guide/`).

- [ ] **Step 3: Open it on the map** — stop the earlier dev app (it was started with `VITE_GUIDE_MODEL=owl-guide-kente-v9`): `pkill -f "electron-vite dev" || true`. Then:

```bash
cd /Users/AI-CCORE/altageris/AICCORE/Afromaha-Visit2D && VITE_GUIDE_MODEL=owl-guide-kente-wrap npm run dev
```
(run in the background; the console must print `[guide] model models/guide/owl-guide-kente-wrap.glb`). The user judges it at distance. `localStorage.setItem('boma.guideModel','owl-guide-kente-wrap')` in the devtools console is the alternative flag if the env variable is inconvenient.

- [ ] **Step 4: Stop** — nothing promoted (W-PROMOTE unpicked).

---

### Task 10: Hand-off notes and the tickets log

**Files:**
- Modify: `docs/superpowers/handoff/2026-08-22-kente-robe-handoff.md` (append a section)
- Modify: `docs/superpowers/tickets/2026-08-23-owl-kente-wrap-tickets.md` (log lines)
- Modify: `docs/superpowers/specs/2026-08-23-owl-kente-wrap-design.md` (one amendment, below)

- [ ] **Step 1: Spec amendment** — in §2 the tail row says "pinned to the knot's back arc"; the build pins it `tail_offset_deg` (30°) *behind* the knot arc's back end, on the back, so it hangs clear of the raised wing's sweep and the knot's bulk hides the join. Replace that cell with: "pinned on the back, `tail_offset_deg` behind the knot arc, so it hangs behind the raised wing; falls free". Add the same sentence to §4 item 4.

- [ ] **Step 2: Handoff section** — append "## The kente wrap (2026-08-23)" with: the three parts; the support loop numbers from `report['kente']['support']`; the gate table verbatim; the three commands (master, guide, renders); what the user said after looking (quote); open items. Keep it under 80 lines.

- [ ] **Step 3: Tickets log** — append dated lines: builds produced, gates passed/failed, user verdict, what was re-tuned.

- [ ] **Step 4: Stop (no commit unless asked)**.

---

## Deviations recorded during execution (2026-08-23, overnight)

The code is the reference where it differs from the task text above:

- `column_angles(theta, gather, cols, seg=None)`: columns are spaced by the support ring's **3D length per cell** (`seg`, computed in `draft` from the ring polygon) times the gather — spacing by angle alone tore the top row where the loop dives under the tablet wing (ratio 3.9).
- `measure_support(mesh, region_map, body, params, sweep_vertices=None, held_out=None)`: the tablet wing's lower contour is the **2nd percentile over the bind and the posed bodies**; the loop radius is the **torso envelope** (not `wrap_r`) widened by a 3-cell maximum and by the posed radius of *fused* vertices near the loop height, the raised wing always excluded; the loop is clamped under `rim − knot_drop`. `Support.info["axis_xz"]` feeds `loop_at`.
- `WrapParams` defaults: `knot_drop` 0.03, `gather_half_deg` 25, `arm_margin` 0.025, `hem_frac` 0.135, `tail_span_deg` 35, `tail_offset_deg` 5 (now measured **behind the root ring's back edge**), plus `neck_bound` / `root_bound` switches.
- `build_kente_wrap`: pins are pushed out of the whole mesh before the drape; the pattern is drafted with the **hem swing bound only**; the solve also uses the neck bound and a root bound built from the raised wing's root ring over the poses (12 samples per clip in the pipeline for wrap builds); final push-outs run against the full mesh, 12 rounds; `pins["tail"]` is the post-push-out row.
- `gates`: `gate_worn` anchors use the torso's own chest/body blend (fallback: the pin's weights); `gate_root_covered(root_ring, axis_xz, garments)` rays from the **torso axis**, and the pipeline passes only ring vertices with `y ≤ y_k + ½ knot_size·H`; `gate_collar_visible(cloth_V, theta_grid, rim_y, axis_xz)` counts cloth above the measured rim; `gate_stretch(..., cols=)` reports worst rows/columns; `run_gates` keywords changed accordingly.
- `knot.build_knot`: proportions reworked after the first render (small core under thick strands, a tie band around the waist, flaring ribbon ends).
- Synthetic-body tests: the fin is a 6 cm box fused to the egg; its crease traps ≤ 6 sheet vertices that `Collider.escape` cannot free, and its thin edges move two pins by up to 0.03 H — both tolerated in `test_wrap.py` with comments; the owl's own build reports 0 inside at rest.

## Deviations, second night (2026-08-23 02:00–04:00)

- `measure_support` gained a **swept-root clearance**: outside the knot arc the loop is capped under the lower
  contour of the raised wing's *root band* taken over the bind pose and every clip sample, blended in over
  `sweep_blend_deg` and limited by `sweep_drop_max`. New `WrapParams`: `sweep_band` 0.12, `sweep_blend_deg` 15,
  `sweep_drop_max` 0.08. This is what took G3 from 58 to 6; it also costs G5 (0.53 → 0.42), which is the same
  lever pulled the other way and is now ticket **W-G5**.
- `root_mask(mesh, labels, wing, band)` added; `root_ring` uses it; `_lower_boundary` accepts a mask as well as
  a label.
- `drape.taut_fall` + `draft(..., taut=)` + `ClothParams.taut_cap` added and left **off**: cutting columns to the
  path they travel instead of their drop relieves stretch but the surplus lands under the raised wing.
- `build_kente_wrap` grew a local `hang(pattern, cloth)`; a cut-hang-measure-recut fit pass was written, measured
  (it does not converge on this body) and removed.
- `knot.build_knot` was rewritten three more times; the shipped shape is a **cinched bunch**: a squashed
  icosphere pinched at the waist with soft flutes, a flat band round the waist, two short curled ends, three
  folds tucked under. 1016 faces. `knot_size` 0.07 → **0.095** (0.07 was invisible at kiosk distance).
- The tail is tapered (`tail_taper` 0.62), cut on the diagonal (`tail_cut_frac` 0.18), twisted per column
  (`tail_curl_deg` 22) and narrowed (`tail_width_frac` 0.22 → **0.15**).
- `tools/garment_diag.py::_zoom_plan` aims the close-ups at the garment parts found in the build.
- Tests: +7 (`taut_fall` ×3, `root_mask`, swept clearance ×2, tail shape ×1). `tests/meshforge` 361 passed.
