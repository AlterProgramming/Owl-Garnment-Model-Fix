"""Tests for meshforge.wrap — the kente cloth tied over the raised wing.

Synthetic body from test_robe.make_body: egg torso, a bulge on +X
(`wing_right`, the tablet wing), a fin standing off -X (`wing_left`,
the raised wing), two legs. The loop must be measured off that body the
same way it is measured off the owl: nothing here is a stored coordinate.
"""
import dataclasses

import numpy as np
import pytest
from scipy.spatial import cKDTree

from meshforge import drape as D
from meshforge import robe as R
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


def test_root_ring_is_where_the_wing_meets_the_torso(body):
    mesh, rm = body
    ring = W.root_ring(mesh, np.asarray(rm.labels), "wing_left", 0.05)
    assert len(ring) >= 8
    assert ring[:, 0].max() < -0.3                      # on the -X side
    torso = mesh.vertices[np.isin(rm.labels, ["chest", "body", "neck"])]
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
    r_body = np.array([np.interp(y, measured.y, measured.envelope[:, i]) for i, y in enumerate(support.y)])
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


# --------------------------------------------------------------------------
# the garment
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dressed(body):
    mesh, rm = body
    return W.build_kente_wrap(mesh, rm, WEAVE, small_params(), FAST_WRAP, n_theta=48), mesh


def test_wrap_keeps_out_of_the_body(dressed):
    wrap, mesh = dressed
    # the synthetic fin is a 6 cm box fused to the egg: the crease where
    # they meet traps a handful of vertices that `escape` cannot free (both
    # faces' normals cancel). The owl has no such crease and its build
    # reports 0 inside at rest; the gates judge the owl
    for prim, limit in ((wrap.sheet, 6), (wrap.tail, 0)):
        assert R.clearance(prim.vertices, mesh)["inside_count"] <= limit, prim.name


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


def test_tail_is_tapered_and_cut(dressed):
    wrap, _ = dressed
    _, cols = wrap.tail_grid
    top = wrap.tail.vertices[:cols]
    bottom = wrap.tail.vertices[-cols:]
    top_width = np.linalg.norm(top[0, [0, 2]] - top[-1, [0, 2]])
    bottom_width = np.linalg.norm(bottom[0, [0, 2]] - bottom[-1, [0, 2]])
    assert bottom_width <= 0.8 * top_width
    assert abs(bottom[0, 1] - bottom[-1, 1]) >= 0.05 * wrap.tail_pattern.length


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


# --------------------------------------------------------------------------
# skinning from the support graph
# --------------------------------------------------------------------------

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

def test_root_mask_is_the_ring_as_a_mask(body):
    mesh, rm = body
    m = W.root_mask(mesh, np.asarray(rm.labels), "wing_left", 0.05)
    ring = W.root_ring(mesh, np.asarray(rm.labels), "wing_left", 0.05)
    assert m.dtype == bool and m.sum() == len(ring)
    assert np.asarray(rm.labels)[m].tolist() == ["wing_left"] * int(m.sum())


def test_loop_clears_where_the_raised_wing_sweeps(body):
    """The wave turns the raised wing through its own root. A loop drawn
    under the wing at rest is inside it at the peak, and the cloth is
    skinned to the torso alone (the wings are forbidden supports), so
    outside the knot arc the loop has to sit under the *swept* root.

    The fixture's fin hangs below the loop already, where the clearance is
    saturated by `sweep_drop_max` in both poses, so the fin is lifted first:
    a wing whose root starts above the loop is the case that matters."""
    import trimesh

    mesh, rm = body
    labels = np.asarray(rm.labels)
    fin = labels == "wing_left"
    V = np.asarray(mesh.vertices, dtype=np.float64).copy()
    V[fin, 1] += 0.5                                          # a wing that rests above the loop
    lifted = trimesh.Trimesh(vertices=V, faces=np.asarray(mesh.faces), process=False)
    m2 = D.measure_body(lifted, rm, small_params(hem_frac=0.125), n_theta=48, n_rows=40)
    swung = V.copy()
    swung[fin, 1] -= 0.35                                     # ... and sweeps down through it
    rest = W.measure_support(lifted, rm, m2, FAST_WRAP, sweep_vertices=[])
    swept = W.measure_support(lifted, rm, m2, FAST_WRAP, sweep_vertices=[swung])
    assert (swept.y <= rest.y + 1e-9).all()                   # never higher anywhere
    assert (swept.y < rest.y - 1e-6).any()                    # ... and lower where the wing sweeps
    inside_arc = np.abs(W._wrap_angle(m2.theta - rest.theta_k)) <= np.radians(FAST_WRAP.gather_half_deg)
    assert swept.y_k == pytest.approx(rest.y_k)                    # the knot is tied on that root: it stays
    # inside the arc only the loop's own smoothing carries the drop across
    assert (rest.y[inside_arc] - swept.y[inside_arc]).max() < 0.02 * m2.size_y
    assert (rest.y - swept.y).max() <= FAST_WRAP.sweep_drop_max * m2.size_y + 1e-9


def test_sweep_clearance_can_be_turned_off(body):
    import trimesh

    mesh, rm = body
    labels = np.asarray(rm.labels)
    fin = labels == "wing_left"
    V = np.asarray(mesh.vertices, dtype=np.float64).copy()
    V[fin, 1] += 0.5
    lifted = trimesh.Trimesh(vertices=V, faces=np.asarray(mesh.faces), process=False)
    m2 = D.measure_body(lifted, rm, small_params(hem_frac=0.125), n_theta=48, n_rows=40)
    swung = V.copy()
    swung[fin, 1] -= 0.35
    off = dataclasses.replace(FAST_WRAP, sweep_band=0.0)
    a = W.measure_support(lifted, rm, m2, off, sweep_vertices=[swung])
    b = W.measure_support(lifted, rm, m2, off, sweep_vertices=[])
    assert np.allclose(a.y, b.y)
