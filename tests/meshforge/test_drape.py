"""Tests for meshforge.drape — the drafted, draped kente tunic.

Built on `test_robe.make_body`'s synthetic owl (egg torso, a folded-wing
bulge on +X, a fin standing off −X, short legs) so nothing here needs the
hi-res asset.

What is checked is what the module exists to fix. The old radial-hull
garment could not have a shoulder and could not come back in below the
widest thing it passed; these tests hold the new one to the opposite:
a pattern that opens and tapers, cloth that hangs from the neckline
without stretching, openings cut only around what is genuinely held out,
and — still, and above all — nothing inside the body.
"""
import dataclasses

import numpy as np
import pytest
import trimesh

from meshforge import drape as D
from meshforge import robe as R
from meshforge.textile import ASANTE_GOLD, WeaveParams
from tests.meshforge.test_robe import SMALL_WEAVE, make_body

FAST = dict(n_u=48, n_v=28, iterations=40, settle_iterations=10, texture_size=(256, 128),
            strips_around=6, edge_band=0.02)
WEAVE = dataclasses.replace(SMALL_WEAVE, strip_cycle=3) if hasattr(SMALL_WEAVE, "strip_cycle") else SMALL_WEAVE


def small_params(**kw):
    return D.ClothParams(**{**FAST, **kw})


@pytest.fixture(scope="module")
def body():
    mesh, region_map = make_body()
    return mesh, region_map


@pytest.fixture(scope="module")
def measured(body):
    mesh, region_map = body
    return D.measure_body(mesh, region_map, small_params(), n_theta=48, n_rows=40)


@pytest.fixture(scope="module")
def dressed(body):
    mesh, region_map = body
    weave = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=3, block_px=12, seed=0)
    return D.build_kente_tunic(mesh, region_map, weave, small_params(), n_theta=48), mesh


# --------------------------------------------------------------------------
# the draft
# --------------------------------------------------------------------------

def test_pattern_opens_from_the_neckline_and_comes_back_in(measured):
    pattern = D.draft(measured, small_params())
    circ = pattern.circumference
    assert circ[0] < circ.max(), "the piece must open below the neckline"
    assert circ[-1] < circ.max(), "and come back in below the chest — that taper is the side seam's"
    # the widest point is in the yoke, not at the hem: a piece whose widest
    # cut is its hem is a lampshade whatever the physics does with it
    assert np.argmax(circ) < len(circ) * 0.5


def test_every_column_is_cut_to_its_own_fall(measured):
    """The bottom row of the pattern is each column's own hem, so a neckline
    that dips at the front does not put the front of the garment on the floor."""
    pattern = D.draft(measured, small_params())
    assert np.allclose(pattern.v[-1], pattern.fall)
    assert np.allclose(pattern.v[0], 0.0)
    # a neckline that dips gives a pattern whose columns differ in length
    dipped = measured._replace(neck_y=measured.neck_y - 0.25 * (measured.theta > 0))
    shaped = D.draft(dipped, small_params())
    assert shaped.fall.max() - shaped.fall.min() == pytest.approx(0.25, abs=0.02)
    assert shaped.length == pytest.approx(shaped.fall.max())


def test_the_pattern_is_periodic_in_u_with_one_duplicated_seam_column(measured):
    pattern = D.draft(measured, small_params())
    rows, cols = pattern.grid_shape
    assert pattern.phi[0, 0] == 0.0 and pattern.phi[0, -1] == 1.0
    assert np.allclose(pattern.v[:, 0], pattern.v[:, -1], atol=1e-6)


def test_rest_lengths_follow_the_circumference_profile(measured):
    pattern = D.draft(measured, small_params())
    rows, cols = pattern.grid_shape
    keep = np.ones(rows * cols, dtype=bool)
    sets = D.constraint_sets(pattern.phi.ravel(), pattern.v.ravel(), pattern.circumference,
                             pattern.v_profile, keep, rows, cols, small_params())
    around = sets[0]                       # the (0, +1) set: one cell around the piece
    row_of = around.i // cols
    top = around.rest[row_of == 0].mean()
    mid = around.rest[row_of == rows // 2].mean()
    assert top < mid, "a cell of cloth at the neckline is narrower than one at the chest"


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------

def test_tethers_stop_a_pinned_sheet_falling_further_than_its_cloth():
    rows, cols = 6, 4
    phi = np.tile(np.arange(cols) / (cols - 1), (rows, 1)).ravel()
    v = np.tile(np.linspace(0.0, 1.0, rows)[:, None], (1, cols)).ravel()
    keep = np.ones(rows * cols, dtype=bool)
    pinned = np.zeros(rows * cols, dtype=bool)
    pinned[:cols] = True
    tet = D.build_tethers(phi, v, np.array([0.1, 0.1]), np.array([0.0, 1.0]), keep, pinned, rows, cols)
    P = np.zeros((rows * cols, 3))
    P[:, 1] = -10.0                        # everything dropped far below the pins
    P[:cols, 1] = 0.0
    D.apply_tethers(P, tet)
    fell = -P[cols:, 1]
    assert (fell <= v[cols:] + 1e-6).all(), "no vertex may hang further than the cloth between it and the pin"
    assert fell.max() > 0.5, "... and it may hang that far"


def test_a_projected_constraint_pulls_a_stretched_edge_back(measured):
    P = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    s = D._Set(np.array([0]), np.array([1]), np.array([1.0]), 1.0)
    w = np.ones(2)
    acc, cnt = np.zeros_like(P), np.zeros(2)
    D._project(P, s, w, acc, cnt)
    P += acc / np.maximum(cnt, 1)[:, None]
    assert np.linalg.norm(P[1] - P[0]) == pytest.approx(1.0, abs=1e-6)


def test_collider_pushes_a_point_inside_the_body_back_out(body):
    mesh, _ = body
    col = D.Collider(mesh, n=20_000)
    P = np.array([[0.0, 0.0, 0.0]])        # dead centre of the torso
    free = np.ones(1, dtype=bool)
    hits, _ = col.resolve(P, 0.02, free)
    assert hits == 1
    d, _ = col.tree.query(P)
    assert d[0] > 0.01


def test_push_out_leaves_nothing_inside():
    """Cloth the solve left a few millimetres inside the body — the case
    push_out is for. It is not a rescue for a garment built inside out."""
    mesh, _ = make_body(with_wings=False)      # one solid: no overlapping fin to be inside of
    col = D.Collider(mesh, n=40_000)
    rng = np.random.default_rng(0)
    surface, face = trimesh.sample.sample_surface(mesh, 80, seed=0)
    P = surface - mesh.face_normals[face] * rng.uniform(0.002, 0.015, (len(surface), 1))
    assert int(np.asarray(mesh.contains(P)).sum()) > 40, "the fixture must start inside"
    keep = np.ones(len(P), dtype=bool)
    D.push_out(P, keep, col, 0.02, rounds=12)
    assert int(np.asarray(mesh.contains(P)).sum()) == 0


# --------------------------------------------------------------------------
# what gets cut, and what does not
# --------------------------------------------------------------------------

def test_held_out_is_about_air_under_the_surface_not_distance_from_it(body):
    """The distinction the openings turn on. The folded wing's bulge is a
    tenth of the body away from the axis but there is no air under it, so
    cloth lies on it; a hand held out in front has air behind it, so cloth
    goes behind it and the pattern is cut. Distance from the torso cannot
    tell these apart — a ray can."""
    mesh, region_map = body
    slab = trimesh.creation.box(extents=[0.16, 0.22, 0.14])
    slab.apply_translation([0.0, -0.05, 0.62])          # torso reaches z = 0.36
    merged = trimesh.util.concatenate([mesh, slab])
    merged = trimesh.Trimesh(vertices=merged.vertices, faces=merged.faces, process=False)
    labels = np.concatenate([np.asarray(region_map.labels),
                             np.full(len(slab.vertices), "wing_right", dtype=object)])
    rm = region_map._replace(labels=labels)
    params = small_params()
    m = D.measure_body(merged, rm, params, n_theta=48, n_rows=40)
    held = D.held_out_vertices(merged, labels, m.scan, m.cells, m.axis_xz, params.hold_tolerance)
    is_slab = np.zeros(len(labels), dtype=bool)
    is_slab[len(mesh.vertices):] = True
    assert held[is_slab].mean() > 0.6, "a solid held out in front is cut around"
    bulge = (labels == "wing_right") & ~is_slab
    assert held[bulge].mean() < 0.15, "a wing fused to the flank is fallen over, not cut around"


def test_opening_field_is_negative_where_a_limb_came_through(measured):
    params = small_params()
    pattern = D.draft(measured, params)
    rows, cols = pattern.grid_shape
    P = D.initial_positions(pattern, measured)
    rows, cols = pattern.grid_shape
    r0, c0 = rows // 2, cols // 3
    hit = P[[(r0 + dr) * cols + c0 + dc for dr in range(-2, 3) for dc in range(-2, 3)]]
    f = D.signed_field(D.opening_mask(P, pattern, hit, params, 0.02, 0.01), 0.02, 0.01, params.open_smooth).ravel()
    assert f.min() < 0, "a point on the cloth opens it"
    assert f.max() > 0, "and only there"
    assert (f < 0).mean() < 0.2
    F = f.reshape(rows, cols)
    assert np.allclose(F[:, 0], F[:, -1]), "the seam column is the same cloth as the first"


def test_an_opening_is_cut_where_a_limb_crosses_the_cloth_and_not_in_front_of_it(measured):
    """An armhole belongs where the limb passes *through* the garment. An
    arm held across the chest is simply in front of it and wants cloth
    behind it, not a hole the size of the arm — cutting one leaves a
    lappet hanging off the forearm, which is what it did."""
    params = small_params()
    pattern = D.draft(measured, params)
    P = D.initial_positions(pattern, measured)
    rows, cols = pattern.grid_shape
    here = P[(rows // 2) * cols + cols // 3]
    out = np.array([here[0], 0.0, here[2]])
    out /= np.linalg.norm(out)
    assert D.opening_mask(P, pattern, (here + 0.02 * out)[None, :], params, 0.02, 0.01).any()
    assert not D.opening_mask(P, pattern, (here + 0.20 * out)[None, :], params, 0.02, 0.01).any()


def test_trim_slivers_removes_a_neck_of_cloth_between_two_openings():
    """A tongue thinner than twice the selvedge is gold on both faces and
    flips out under gravity — measured on the owl, beside the hand slit."""
    du = dv = 0.01
    mask = np.zeros((31, 31), dtype=bool)
    mask[:, :14] = True
    mask[:, 17:] = True                    # a 3-cell (0.03) neck of cloth
    assert D.trim_slivers(mask, 0.05, du, dv).all(), "a 30 mm neck does not survive a 50 mm minimum"
    wide = np.zeros((31, 31), dtype=bool)
    wide[:, :5] = True
    wide[:, 26:] = True                    # a 21-cell (0.21) panel
    out = D.trim_slivers(wide, 0.05, du, dv)
    assert not out.all(), "a wide panel survives"
    assert (out >= wide).all(), "trimming only ever widens an opening"


def test_offcuts_are_swallowed_by_the_openings():
    """A cut that isolates a patch leaves it held by nothing but its
    tethers; it swings free and, being inside the selvedge everywhere,
    renders as a gold sail. A garment is one piece."""
    mask = np.zeros((21, 31), dtype=bool)
    mask[5:15, :] = True                       # a full band of opening...
    mask[6:9, 12:14] = False                   # ... with a 6-cell island of cloth in it
    out = D.drop_offcuts(mask, 0.04)
    assert out[6:9, 12:14].all(), "the island is offcut and goes"
    assert not out[0].any() and not out[-1].any(), "the two real panels stay"
    # nothing to do when the cloth is already one piece
    one = np.zeros((21, 31), dtype=bool)
    one[5:8, 4:9] = True
    assert (D.drop_offcuts(one, 0.04) == one).all()


def test_no_opening_where_nothing_is_held_out(measured):
    params = small_params()
    pattern = D.draft(measured, params)
    P = D.initial_positions(pattern, measured)
    f = D.signed_field(D.opening_mask(P, pattern, np.zeros((0, 3)), params, 0.02, 0.01), 0.02, 0.01, params.open_smooth).ravel()
    assert (f > 0).all()


# --------------------------------------------------------------------------
# the swept bound
# --------------------------------------------------------------------------

def test_radial_bound_pushes_cloth_out_to_where_a_limb_swings():
    axis = np.array([0.0, 0.0])
    rest = np.array([[0.2, 0.0, 0.0]])
    swung = np.array([[0.5, 0.0, 0.0]])
    b = D.RadialBound([rest, swung], axis, y_lo=-0.2, y_hi=0.2, n_theta=16, n_rows=4, smooth=0.0)
    P = np.array([[0.3, 0.0, 0.0], [0.3, 1.0, 0.0]])     # one in the band, one above it
    moved = b.apply(P, 0.02, free=np.ones(2, dtype=bool))
    assert moved == 1
    assert np.hypot(P[0, 0], P[0, 2]) == pytest.approx(0.52, abs=1e-6)
    assert P[1, 0] == pytest.approx(0.3), "outside the band the bound says nothing"


# --------------------------------------------------------------------------
# the garment
# --------------------------------------------------------------------------

def test_tunic_keeps_out_of_the_body(dressed):
    tunic, mesh = dressed
    c = R.clearance(tunic.primitive.vertices, mesh)
    assert c["inside_count"] == 0
    assert c["min"] > 0.001


def test_tunic_is_not_a_lampshade(dressed):
    """The regression this module exists for: the radial hull carried the
    widest thing it passed down to the hem. A drafted piece must be
    narrower at the hem than at the shoulder."""
    tunic, mesh = dressed
    V = tunic.primitive.vertices
    y0, y1 = mesh.bounds[0][1], mesh.bounds[1][1]
    H = y1 - y0
    lo = V[V[:, 1] < V[:, 1].min() + 0.12 * H]
    hi = V[V[:, 1] > V[:, 1].max() - 0.20 * H]
    hem_w = lo[:, 0].max() - lo[:, 0].min()
    top_w = hi[:, 0].max() - hi[:, 0].min()
    assert hem_w < top_w, f"hem {hem_w:.3f} is not narrower than the shoulder {top_w:.3f}"


def test_tunic_hangs_from_the_neckline_and_reaches_the_hem(dressed):
    tunic, mesh = dressed
    V = tunic.primitive.vertices
    y0 = mesh.bounds[0][1]
    H = mesh.bounds[1][1] - y0
    hem = (V[:, 1].min() - y0) / H
    assert 0.05 < hem < 0.25, f"hem landed at {hem:.3f} of the height"
    top = (V[:, 1].max() - y0) / H
    assert top == pytest.approx(tunic.info["neckline_frac"][1], abs=0.03), "the top edge is the neckline"


def test_tunic_primitive_is_valid(dressed):
    tunic, _ = dressed
    prim = tunic.primitive
    n = len(prim.vertices)
    assert prim.faces.min() >= 0 and prim.faces.max() < n
    assert len(prim.uvs) == n and len(prim.normals) == n
    assert np.isfinite(prim.vertices).all()
    assert prim.uvs.min() >= -1e-6 and prim.uvs.max() <= 1 + 1e-6
    assert np.allclose(np.linalg.norm(prim.normals, axis=1), 1.0, atol=1e-5)
    assert prim.material.base_color_image is not None


def test_tunic_normals_face_out(dressed):
    tunic, _ = dressed
    prim = tunic.primitive
    tri = prim.vertices[prim.faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    c = tri.mean(axis=1)
    radial = c - np.array([tunic.info["axis_xz"][0] if "axis_xz" in tunic.info else 0.0, 0.0, 0.0])
    radial[:, 1] = 0.0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    assert float(np.mean(np.sum(fn * radial, axis=1))) > 0.3


def test_tunic_uv_chart_is_the_pattern(dressed):
    """The point of drafting: the chart *is* the cloth, so the weave runs
    along the grain and bends where the cloth bends."""
    tunic, _ = dressed
    rows, cols = tunic.grid_shape
    uv = tunic.primitive.uvs
    col = tunic.grid_index % cols
    plain = col < cols - 1
    assert np.corrcoef(uv[plain, 0], (tunic.grid_index[plain] % cols) / (cols - 1))[0, 1] > 0.99


def test_tunic_grid_index_matches_the_grid(dressed):
    tunic, _ = dressed
    rows, cols = tunic.grid_shape
    assert tunic.grid_index.max() < rows * cols
    assert len(np.unique(tunic.grid_index)) == len(tunic.grid_index)


def test_more_ease_means_more_cloth(body):
    mesh, region_map = body
    weave = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=3, block_px=12, seed=0)
    m = D.measure_body(mesh, region_map, small_params(), n_theta=48, n_rows=40)
    tight = D.draft(m, small_params(chest_ease=0.05))
    full = D.draft(m, small_params(chest_ease=0.40))
    assert full.circumference.max() > tight.circumference.max() * 1.2


def test_tunic_rejects_a_weave_that_cannot_meet_itself(body):
    mesh, region_map = body
    weave = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=3, block_px=12, seed=0)
    with pytest.raises(ValueError, match="strips_around"):
        D.build_kente_tunic(mesh, region_map, weave, small_params(strips_around=7), n_theta=48)


def test_measure_body_needs_a_torso():
    mesh, region_map = make_body()
    empty = region_map._replace(labels=np.full(len(mesh.vertices), "head", dtype=object))
    with pytest.raises(ValueError, match="no torso"):
        D.measure_body(mesh, empty, small_params(), n_theta=48, n_rows=40)


# --------------------------------------------------------------------------
# the seam, the size and the yoke (2026-08-22 diagnosis)
# --------------------------------------------------------------------------

def test_seam_columns_coincide_so_the_cloth_is_a_tube(dressed):
    """The chart's duplicated column is the same cloth as column 0: the
    solve keeps the two coincident. Measured on the first tunic, nothing
    joined them and the back gaped 0.15 at the hem."""
    tunic, _ = dressed
    rows, cols = tunic.grid_shape
    col = tunic.grid_index % cols
    row = tunic.grid_index // cols
    V = tunic.primitive.vertices
    first = {r: i for i, (r, c) in enumerate(zip(row, col)) if c == 0}
    last = {r: i for i, (r, c) in enumerate(zip(row, col)) if c == cols - 1}
    both = sorted(set(first) & set(last))
    assert len(both) > rows // 2
    gap = np.array([np.linalg.norm(V[first[r]] - V[last[r]]) for r in both])
    assert gap.max() < 1e-9, f"seam gap up to {gap.max():.4f}"


def test_constraints_wrap_across_the_seam(measured):
    pattern = D.draft(measured, small_params())
    rows, cols = pattern.grid_shape
    keep = np.ones(rows * cols, dtype=bool)
    a, b = D._pairs(rows, cols, keep, 0, 1)
    n = cols - 1
    assert np.any((a % cols == n - 1) & (b % cols == 0)), "the last real column must be joined to column 0"
    assert not np.any(a % cols == cols - 1), "the copy column carries no constraints of its own"


def test_pattern_is_cut_for_what_the_cloth_wraps_not_the_bare_torso(measured):
    """The synthetic body has a wing bulge fused on +X: the perimeter the
    cloth falls over is longer than the mirrored torso girth, and the
    pattern must be cut to it (the first tunic was 35-75 % short)."""
    pattern = D.draft(measured, small_params(chest_ease=0.0))
    band = (measured.y >= measured.y_hem) & (measured.y <= measured.neck_y.max())
    torso_girth = float(np.nanmax(2 * np.pi * np.nanmean(measured.envelope[band], axis=1)))
    assert pattern.circumference.max() > torso_girth * 1.02


def test_the_yoke_is_never_cut(dressed):
    tunic, _ = dressed
    rows, cols = tunic.grid_shape
    v = tunic.pattern.v.ravel()[tunic.grid_index]
    params = small_params()
    yoke_rows = np.nonzero(tunic.pattern.v[:, 0] < params.yoke_keep)[0]
    expected = len(yoke_rows) * cols
    got = int((v < params.yoke_keep).sum())
    assert got == expected, f"{expected - got} yoke vertices were cut away"


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

def test_taut_fall_is_longer_than_the_drop_over_a_bulge():
    """A column cut to the drop from its support is short wherever the body
    is wider below it: the cloth has to travel out and over first."""
    y = np.linspace(0.0, 1.0, 21)
    r = np.where((y > 0.3) & (y < 0.7), 0.5, 0.2)[:, None] * np.ones((1, 4))
    theta = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    th = np.array([0.0])
    fall, path_y = D.taut_fall(y, r, theta, th, np.array([0.9]), np.array([0.2]), 0.05)
    drop = 0.9 - 0.05
    assert fall[0] > drop * 1.15                      # out to the bulge and back down
    assert path_y.shape == (1, 64)
    assert path_y[0, 0] == pytest.approx(0.9) and path_y[0, -1] == pytest.approx(0.05, abs=1e-6)
    assert (np.diff(path_y[0]) <= 1e-9).all()         # it only ever falls


def test_taut_fall_matches_the_drop_on_a_straight_body():
    y = np.linspace(0.0, 1.0, 21)
    r = np.full((21, 4), 0.3)
    theta = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    fall, _ = D.taut_fall(y, r, theta, np.array([0.0]), np.array([0.9]), np.array([0.3]), 0.05)
    assert fall[0] == pytest.approx(0.85, abs=1e-6)


def test_taut_fall_never_follows_the_body_back_in():
    """Below the widest thing it has passed, hanging cloth falls straight —
    cut to the body's own inward curve the piece pools on the floor."""
    y = np.linspace(0.0, 1.0, 21)
    r = np.linspace(0.5, 0.05, 21)[:, None] * np.ones((1, 4))     # a cone, widest at the bottom... reversed below
    theta = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    fall, path = D.taut_fall(y, r[::-1], theta, np.array([0.0]), np.array([0.95]), np.array([0.5]), 0.05)
    assert fall[0] == pytest.approx(0.9, abs=0.02)                 # a straight fall, no detour inward
