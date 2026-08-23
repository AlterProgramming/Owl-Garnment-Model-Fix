"""Tests for meshforge.robe — the kente robe garment.

Built on a synthetic owl-ish body (an egg-shaped torso with a bump for a
folded wing on +X, a fin for a raised wing on −X, short legs below) with
labels laid out the way regions.classify would label it, so nothing here
needs the hi-res asset. The properties checked are the ones the garment
is *for*: it stands off the body everywhere (the user's standard — ample
cloth, air between subject and clothing), it never enters the body, it
ends under the wing that waves and falls over the wing that doesn't, and
it is a valid skinned, textured primitive.
"""
import dataclasses

import numpy as np
import pytest
import trimesh

from meshforge import robe as R
from meshforge.regions import RegionMap
from meshforge.rigexport import AnimationSpec, Track
from meshforge.textile import ASANTE_GOLD, WeaveParams

SMALL_WEAVE = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=2, block_px=12, seed=0)


def make_body(with_wings: bool = True):
    """Egg torso (radius 0.4, half-height 0.6) + two cylinder legs below.
    With wings: the +X flank *bulges* out to ~0.52 ("wing_right" — a wing
    folded against the body is one solid with it on the welded owl, so it
    is modelled as a deformation of the same surface, not a second shell
    the rays would cross twice), and a thin fin stands off the −X flank
    above the equator ("wing_left", the wing that waves). Labels by
    height/side like the real classifier's crop boxes."""
    torso = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    torso.apply_scale([0.40, 0.60, 0.36])
    if with_wings:
        V = torso.vertices.copy()
        side = np.clip((V[:, 0] - 0.15) / 0.25, 0.0, 1.0)
        band = np.clip(1.0 - np.abs(V[:, 1] + 0.05) / 0.35, 0.0, 1.0)
        bulge = 0.12 * (side * side * (3 - 2 * side)) * (band * band * (3 - 2 * band))
        V[:, 0] += bulge
        torso = trimesh.Trimesh(vertices=V, faces=torso.faces, process=False)
    parts = [torso]
    for x in (-0.12, 0.12):
        leg = trimesh.creation.cylinder(radius=0.07, height=0.5, sections=24)
        leg.apply_translation([x, -0.75, 0.0])
        parts.append(leg)
    if with_wings:
        fin = trimesh.creation.box(extents=[0.06, 0.5, 0.25])
        fin.apply_translation([-0.41, 0.27, 0.0])
        parts.append(fin)
    body = trimesh.util.concatenate(parts)
    body = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=True)
    V = body.vertices
    labels = np.full(len(V), "body", dtype=object)
    labels[V[:, 1] > -0.1] = "chest"
    labels[V[:, 1] > 0.45] = "neck"
    labels[V[:, 1] < -0.55] = "leg_left"
    labels[(V[:, 1] < -0.55) & (V[:, 0] > 0)] = "leg_right"
    if with_wings:
        labels[(V[:, 0] > 0.42) & (V[:, 1] > -0.40) & (V[:, 1] < 0.30)] = "wing_right"
        labels[(V[:, 0] < -0.37) & (V[:, 1] > 0.02)] = "wing_left"
    bmin, bmax = body.bounds
    f = (V - bmin) / (bmax - bmin)
    n_bins = 36
    collar = {
        "theta_bins": np.linspace(-np.pi, np.pi, n_bins + 1),
        "y_lo": np.full(n_bins, 0.80), "y_hi": np.full(n_bins, 0.86),
        "centre_xz": np.array([0.5, 0.5]), "theta": np.zeros(len(V)),
    }
    rm = RegionMap(labels=labels, joints=[], fractions=f, collar=collar, stats={})
    return body, rm


# the synthetic fin is cut an armhole (the original construction, still
# available through `stop_below`); the owl's raised wing is draped over at
# the root and wears a sleeve instead (test_sleeve.py)
ARMHOLE = dict(stop_below=("wing_left",), drape_over=("wing_right", "tail"))
FAST = dataclasses.replace(R.RobeParams(), n_theta=48, n_rows=20, texture_size=(256, 64), pleats=6, **ARMHOLE)


@pytest.fixture(scope="module")
def dressed():
    body, rm = make_body()
    robe = R.build_kente_robe(body, rm, SMALL_WEAVE, FAST)
    return body, rm, robe


# --------------------------------------------------------------------------
# the measurements
# --------------------------------------------------------------------------

def test_radial_scan_reports_crossings_nearest_first_with_labels():
    body, rm = make_body(with_wings=False)
    scan = R.radial_scan(body, rm.labels, np.array([0.0, 0.0]), np.array([-0.2, 0.0, 0.2]), 16)
    assert scan.theta.shape == (16,)
    for r in range(3):
        for c in range(16):
            rad = scan.radius[r][c]
            assert len(rad) >= 1
            assert np.all(np.diff(rad) >= 0)
            assert len(scan.label[r][c]) == len(rad)
    # an egg of radius 0.40 x 0.36: every first crossing near the equator lies in that range
    mid = np.concatenate([scan.radius[1][c][:1] for c in range(16)])
    assert mid.min() > 0.33 and mid.max() < 0.42


def test_radial_scan_counts_a_vertex_hit_once():
    # a cube: rays at the diagonal angles pass exactly through its corners,
    # where two slice segments meet — one crossing, not two
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    labels = np.full(len(cube.vertices), "chest", dtype=object)
    scan = R.radial_scan(cube, labels, np.array([0.0, 0.0]), np.array([0.0]), 4)   # cells at ±45°, ±135°
    for c in range(4):
        assert len(scan.radius[0][c]) == 1
        assert abs(scan.radius[0][c][0] - 0.5 * np.sqrt(2)) < 1e-6


def test_torso_envelope_is_the_body_and_mirrors_the_left_half():
    body, rm = make_body()
    scan = R.radial_scan(body, rm.labels, np.array([0.0, 0.0]), np.linspace(-0.4, 0.4, 9), 32)
    env = R.torso_envelope(scan, mirror=True)
    assert env.shape == (9, 32)
    assert not np.isnan(env).any()
    # the fused +X bump (radius up to 0.5) must not be in the envelope:
    # the mirrored −X flank is ~0.40
    assert env.max() < 0.45
    # exact mirror: column k and column n-1-k agree
    assert np.allclose(env, env[:, ::-1])


def test_classify_cells_sees_the_folded_wing_as_drape_and_the_raised_as_stop():
    body, rm = make_body()
    scan = R.radial_scan(body, rm.labels, np.array([0.0, 0.0]), np.linspace(-0.4, 0.4, 9), 32)
    env = R.torso_envelope(scan)
    cells = R.classify_cells(scan, env, dataclasses.replace(R.RobeParams(), **ARMHOLE))
    right = (scan.theta > np.radians(60)) & (scan.theta < np.radians(120))
    left = (scan.theta < -np.radians(60)) & (scan.theta > -np.radians(120))
    mid_rows = slice(3, 6)
    assert (cells.kind[mid_rows][:, right] == "drape").any()
    assert (cells.kind[mid_rows][:, left] == "stop").any()
    front = np.abs(scan.theta) < np.radians(20)
    assert (cells.kind[:, front] == "torso").all()
    # with the defaults both wings are draped over: nothing stops the cloth
    draped = R.classify_cells(scan, env, R.RobeParams())
    assert (draped.kind[mid_rows][:, left] == "drape").any()
    assert not (draped.kind == "stop").any()


def _held_body():
    """A torso block with a second block held beside it at a given gap."""
    def make(gap):
        torso = trimesh.creation.box(extents=[0.6, 1.0, 0.6])
        held = trimesh.creation.box(extents=[0.1, 0.3, 0.3])
        held.apply_translation([0.3 + gap + 0.05, 0.0, 0.0])
        body = trimesh.util.concatenate([torso, held])
        labels = np.where(body.vertices[:, 0] > 0.31, "wing_right", "chest")
        return body, labels
    return make


def test_classify_cells_merges_a_drape_solid_pressed_close_but_holds_one_held_out():
    make = _held_body()
    params = R.RobeParams()
    for gap, expect_drape in ((0.05, True), (0.30, False)):
        body, labels = make(gap)
        scan = R.radial_scan(body, labels, np.array([0.0, 0.0]), np.array([-0.05, 0.0, 0.05]), 32)
        env = R.torso_envelope(scan, mirror=False)
        cells = R.classify_cells(scan, env, params)
        col = int(np.argmin(np.abs(scan.theta - np.pi / 2)))      # +X
        if expect_drape:
            assert cells.kind[1, col] == "drape"
            assert cells.drape_r[1, col] > 0.3 + gap + 0.09        # the far face of the held block
            assert cells.held[1][col] == []
        else:
            assert cells.kind[1, col] == "torso"
            assert len(cells.held[1][col]) == 1
            r_in, r_out = cells.held[1][col][0]
            assert abs(r_in - (0.3 + gap)) < 0.02 and abs(r_out - (0.3 + gap + 0.1)) < 0.02


def test_classify_cells_joins_a_held_solid_with_the_same_label_behind_it():
    # a palm (thin block) with a tablet (thin block) 0.1 further out, both "wing_right"
    torso = trimesh.creation.box(extents=[0.6, 1.0, 0.6])
    palm = trimesh.creation.box(extents=[0.04, 0.3, 0.3]).apply_translation([0.3 + 0.2 + 0.02, 0.0, 0.0])
    tablet = trimesh.creation.box(extents=[0.02, 0.3, 0.3]).apply_translation([0.3 + 0.2 + 0.04 + 0.1 + 0.01, 0.0, 0.0])
    body = trimesh.util.concatenate([torso, palm, tablet])
    labels = np.where(body.vertices[:, 0] > 0.31, "wing_right", "chest")
    scan = R.radial_scan(body, labels, np.array([0.0, 0.0]), np.array([-0.05, 0.0, 0.05]), 32)
    env = R.torso_envelope(scan, mirror=False)
    cells = R.classify_cells(scan, env, R.RobeParams())
    col = int(np.argmin(np.abs(scan.theta - np.pi / 2)))
    assert cells.kind[1, col] == "torso" and len(cells.held[1][col]) == 1
    r_in, r_out = cells.held[1][col][0]
    assert abs(r_in - 0.5) < 0.02 and abs(r_out - 0.66) < 0.02          # palm's near face .. tablet's far face


def test_classify_cells_does_not_join_held_solids_across_an_always_solid_and_rejects_clashing_params():
    # palm, then the tail (drape_always), then more hand: two held intervals, the tail draped over
    torso = trimesh.creation.box(extents=[0.6, 1.0, 0.6])
    palm = trimesh.creation.box(extents=[0.04, 0.3, 0.3]).apply_translation([0.52, 0.0, 0.0])
    tail = trimesh.creation.box(extents=[0.04, 0.3, 0.3]).apply_translation([0.60, 0.0, 0.0])
    hand2 = trimesh.creation.box(extents=[0.04, 0.3, 0.3]).apply_translation([0.68, 0.0, 0.0])
    body = trimesh.util.concatenate([torso, palm, tail, hand2])
    x = body.vertices[:, 0]
    labels = np.where(x > 0.64, "wing_right", np.where(x > 0.56, "tail", np.where(x > 0.31, "wing_right", "chest")))
    scan = R.radial_scan(body, labels, np.array([0.0, 0.0]), np.array([-0.05, 0.0, 0.05]), 32)
    env = R.torso_envelope(scan, mirror=False)
    cells = R.classify_cells(scan, env, R.RobeParams())
    col = int(np.argmin(np.abs(scan.theta - np.pi / 2)))
    assert abs(cells.drape_r[1, col] - 0.62) < 0.02                      # the tail's far face
    assert len(cells.held[1][col]) == 2
    assert abs(cells.held[1][col][0][1] - 0.54) < 0.02 and abs(cells.held[1][col][1][0] - 0.66) < 0.02
    body0, rm0 = make_body(with_wings=False)
    with pytest.raises(ValueError):
        R.build_kente_robe(body0, rm0, SMALL_WEAVE, dataclasses.replace(FAST, stop_below=("tail",)))


def test_cut_grid_drops_outside_faces_and_snaps_edge_vertices_to_the_zero_level():
    rows, cols = 12, 16
    r, c = np.mgrid[0:rows, 0:cols]
    F = np.hypot(r - 5.5, c - 7.5) - 3.2            # a round hole in the middle
    faces, src_a, src_b, frac, n_snapped = R.cut_grid(F)
    Fv = F.ravel()
    assert (Fv[faces] >= 0).any(axis=1).all()
    assert n_snapped > 0
    used = np.unique(faces)
    moved = used[frac[used] > 0]
    assert set(moved) == set(used[Fv[used] < 0])     # every outside vertex still used was snapped
    assert (src_b[moved] == moved).all() and (Fv[src_a[moved]] >= 0).all()
    # linear interpolation of F along the snap segment lands on the edge
    f_snapped = Fv[src_a[moved]] + frac[moved] * (Fv[src_b[moved]] - Fv[src_a[moved]])
    assert np.allclose(f_snapped, 0.0, atol=1e-9)
    # unmoved vertices are their own source
    still = np.setdiff1d(np.arange(rows * cols), moved)
    assert (src_a[still] == still).all() and (frac[still] == 0).all()
    # no hole: every face kept, nothing snapped
    faces_all, _, _, frac_all, n0 = R.cut_grid(np.ones((rows, cols)))
    assert len(faces_all) == 2 * (rows - 1) * (cols - 1) and n0 == 0 and not frac_all.any()


def test_smooth_grid_weights_softens_a_step_and_keeps_rows_normalised():
    rows, cols = 6, 9                                  # cols includes the seam copy
    n = rows * cols
    W = np.zeros((n, 2))
    col = np.arange(n) % cols
    W[:, 0] = (col < 4).astype(float)                  # joint 0 on one side of the seam...
    W[:, 1] = 1.0 - W[:, 0]
    W[col == cols - 1] = W[col == 0]                   # ... the seam column copies column 0
    grid_index = np.arange(n)
    out = R.smooth_grid_weights(W, grid_index, (rows, cols), sigma=1.0)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[col == cols - 1], out[col == 0])        # periodic: the seam stays a copy
    # the step at column 3|4 is now a ramp
    row0 = out[:cols, 0]
    assert 0.05 < row0[3] < 0.95 and 0.05 < row0[4] < 0.95
    assert np.all(np.diff(row0[1:4]) <= 1e-9)
    # cut-away cells (missing from grid_index) do not pull their neighbours toward zero
    keep = np.ones(n, dtype=bool)
    keep[(col == 1) & (np.arange(n) // cols == 2)] = False
    out2 = R.smooth_grid_weights(W[keep], grid_index[keep], (rows, cols), sigma=1.0)
    assert np.allclose(out2.sum(axis=1), 1.0)


def test_hole_field_is_negative_inside_zero_on_the_boundary_and_periodic():
    theta = -np.pi + (np.arange(36) + 0.5) * (2 * np.pi / 36)
    y = np.linspace(0.0, 1.0, 21)
    mask = np.zeros((21, 36), dtype=bool)
    mask[8:13, 34:36] = True                      # a hole straddling the seam ...
    mask[8:13, 0:2] = True                        # ... (columns 34, 35, 0, 1)
    h = R._hole_field(mask, theta, y, s_scale=1.0, smooth_cells=0.0)
    assert h.signed_distance(theta[35], y[10]) < 0 and h.signed_distance(theta[0], y[10]) < 0
    assert h.signed_distance(theta[18], y[10]) > 0.5          # the far side of the cylinder
    assert h.signed_distance(theta[35], y[3]) > 0
    # the boundary sits between the last cut cell and the first kept one
    edge_y = 0.5 * (y[12] + y[13])
    assert abs(h.signed_distance(theta[35], edge_y)) < 0.02
    # the centre is on the seam, the extent spans it
    assert abs(abs(h.centre[0]) - np.pi) < 0.1 or abs(h.centre[0]) < 0.1
    assert h.extent[3] > h.extent[2]
    # smoothing never shrinks the hole
    hs = R._hole_field(mask, theta, y, s_scale=1.0, smooth_cells=1.0)
    rows, cols = np.nonzero(mask)
    assert (hs.signed_distance(theta[cols], y[rows]) <= 1e-9).all()
    # heights past the scanned rows read as the nearest row, never extrapolate
    top = np.zeros_like(mask)
    top[-3:, 10:14] = True                        # a hole on the top row
    ht = R._hole_field(top, theta, y, s_scale=1.0, smooth_cells=0.0)
    assert ht.signed_distance(theta[12], y[-1] + 5.0) < 0
    assert np.isclose(ht.signed_distance(theta[12], y[-1] + 5.0), ht.signed_distance(theta[12], y[-1]))
    assert h.signed_distance(theta[18], y[0] - 5.0) > 0
    with pytest.raises(ValueError):
        R._hole_field(mask[:1], theta, y[:1], s_scale=1.0)


def test_cloth_is_cut_around_obstacle_points():
    body, rm = make_body(with_wings=False)
    params = dataclasses.replace(FAST, clearance_top=0.02, clearance_hem=0.06, pleat_depth=0.02)
    plain = R.build_kente_robe(body, rm, SMALL_WEAVE, params)
    # a ring of points through the cloth on the +X side at mid-height
    r = float(np.mean(np.linalg.norm(plain.primitive.vertices[:, [0, 2]], axis=1)))
    ang = np.linspace(np.radians(80), np.radians(100), 12)
    ring = np.stack([r * np.sin(ang), np.full_like(ang, 0.05), r * np.cos(ang)], axis=1)
    cut = R.build_kente_robe(body, rm, SMALL_WEAVE, params, obstacle_points=[ring])
    assert plain.holes == [] and len(cut.holes) == 1
    assert cut.info["obstacle_points"] == 12 and cut.info["slit_cells"] > 0
    h = cut.holes[0]
    assert abs(np.degrees(h.centre[0]) - 90) < 10
    # no cloth vertex near the ring any more
    d = np.linalg.norm(cut.primitive.vertices[:, None, :] - ring[None, :, :], axis=2).min(axis=1)
    assert d.min() > 0.02
    assert R.clearance(cut.primitive.vertices, body)["inside_count"] == 0
    # a ring well inside the cloth (1.5 x obstacle_near under its local radius) is not cut;
    # one just under it (0.5 x) is
    P = plain.primitive.vertices
    th_p = np.arctan2(P[:, 0] - plain.axis_xz[0], P[:, 2] - plain.axis_xz[1])
    r_p = np.hypot(P[:, 0] - plain.axis_xz[0], P[:, 2] - plain.axis_xz[1])
    def local_radius(a):
        near = np.argmin(np.abs(np.angle(np.exp(1j * (th_p - a)))) * 10 + np.abs(P[:, 1] - 0.05))
        return r_p[near]
    r_loc = np.array([local_radius(a) for a in ang])
    deep = np.stack([(r_loc - 1.5 * params.obstacle_near) * np.sin(ang), np.full_like(ang, 0.05),
                     (r_loc - 1.5 * params.obstacle_near) * np.cos(ang)], axis=1)
    shallow = np.stack([(r_loc - 0.5 * params.obstacle_near) * np.sin(ang), np.full_like(ang, 0.05),
                        (r_loc - 0.5 * params.obstacle_near) * np.cos(ang)], axis=1)
    assert R.build_kente_robe(body, rm, SMALL_WEAVE, params, obstacle_points=[deep]).holes == []
    assert len(R.build_kente_robe(body, rm, SMALL_WEAVE, params, obstacle_points=[shallow]).holes) == 1


def test_garment_overlap_counts_vertices_poking_outside_the_outer_garment():
    outer = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    inner = trimesh.creation.icosphere(subdivisions=2, radius=0.9)
    rep = R.garment_overlap(inner.vertices, outer.vertices, outer.faces, n_samples=20_000)
    assert rep["outside_count"] == 0 and rep["count"] == len(inner.vertices)
    assert 0.05 < rep["min_depth"] < 0.15
    poking = trimesh.creation.icosphere(subdivisions=2, radius=1.1)
    rep2 = R.garment_overlap(poking.vertices, outer.vertices, outer.faces, n_samples=20_000)
    assert rep2["outside_count"] == len(poking.vertices) and rep2["max_outside"] > 0.05
    # nothing to measure (the guide build's tunic sits below the sleeve): zeros, not a crash
    assert R.garment_overlap(np.zeros((0, 3)), outer.vertices, outer.faces, n_samples=2_000)["count"] == 0


# --------------------------------------------------------------------------
# the garment
# --------------------------------------------------------------------------

def test_robe_stands_off_the_body_everywhere(dressed):
    body, rm, robe = dressed
    c = R.clearance(robe.primitive.vertices, body)
    assert c["inside_count"] == 0
    assert c["min"] >= FAST.clearance_top * 0.8


def _grid_rows_cols(robe):
    rows, cols = robe.grid_shape
    return robe.grid_index // cols, robe.grid_index % cols


def test_robe_is_ample_at_the_hem_not_skin_tight(dressed):
    body, rm, robe = dressed
    rows, cols = robe.grid_shape
    r_idx, _ = _grid_rows_cols(robe)
    hem = robe.primitive.vertices[r_idx == rows - 1]
    assert len(hem) == cols
    r_hem = np.hypot(hem[:, 0] - robe.axis_xz[0], hem[:, 2] - robe.axis_xz[1])
    # the torso's widest radius is 0.40 (x) / 0.36 (z); the hem must hang
    # well outside it — the cloth falls from the widest point, it does not
    # follow the body back in toward the legs
    assert r_hem.min() > 0.36 + FAST.clearance_hem * 0.5
    # ... and the legs (radius 0.07 around x=±0.12) are far inside the cloth
    assert r_hem.min() > 0.25


def test_robe_cuts_an_armhole_around_the_raised_wing_and_covers_the_folded_one(dressed):
    body, rm, robe = dressed
    theta = -np.pi + (np.arange(FAST.n_theta) + 0.5) * (2 * np.pi / FAST.n_theta)
    left = (theta < -np.radians(75)) & (theta > -np.radians(105))
    right = (theta > np.radians(70)) & (theta < np.radians(110))
    # the neckline is the collar cap everywhere (0.78 of the 1.6-tall body
    # above its base at -1.0 = 0.248, minus the gap) — it is not lowered
    # under the fin any more; the fin gets an armhole instead
    assert robe.y_top[left].min() > 0.2
    assert robe.y_top[right].min() > 0.2
    P = robe.primitive.vertices
    # no cloth inside the fin itself (x -0.44..-0.38, y 0.02..0.52, |z| < 0.125), and none inside the body
    in_fin = (P[:, 0] < -0.38 + 0.005) & (P[:, 0] > -0.44 - 0.005) & (P[:, 1] > 0.02 - 0.005) & (np.abs(P[:, 2]) < 0.125 + 0.005)
    assert not in_fin.any()
    assert R.clearance(P, body)["inside_count"] == 0
    # ... but cloth in front of and behind the fin at the same heights (the shoulder strips)
    th = np.degrees(np.arctan2(P[:, 0] - robe.axis_xz[0], P[:, 2] - robe.axis_xz[1]))
    front_strip = (th > -62) & (th < -50) & (P[:, 1] > 0.05) & (P[:, 1] < 0.2)
    back_strip = (th < -118) & (th > -130) & (P[:, 1] > 0.05) & (P[:, 1] < 0.2)
    assert front_strip.any() and back_strip.any()
    # and on the bulge side the cloth is outside the bulge (x up to 0.52)
    right_cols = (th > 75) & (th < 105)
    assert P[right_cols, 0].max() > 0.52 + FAST.clearance_top
    assert len(robe.holes) >= 1
    assert robe.info["stop_cells"] > 0


def test_robe_primitive_is_valid(dressed):
    body, rm, robe = dressed
    prim = robe.primitive
    n = len(prim.vertices)
    assert prim.faces.min() >= 0 and prim.faces.max() < n
    assert prim.uvs.shape == (n, 2)
    assert prim.uvs.min() >= 0.0 and prim.uvs.max() <= 1.0
    assert np.allclose(np.linalg.norm(prim.normals, axis=1), 1.0)
    # outward: normals agree with the radial direction on average
    radial = prim.vertices - np.array([robe.axis_xz[0], 0.0, robe.axis_xz[1]])
    radial[:, 1] = 0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    assert np.mean(np.sum(prim.normals * radial, axis=1)) > 0.5
    assert prim.material.double_sided
    assert prim.material.base_color_image.size == FAST.texture_size
    assert prim.material.normal_image is None
    assert robe.info["faces"] == len(prim.faces)


def test_robe_seam_is_closed_and_shades_seamlessly(dressed):
    body, rm, robe = dressed
    rows, cols = robe.grid_shape
    r_idx, c_idx = _grid_rows_cols(robe)
    first = {int(r): i for i, (r, c) in enumerate(zip(r_idx, c_idx)) if c == 0}
    last = {int(r): i for i, (r, c) in enumerate(zip(r_idx, c_idx)) if c == cols - 1}
    assert set(first) == set(last) and len(first) == rows
    for r in first:
        assert np.allclose(robe.primitive.vertices[first[r]], robe.primitive.vertices[last[r]])
        assert np.allclose(robe.primitive.normals[first[r]], robe.primitive.normals[last[r]])


def test_robe_texture_uses_only_colorway_colors(dressed):
    body, rm, robe = dressed
    img = np.asarray(robe.primitive.material.base_color_image).astype(np.float64) / 255.0
    declared = np.array([*SMALL_WEAVE.colorway.ground, SMALL_WEAVE.colorway.accent, SMALL_WEAVE.colorway.motif,
                         FAST.edge_band_color])
    covered = img[..., 3] > 0.5                  # the armhole / slit interiors are not cloth
    px = img[covered][:, :3]
    close = np.all(np.abs(px[:, None, :] - declared[None, :, :]) < 2.5 / 255, axis=-1)
    assert close.any(axis=-1).mean() > 0.98     # gutter dilation may blend a few border texels


def test_robe_normal_map_is_optional_and_valid():
    body, rm = make_body()
    robe = R.build_kente_robe(body, rm, SMALL_WEAVE, dataclasses.replace(FAST, bake_normal=True))
    img = robe.primitive.material.normal_image
    assert img is not None and img.size == FAST.texture_size
    n = np.asarray(img).astype(np.float64)[..., :3] / 255.0 * 2.0 - 1.0
    assert np.median(n[..., 2]) > 0.9


def test_robe_raises_without_a_torso():
    body, rm = make_body(with_wings=False)
    rm = RegionMap(labels=np.full(len(body.vertices), "head", dtype=object), joints=[], fractions=rm.fractions,
                   collar=rm.collar, stats={})
    with pytest.raises(ValueError):
        R.build_kente_robe(body, rm, SMALL_WEAVE, FAST)


def test_more_clearance_means_a_wider_robe():
    body, rm = make_body(with_wings=False)
    tight = R.build_kente_robe(body, rm, SMALL_WEAVE, dataclasses.replace(FAST, clearance_hem=0.04, pleats=0, pleat_depth=0.0))
    loose = R.build_kente_robe(body, rm, SMALL_WEAVE, dataclasses.replace(FAST, clearance_hem=0.16, pleats=0, pleat_depth=0.0))
    rows, cols = tight.grid_shape
    hem_t = tight.primitive.vertices[_grid_rows_cols(tight)[0] == rows - 1]
    hem_l = loose.primitive.vertices[_grid_rows_cols(loose)[0] == rows - 1]
    r_t = np.hypot(*(hem_t[:, [0, 2]] - tight.axis_xz).T)
    r_l = np.hypot(*(hem_l[:, [0, 2]] - loose.axis_xz).T)
    assert np.all(r_l > r_t)
    assert np.isclose(np.median(r_l - r_t), 0.12, atol=0.02)


# --------------------------------------------------------------------------
# skinning + measurement helpers
# --------------------------------------------------------------------------

def test_transfer_body_weights_blends_from_the_pool_and_respects_allowed_joints():
    body_V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]])
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]])
    pool = np.array([True, True, True, False])
    garment = np.array([[0.5, 0.0, 0.0], [5.0, 5.0, 5.0]])
    out, nearest = R.transfer_body_weights(garment, body_V, pool, W, k=2)
    assert out.shape == (2, 3)
    assert np.allclose(out.sum(axis=1), 1.0)
    # the midpoint between vertex 0 and 1 blends them equally; never the excluded vertex 3
    assert np.allclose(out[0], [0.5, 0.5, 0.0])
    assert nearest[1] != 3
    out2, _ = R.transfer_body_weights(garment, body_V, pool, W, k=2, allowed_joints=np.array([True, False, False]))
    assert np.allclose(out2[:, 1:], 0.0)
    assert np.allclose(out2.sum(axis=1), 1.0)


def test_transfer_body_weights_rejects_an_empty_pool():
    with pytest.raises(ValueError):
        R.transfer_body_weights(np.zeros((1, 3)), np.zeros((2, 3)), np.array([False, False]), np.eye(2))


def test_clearance_counts_vertices_inside_the_body():
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    outside = np.array([[1.2, 0.0, 0.0], [0.0, 1.5, 0.0]])
    inside = np.array([[0.5, 0.0, 0.0]])
    c_out = R.clearance(outside, body)
    assert c_out["inside_count"] == 0 and c_out["min"] > 0.15
    c_in = R.clearance(np.concatenate([outside, inside]), body)
    assert c_in["inside_count"] == 1
    assert np.isclose(c_in["inside_fraction"], 1 / 3)


def test_clip_pose_at_picks_the_nearest_keyframe():
    track = Track(joint="chest", path="rotation", times=np.array([0.0, 1.0, 2.0]),
                  values=np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float64))
    anim = AnimationSpec("t", [track])
    rot, tr = R.clip_pose_at(anim, 0.9)
    assert np.allclose(rot["chest"], [0, 0, 1, 0])
    assert tr == {}


def test_posed_clearance_moves_garment_with_its_joints():
    from meshforge.fk import Joint

    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    joints = [Joint("body", None, np.zeros(3)), Joint("chest", "body", np.array([0.0, 0.0, 0.0]))]
    names = ["body", "chest"]
    # the body rides `chest` (a child of `body`); the clip lifts `chest` only
    body_W = np.tile([0.0, 1.0], (len(body.vertices), 1))
    garment = np.array([[1.3, 0.0, 0.0], [0.0, 1.3, 0.0]])
    track = Track(joint="chest", path="translation", times=np.array([0.0, 1.0]),
                  values=np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]))
    clip = [AnimationSpec("lift", [track])]
    # garment on `chest` too: body and garment move together, clearance unchanged
    res = R.posed_clearance(garment, np.array([[0.0, 1.0], [0.0, 1.0]]), body, body_W, joints, names, clip,
                            samples_per_clip=3)
    assert res["worst"]["inside_count"] == 0
    assert np.isclose(res["worst"]["min"], 0.3, atol=0.02)
    # garment on `body` (which does not move): left behind, the lifted sphere swallows its top vertex
    res2 = R.posed_clearance(garment, np.array([[1.0, 0.0], [1.0, 0.0]]), body, body_W, joints, names, clip,
                             samples_per_clip=3)
    assert res2["worst"]["inside_count"] >= 1
    assert res2["worst"]["clip"] == "lift" and res2["worst"]["time"] > 0


# --------------------------------------------------------------------------
# review findings (2026-08-22 luna review) turned into tests
# --------------------------------------------------------------------------

def test_clearance_inside_test_is_geometric_not_vertex_normal_based():
    """A unit box has only corner vertices, whose averaged normals point
    diagonally: a point hovering just outside a face reads as 'inside'
    under a nearest-vertex-normal test (the first version of `clearance`
    did exactly that — 4 of 5 outside points flagged). The test must use
    the surface itself."""
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    outside = np.array([[0.6, 0.0, 0.0], [-0.6, 0.1, 0.1], [0.0, 0.55, 0.0], [0.2, 0.0, 0.7], [0.0, -0.8, 0.0]])
    inside = np.array([[0.0, 0.0, 0.0], [0.45, 0.0, 0.0]])
    c_out = R.clearance(outside, box)
    assert c_out["inside_count"] == 0
    assert np.isclose(c_out["min"], 0.05, atol=0.01)
    c_in = R.clearance(np.concatenate([outside, inside]), box)
    assert c_in["inside_count"] == 2


def test_surface_samples_follow_a_posed_body():
    body = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    samples = R.sample_surface(body, 2000, seed=1)
    p0, n0 = samples.evaluate(body.vertices, body.faces)
    p1, n1 = samples.evaluate(body.vertices + np.array([0.0, 2.0, 0.0]), body.faces)
    assert np.allclose(p1 - p0, [0.0, 2.0, 0.0])
    assert np.allclose(n1, n0)
    assert np.allclose(np.linalg.norm(n0, axis=1), 1.0)
    # outward: samples' normals point away from the centre
    assert np.all(np.sum(p0 * n0, axis=1) > 0)


def make_body_with_held_solid(gap: float):
    """Egg torso plus a detached block in front of it (a hand holding a
    tablet), `gap` away from the belly at the equator, labelled
    `wing_right` like the owl's — a solid the rays meet *after* the body."""
    torso = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    torso.apply_scale([0.40, 0.60, 0.36])
    block = trimesh.creation.box(extents=[0.30, 0.24, 0.08])
    block.apply_translation([0.0, 0.05, 0.36 + gap + 0.04])
    body = trimesh.util.concatenate([torso, block])
    body = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=True)
    V = body.vertices
    labels = np.full(len(V), "body", dtype=object)
    labels[V[:, 1] > -0.1] = "chest"
    labels[V[:, 2] > 0.36 + gap - 1e-6] = "wing_right"
    bmin, bmax = body.bounds
    f = (V - bmin) / (bmax - bmin)
    n_bins = 36
    collar = {"theta_bins": np.linspace(-np.pi, np.pi, n_bins + 1), "y_lo": np.full(n_bins, 0.90),
              "y_hi": np.full(n_bins, 0.95), "centre_xz": np.array([0.5, 0.5]), "theta": np.zeros(len(V))}
    return body, RegionMap(labels=labels, joints=[], fractions=f, collar=collar, stats={}), block


def test_cloth_is_cut_around_a_held_out_solid_it_would_cut_through():
    # drape_merge_gap 0: these test the slit itself; the owl's merge rule
    # (a solid pressed within 0.13 of the shoulder is draped over) would
    # otherwise swallow the 0.05 block
    params = dataclasses.replace(FAST, clearance_top=0.02, clearance_hem=0.06, pleat_depth=0.02, drape_merge_gap=0.0)
    body, rm, block = make_body_with_held_solid(gap=0.05)
    robe = R.build_kente_robe(body, rm, SMALL_WEAVE, params)
    theta = -np.pi + (np.arange(params.n_theta) + 0.5) * (2 * np.pi / params.n_theta)
    front = np.abs(theta) < np.radians(12)
    # the neckline is not lowered: the block gets a slit of its own
    assert robe.y_top[front].min() > 0.2
    assert len(robe.holes) >= 1 and robe.info["slit_cells"] > 0
    # no robe vertex ends up inside the block (with a small tolerance band)
    lo, hi = block.bounds
    P = robe.primitive.vertices
    in_block = np.all((P > lo - 0.01) & (P < hi + 0.01), axis=1)
    assert not in_block.any()
    assert R.clearance(P, body)["inside_count"] == 0
    # cloth survives beside the block at its height
    th = np.degrees(np.arctan2(P[:, 0] - robe.axis_xz[0], P[:, 2] - robe.axis_xz[1]))
    beside = (np.abs(np.abs(th) - 45) < 8) & (P[:, 1] > -0.05) & (P[:, 1] < 0.15)
    assert beside.any()


def test_cloth_passes_behind_a_solid_held_far_enough_out():
    # drape_merge_gap 0: these test the slit itself; the owl's merge rule
    # (a solid pressed within 0.13 of the shoulder is draped over) would
    # otherwise swallow the 0.05 block
    params = dataclasses.replace(FAST, clearance_top=0.02, clearance_hem=0.06, pleat_depth=0.02, drape_merge_gap=0.0)
    body, rm, block = make_body_with_held_solid(gap=0.30)
    robe = R.build_kente_robe(body, rm, SMALL_WEAVE, params)
    theta = -np.pi + (np.arange(params.n_theta) + 0.5) * (2 * np.pi / params.n_theta)
    front = np.abs(theta) < np.radians(12)
    # far enough away that the cloth hangs behind it: the front edge stays up
    assert robe.y_top[front].min() > 0.3
    assert R.clearance(robe.primitive.vertices, body)["inside_count"] == 0


def test_robe_reports_its_holes(dressed):
    body, rm, robe = dressed
    assert len(robe.info["holes"]) == len(robe.holes) >= 1
    h = robe.info["holes"][0]
    assert set(h) == {"centre_theta_deg", "centre_y_frac", "theta_deg_range", "y_frac_range"}
    assert h["theta_deg_range"][0] <= h["centre_theta_deg"] <= h["theta_deg_range"][1]
    assert h["y_frac_range"][0] <= h["centre_y_frac"] <= h["y_frac_range"][1]
    body2, rm2 = make_body(with_wings=False)
    plain = R.build_kente_robe(body2, rm2, SMALL_WEAVE, FAST)
    assert plain.holes == [] and plain.info["snapped_vertices"] == 0
    assert plain.info["vertices"] == FAST.n_rows * (FAST.n_theta + 1)


def test_robe_rejects_bad_parameters():
    body, rm = make_body(with_wings=False)
    with pytest.raises(ValueError):
        R.build_kente_robe(body, rm, SMALL_WEAVE, dataclasses.replace(FAST, n_theta=4))
    with pytest.raises(ValueError):
        R.build_kente_robe(body, rm, SMALL_WEAVE, dataclasses.replace(FAST, n_rows=1))
    with pytest.raises(ValueError):
        # SMALL_WEAVE has strip_cycle=2: 9 strips cannot meet themselves at the seam
        R.build_kente_robe(body, rm, SMALL_WEAVE, dataclasses.replace(FAST, strips_around=9))


def test_torso_precondition_honours_torso_labels():
    body, rm = make_body(with_wings=False)
    relabelled = RegionMap(labels=np.where(rm.labels == "body", "trunk", rm.labels).astype(object), joints=[],
                           fractions=rm.fractions, collar=rm.collar, stats={})
    # "trunk" is not a torso label by default ...
    with pytest.raises(ValueError):
        R.build_kente_robe(body, relabelled, SMALL_WEAVE, FAST, torso_labels=("neck",))
    # ... but dressing a body whose torso is called something else must work when told so
    robe = R.build_kente_robe(body, relabelled, SMALL_WEAVE, FAST, torso_labels=("trunk", "chest", "neck", "leg_left", "leg_right"))
    assert robe.info["faces"] > 0
