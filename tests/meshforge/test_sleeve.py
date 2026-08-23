"""Tests for meshforge.sleeve — the kente sleeve on the wing that waves.

Same synthetic body as test_robe.py: the −X fin (a 0.06 x 0.5 x 0.25 plate
standing off the egg above its equator, fused to it at the bottom) is the
wing. The properties checked are the ones the sleeve is for: it wraps the
wing with air all round, never enters the wing or the body, is wide past
the shoulder and snug at the cuff, stops short of the wing's end, and is a
valid skinned, textured primitive whose root hides under the tunic.
"""
import dataclasses

import numpy as np
import pytest
import trimesh

from meshforge import robe as R
from meshforge import sleeve as S
from tests.meshforge.test_robe import FAST, SMALL_WEAVE, make_body

PIVOT = np.array([-0.40, 0.05, 0.0])          # where the fin meets the egg
FIN = dict(x=(-0.44, -0.38), y=(0.02, 0.52), z=(-0.125, 0.125))
SMALL = dataclasses.replace(S.SleeveParams(), n_phi=24, station_spacing=0.02, texture_size=(256, 64))


def in_fin(P, pad=0.0):
    return ((P[:, 0] > FIN["x"][0] - pad) & (P[:, 0] < FIN["x"][1] + pad) & (P[:, 1] > FIN["y"][0] - pad)
            & (P[:, 1] < FIN["y"][1] + pad) & (np.abs(P[:, 2]) < FIN["z"][1] + pad))


@pytest.fixture(scope="module")
def sleeved():
    body, rm = make_body()
    slv = S.build_sleeve(body, rm.labels, PIVOT, SMALL_WEAVE, SMALL, wing_labels=("wing_left",))
    return body, rm, slv


def test_wing_frame_points_along_the_fin_away_from_the_pivot():
    body, rm = make_body()
    fr = S.wing_frame(S.label_samples(body, rm.labels, ("wing_left",)), PIVOT)
    assert fr.u[1] > 0.95                                  # the fin stands up
    assert abs(fr.n[0]) > 0.95                             # ... and is thin in x
    assert abs(fr.w[2]) > 0.95
    assert np.allclose([fr.u @ fr.w, fr.u @ fr.n, fr.w @ fr.n], 0.0, atol=1e-9)
    assert fr.s_lo < 0.05 and 0.4 < fr.s_hi < 0.5          # 0.02..0.52 above the pivot at 0.05


def test_separation_finds_where_the_fin_leaves_the_egg():
    body, rm = make_body()
    samples = S.label_samples(body, rm.labels, ("wing_left",))
    fr = S.wing_frame(samples, PIVOT)
    s_sep, stations, gaps = S.separation(samples, fr, 0.02, 0.02)
    # the egg's x-radius at height y is 0.40*sqrt(1 - (y/0.6)^2); it is
    # 0.02 inside the fin's face (x = -0.38) at y ~ 0.25, i.e. s ~ 0.2
    assert 0.1 < s_sep < 0.3
    assert np.nanmin(gaps[stations > s_sep + 0.02]) >= 0.02


def test_occupied_intervals_handle_inside_and_outside_starts_and_merge_fuzz():
    # ray starting inside a wing solid, then a body solid beyond it
    iv = S._occupied_intervals(np.array([0.1, 0.3, 0.5]), np.array([True, False, False]), 0.01)
    assert iv == [(0.0, 0.1, True), (0.3, 0.5, False)]
    # starting outside: enter/exit pairs; a re-entry within the fuzz is merged
    iv = S._occupied_intervals(np.array([0.1, 0.2, 0.205, 0.3]), np.array([True, True, True, True]), 0.01)
    assert iv == [(0.1, 0.3, True)]
    assert S._occupied_intervals(np.zeros(0), np.zeros(0, dtype=bool), 0.01) == []


def test_occupied_intervals_merge_is_conservative_about_the_label():
    # a body solid a hair behind the wing: the merged interval is NOT wing
    iv = S._occupied_intervals(np.array([0.2, 0.4, 0.41, 0.6]), np.array([False, False, True, True]), 0.02)
    assert iv == [(0.2, 0.6, False)]
    iv = S._occupied_intervals(np.array([0.4, 0.41, 0.6]), np.array([True, False, False]), 0.02)
    assert iv == [(0.0, 0.6, False)]


def test_dedupe_crossings_collapses_hits_through_a_shared_vertex():
    t, lab = S.dedupe_crossings(np.array([0.5, 0.1, 0.3, 0.3 + 1e-12]), np.array(["a", "b", "c", "d"]))
    assert np.allclose(t, [0.1, 0.3, 0.5]) and list(lab) == ["b", "c", "a"]
    t, lab = S.dedupe_crossings(np.zeros(0), np.zeros(0, dtype=object))
    assert len(t) == 0


def test_section_scan_parity_survives_rays_through_polyline_vertices():
    # a box wing sliced by a plane is a square; rays from its centre at the
    # diagonal angles pass exactly through its corners, where two slice
    # segments meet — one crossing each, not two
    box = trimesh.creation.box(extents=[0.2, 1.0, 0.2])
    labels = np.full(len(box.vertices), "wing_left", dtype=object)
    fr = S.WingFrame(pivot=np.array([0.0, -0.5, 0.0]), u=np.array([0.0, 1.0, 0.0]), w=np.array([0.0, 0.0, 1.0]),
                     n=np.array([1.0, 0.0, 0.0]), s_lo=0.0, s_hi=1.0)
    sc = S.section_scan(box, labels, fr, np.array([0.5]), 4, S.SleeveParams(), ("wing_left",))   # rays at ±45°, ±135°
    assert not sc.fused.any() and not sc.open.any()
    assert np.allclose(sc.wing_r, 0.1 * np.sqrt(2), atol=1e-6)
    assert np.isinf(sc.body_r).all()


def test_clearance_schedule_short_sleeve_still_ends_at_the_cuff_air():
    p = S.SleeveParams()
    s_sep, s_cuff = 0.1, 0.1 + p.wide_at * 0.5
    c = S.clearance_schedule(np.array([s_sep, 0.5 * (s_sep + s_cuff), s_cuff]), s_sep, s_cuff, p)
    assert np.isclose(c[0], p.clearance_root) and np.isclose(c[-1], p.clearance_cuff)
    assert min(p.clearance_root, p.clearance_cuff) <= c[1] <= max(p.clearance_root, p.clearance_cuff)


def test_clearance_schedule_is_wide_past_the_shoulder_and_snug_at_the_cuff():
    p = S.SleeveParams()
    s = np.array([0.0, 0.1, 0.1 + p.wide_at, 0.5])
    c = S.clearance_schedule(s, 0.1, 0.5, p)
    assert np.isclose(c[0], p.clearance_root) and np.isclose(c[1], p.clearance_root)
    assert np.isclose(c[2], p.clearance_wide) and np.isclose(c[3], p.clearance_cuff)
    assert c[2] > c[1] and c[2] > c[3]


def test_sleeve_wraps_the_fin_with_air_and_never_enters_fin_or_body(sleeved):
    body, rm, slv = sleeved
    P = slv.primitive.vertices
    assert not in_fin(P, pad=0.003).any()            # min_clearance is along the ray; oblique rays keep ~2/3 of it
    rep = R.clearance(P, body)
    assert rep["inside_count"] == 0
    assert rep["min"] >= 0.004          # body_margin is along the ray; the egg is oblique to it
    # cloth on both faces of the fin and beyond both its edges, at every station
    s = (P - PIVOT) @ slv.frame.u
    for st in np.linspace(slv.info["s_separation"] + 0.02, slv.info["s_cuff"] - 0.02, 4):
        ring = P[np.abs(s - st) < 0.015]
        assert (ring[:, 0] < FIN["x"][0]).any() and (ring[:, 0] > FIN["x"][1]).any()
        assert (ring[:, 2] < FIN["z"][0]).any() and (ring[:, 2] > FIN["z"][1]).any()


def test_sleeve_is_wide_at_the_shoulder_and_tapers_to_the_cuff(sleeved):
    body, rm, slv = sleeved
    r = slv.info["mean_radius"]
    assert r["cuff"] < r["separation"] + 0.01
    # the widest station sits just past the separation
    P = slv.primitive.vertices
    s = (P - PIVOT) @ slv.frame.u
    widths = []
    for st in np.linspace(slv.info["s_separation"], slv.info["s_cuff"], 6):
        ring = P[np.abs(s - st) < 0.015]
        widths.append(np.ptp(ring[:, 2]))
    # the widest ring is just past the separation, and the cuff is
    # narrower by most of the clearance difference (the station smoothing
    # rounds the schedule off a little)
    assert int(np.argmax(widths)) <= 2
    assert max(widths) - widths[-1] > 2 * (SMALL.clearance_wide - SMALL.clearance_cuff) * 0.6


def test_sleeve_stops_short_of_the_wing_end_and_reaches_back_under_the_root(sleeved):
    body, rm, slv = sleeved
    info = slv.info
    free = slv.frame.s_hi - info["s_separation"]
    assert np.isclose(info["s_cuff"], info["s_separation"] + SMALL.length_frac * free, atol=1e-3)   # info is rounded
    assert info["s_cuff"] < slv.frame.s_hi - 0.05
    assert info["s_root"] < info["s_separation"]
    # the side of the root ring against the egg is cut away: cells there are missing
    assert info["cut_cells"] > 0
    P = slv.primitive.vertices
    assert P[:, 1].max() < FIN["y"][1] - 0.05


def test_sleeve_primitive_is_valid_and_seam_closed(sleeved):
    body, rm, slv = sleeved
    p = slv.primitive
    assert p.faces.min() >= 0 and p.faces.max() < len(p.vertices)
    assert np.isfinite(p.vertices).all() and np.isfinite(p.normals).all()
    assert np.allclose(np.linalg.norm(p.normals, axis=1), 1.0, atol=1e-6)
    assert p.uvs.min() >= -1e-9 and p.uvs.max() <= 1 + 1e-9
    # normals point away from the wing: outward along the section radial
    radial = p.vertices - PIVOT
    radial -= (radial @ slv.frame.u)[:, None] * slv.frame.u
    assert np.mean(np.sum(p.normals * radial, axis=1) > 0) > 0.9
    mesh = trimesh.Trimesh(vertices=p.vertices, faces=p.faces, process=False)
    assert mesh.area > 0
    # the seam column duplicates column 0 in position: welding closes it
    welded = trimesh.Trimesh(vertices=p.vertices, faces=p.faces, process=True)
    assert len(welded.vertices) < len(p.vertices)
    rows, cols = slv.grid_shape
    assert cols == SMALL.n_phi + 1 and len(slv.grid_index) == len(p.vertices)


def test_sleeve_texture_uses_only_colorway_and_cuff_colors(sleeved):
    body, rm, slv = sleeved
    img = np.asarray(slv.primitive.material.base_color_image).astype(np.float64) / 255.0
    assert img.shape[:2] == SMALL.texture_size[::-1]
    covered = img[img[..., 3] > 0.5][:, :3]
    cw = SMALL_WEAVE.colorway
    palette = np.array([*cw.ground, cw.accent, cw.motif, SMALL.band_color])
    d = np.linalg.norm(covered[:, None, :] - palette[None, :, :], axis=2).min(axis=1)
    assert np.percentile(d, 99) < 0.06
    # the gold cuff is there: a band of texels at the cuff end of the chart
    gold = np.linalg.norm(covered - np.array(SMALL.band_color), axis=1) < 0.05
    assert gold.mean() > 0.02


def test_sleeve_rejects_bad_parameters():
    body, rm = make_body()
    with pytest.raises(ValueError):
        S.build_sleeve(body, rm.labels, PIVOT, SMALL_WEAVE, dataclasses.replace(SMALL, n_phi=4), wing_labels=("wing_left",))
    with pytest.raises(ValueError):
        S.build_sleeve(body, rm.labels, PIVOT, SMALL_WEAVE, dataclasses.replace(SMALL, length_frac=1.5), wing_labels=("wing_left",))
    with pytest.raises(ValueError):
        S.build_sleeve(body, rm.labels, PIVOT, SMALL_WEAVE, dataclasses.replace(SMALL, strips_around=5), wing_labels=("wing_left",))
    with pytest.raises(ValueError):
        S.build_sleeve(body, rm.labels, PIVOT, SMALL_WEAVE, SMALL, wing_labels=("no_such_label",))


def test_sleeve_root_hides_under_a_tunic_draped_over_the_root(sleeved):
    body, rm, slv = sleeved
    # the tunic with the fin draped over (the owl's construction), not cut an armhole
    draped = dataclasses.replace(FAST, stop_below=(), drape_over=("wing_right", "wing_left", "tail"))
    robe = R.build_kente_robe(body, rm, SMALL_WEAVE, draped)
    assert len(robe.holes) == 0
    P = slv.primitive.vertices
    s = (P - PIVOT) @ slv.frame.u
    # the part of the root ring that is below the tunic's neckline (the
    # synthetic collar cap sits low: the fin leaves the egg right at it)
    root = P[(s <= slv.info["s_separation"]) & (P[:, 1] < robe.y_top.min() - 0.02)]
    assert len(root) > 10
    rep = R.garment_overlap(root, robe.primitive.vertices, robe.primitive.faces, n_samples=50_000)
    assert rep["outside_count"] == 0
