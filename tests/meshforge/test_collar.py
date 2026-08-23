"""Tests for meshforge.collar — the band frame and the lettering decal.

These are built on a synthetic collar (a cylinder band with a known,
deliberately tilted-and-curved red facing) rather than the owl, so the
expected centreline is known in closed form and the assertions can be
about accuracy rather than "looks right".
"""
import numpy as np
import pytest
import trimesh

from meshforge.bake import render_text_image
from meshforge.collar import (
    BandFrame,
    build_collar_decal,
    despike_band,
    measure_band_frame,
)
from meshforge.regions import RegionMap

ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


# --------------------------------------------------------------------------
# a synthetic owl-ish collar
# --------------------------------------------------------------------------

def _band_centre(theta: np.ndarray) -> np.ndarray:
    """The truth the frame has to recover: a crescent that dips in front."""
    return 0.10 * np.cos(theta) + 0.02 * theta


def make_collar(n_theta: int = 160, n_y: int = 26, radius: float = 1.0,
                half_height: float = 0.10):
    """A cylinder strip whose red facing follows `_band_centre`, with white
    above and below it, plus dark rims just outside the red."""
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    s = np.linspace(-3.0, 3.0, n_y)                     # in units of half_height
    T, S = np.meshgrid(th, s, indexing="ij")
    Y = _band_centre(T) + S * half_height
    V = np.stack([radius * np.sin(T), Y, radius * np.cos(T)], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(n_theta):
        j = (i + 1) % n_theta
        for k in range(n_y - 1):
            a, b = i * n_y + k, j * n_y + k
            faces.append([a, b, a + 1])
            faces.append([b, b + 1, a + 1])
    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(faces), process=False)

    sflat = S.reshape(-1)
    colors = np.tile(np.array([0.95, 0.94, 0.92]), (len(V), 1))   # white
    colors[np.abs(sflat) <= 1.0] = (0.72, 0.11, 0.09)             # red facing
    colors[(np.abs(sflat) > 1.0) & (np.abs(sflat) <= 1.4)] = (0.10, 0.10, 0.11)  # rims
    return mesh, colors, sflat


def make_region_map(mesh, colors) -> RegionMap:
    """A RegionMap with the *coarse* collar description deliberately wrong —
    too tall and flat — which is the situation on the real asset."""
    V = np.asarray(mesh.vertices)
    bmin, bmax = mesh.bounds
    f = (V - bmin) / (bmax - bmin)
    n_bins = 36
    collar = {
        "theta_bins": np.linspace(-np.pi, np.pi, n_bins + 1),
        "y_lo": np.full(n_bins, 0.18),
        "y_hi": np.full(n_bins, 0.82),
        "centre_xz": np.array([0.5, 0.5]),
        "theta": np.zeros(len(V)),
    }
    labels = np.full(len(V), "neck", dtype=object)
    return RegionMap(labels=labels, joints=[], fractions=f, collar=collar, stats={})


@pytest.fixture(scope="module")
def collar_fixture():
    mesh, colors, s = make_collar()
    return mesh, colors, s, make_region_map(mesh, colors)


# --------------------------------------------------------------------------
# measure_band_frame
# --------------------------------------------------------------------------

def test_frame_recovers_the_curved_centreline_the_coarse_one_misses(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    frame = measure_band_frame(mesh, rm, colors, span_deg=60.0)

    bmin, bmax = mesh.bounds
    size = bmax - bmin
    th = np.linspace(-np.radians(55), np.radians(55), 40)
    got = frame.mid(th) * size[1] + bmin[1]              # back to world y
    want = _band_centre(th)
    assert np.abs(got - want).max() < 0.02               # 20 % of the band's half height

    # the coarse frame is flat, so it cannot track the 0.2-unit swing at all
    coarse = np.full_like(th, 0.5) * size[1] + bmin[1]
    assert np.abs(coarse - want).max() > np.abs(got - want).max() * 3


def test_frame_height_matches_the_red_facing_not_the_dark_window(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    frame = measure_band_frame(mesh, rm, colors, span_deg=60.0)
    size = mesh.bounds[1] - mesh.bounds[0]
    # the true facing is 2 x half_height tall; the 3rd/97th percentile rims
    # trim a little off each end, so expect a slight under-read, never an
    # over-read (over-reading is exactly the coarse frame's failure)
    measured = frame.height * size[1]
    assert 0.75 * 0.20 <= measured <= 1.02 * 0.20
    assert frame.diagnostics["coarse_height"] > measured * 2         # the wrong one


def test_frame_clamps_outside_its_fitted_span(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    frame = measure_band_frame(mesh, rm, colors, span_deg=45.0)
    lo_edge, _ = frame.bounds(np.array([np.radians(45)]))
    lo_far, _ = frame.bounds(np.array([np.radians(170)]))
    assert lo_far == pytest.approx(lo_edge)              # clamped, not extrapolated


def test_frame_falls_back_when_there_is_no_red_to_measure(collar_fixture):
    mesh, _colors, _s, rm = collar_fixture
    white = np.tile(np.array([0.9, 0.9, 0.9]), (len(mesh.vertices), 1))
    frame = measure_band_frame(mesh, rm, white, span_deg=60.0)
    assert frame.diagnostics.get("fallback") is True
    lo, hi = frame.bounds(np.array([0.0]))
    assert hi > lo                                       # still a usable band


# --------------------------------------------------------------------------
# build_collar_decal
# --------------------------------------------------------------------------

def test_decal_uvs_follow_the_band_rather_than_a_constant_height(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    decal = build_collar_decal(mesh, rm, text="AI-CCORE", colors=colors,
                               theta_span_deg=50.0, font_paths=[ARIAL_BOLD])
    prim = decal.primitive
    V = prim.vertices
    theta = np.arctan2(V[:, 0], V[:, 2])
    v = prim.uvs[:, 1]

    # v = 0.5 must sit on the band centre at every angle, so the world height
    # of the v = 0.5 contour has to swing with the band, not stay flat.
    for lo_deg, hi_deg in [(-45, -25), (-10, 10), (25, 45)]:
        m = (theta > np.radians(lo_deg)) & (theta < np.radians(hi_deg))
        assert m.sum() > 20
        slope, intercept = np.polyfit(v[m], V[m, 1], 1)
        y_half = slope * 0.5 + intercept
        want = _band_centre(np.array([np.radians((lo_deg + hi_deg) / 2)]))[0]
        assert abs(y_half - want) < 0.03


def test_decal_patch_stays_inside_the_red_facing(collar_fixture):
    mesh, colors, s, rm = collar_fixture
    decal = build_collar_decal(mesh, rm, text="AI-CCORE", colors=colors,
                               theta_span_deg=50.0, font_paths=[ARIAL_BOLD])
    # every source vertex the patch used must be red (|s| <= 1), never rim or white
    assert np.abs(s[decal.source_vertex_index]).max() <= 1.0


def test_decal_image_is_sized_to_the_band_aspect(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    decal = build_collar_decal(mesh, rm, text="AI-CCORE", colors=colors,
                               theta_span_deg=50.0, font_paths=[ARIAL_BOLD],
                               image_height=512)
    w, h = decal.info["image"]
    assert h == 512
    assert w / h == pytest.approx(decal.info["aspect"], rel=0.01)


def test_plate_is_opaque_in_the_middle_and_transparent_at_the_patch_edge(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    decal = build_collar_decal(mesh, rm, text="AI-CCORE", colors=colors,
                               theta_span_deg=50.0, font_paths=[ARIAL_BOLD],
                               plate_color=(0.70, 0.11, 0.09))
    img = np.asarray(decal.primitive.material.base_color_image)
    h, w = img.shape[:2]
    v0, v1 = decal.info["v_range"]
    assert img[h // 2, w // 2, 3] == 255
    # alpha must have died by the row where the geometry stops
    assert img[max(int(v0 * h) - 1, 0), w // 2, 3] == 0
    assert img[min(int(v1 * h) + 1, h - 1), w // 2, 3] == 0
    assert decal.info["plate"] is True


def test_no_plate_leaves_the_background_transparent(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    decal = build_collar_decal(mesh, rm, text="AI-CCORE", colors=colors,
                               theta_span_deg=50.0, font_paths=[ARIAL_BOLD])
    img = np.asarray(decal.primitive.material.base_color_image)
    assert img[2, 2, 3] == 0
    assert decal.info["plate"] is False


def test_decal_requires_colors_or_a_frame(collar_fixture):
    mesh, _colors, _s, rm = collar_fixture
    with pytest.raises(ValueError, match="colors"):
        build_collar_decal(mesh, rm, text="AI-CCORE", font_paths=[ARIAL_BOLD])


def test_decal_accepts_a_prebuilt_frame(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    frame = measure_band_frame(mesh, rm, colors, span_deg=60.0)
    decal = build_collar_decal(mesh, rm, text="AI-CCORE", frame=frame,
                               theta_span_deg=50.0, font_paths=[ARIAL_BOLD])
    assert isinstance(frame, BandFrame)
    assert decal.info["band"]["bins_used"] > 10


# --------------------------------------------------------------------------
# despike_band
# --------------------------------------------------------------------------

def _bump(mesh, idx, height=0.02):
    V = np.asarray(mesh.vertices).copy()
    V[idx] += np.asarray(mesh.vertex_normals)[idx] * height
    return V, trimesh.Trimesh(vertices=V, faces=mesh.faces.copy(), process=False)


def test_despike_removes_an_isolated_spike_on_the_band(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    frame = measure_band_frame(mesh, rm, colors, span_deg=60.0)
    V0 = np.asarray(mesh.vertices)
    theta = np.arctan2(V0[:, 0], V0[:, 2])
    spike = np.where((np.abs(theta) < 0.2)
                     & (np.abs(V0[:, 1] - _band_centre(theta)) < 0.03))[0][:1]
    V, bumpy = _bump(mesh, spike)

    fixed, moved = despike_band(bumpy, rm, frame, threshold=0.003)
    assert moved > 0
    residual = np.linalg.norm(np.asarray(fixed.vertices)[spike] - V0[spike], axis=1).max()
    assert residual < 0.02 * 0.25            # a lone spike comes almost all the way down

    # vertices far from the band must not move at all
    far = np.abs(V0[:, 1] - _band_centre(theta)) > 0.25
    assert np.allclose(np.asarray(fixed.vertices)[far], V[far])


def test_despike_only_partly_lowers_a_broad_plateau(collar_fixture):
    """Honest limit: the clamp is on the Laplacian residual, and the middle
    of a raised patch has almost no residual — its neighbours are raised
    too. That is why the decal also floats on a plate rather than relying
    on this pass alone."""
    mesh, colors, _s, rm = collar_fixture
    frame = measure_band_frame(mesh, rm, colors, span_deg=60.0)
    V0 = np.asarray(mesh.vertices)
    theta = np.arctan2(V0[:, 0], V0[:, 2])
    plateau = np.where((np.abs(theta) < 0.2)
                       & (np.abs(V0[:, 1] - _band_centre(theta)) < 0.03))[0][:6]
    _V, bumpy = _bump(mesh, plateau)

    fixed, moved = despike_band(bumpy, rm, frame, threshold=0.003)
    assert moved > 0
    residual = np.linalg.norm(np.asarray(fixed.vertices)[plateau] - V0[plateau], axis=1).max()
    assert 0.02 * 0.2 < residual < 0.02 * 0.8


def test_despike_is_a_noop_on_a_smooth_band(collar_fixture):
    mesh, colors, _s, rm = collar_fixture
    frame = measure_band_frame(mesh, rm, colors, span_deg=60.0)
    fixed, moved = despike_band(mesh, rm, frame, threshold=0.01)
    assert moved == 0
    assert np.allclose(fixed.vertices, mesh.vertices)


# --------------------------------------------------------------------------
# render_text_image tracking / fit
# --------------------------------------------------------------------------

def _ink_box(img):
    a = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(a > 128)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_tracking_widens_the_word_without_making_it_taller():
    tight = render_text_image("AI-CCORE", 1200, 300, [ARIAL_BOLD], tracking=0.0)
    loose = render_text_image("AI-CCORE", 1200, 300, [ARIAL_BOLD], tracking=0.18)
    tx0, ty0, tx1, ty1 = _ink_box(tight)
    lx0, ly0, lx1, ly1 = _ink_box(loose)
    assert (lx1 - lx0) >= (tx1 - tx0) * 0.9          # still fills the width budget
    assert (ly1 - ly0) < (ty1 - ty0)                 # so the glyphs had to shrink


def test_fit_controls_cap_height_as_a_fraction_of_the_canvas():
    tall = render_text_image("AB", 2000, 400, [ARIAL_BOLD], fit=(0.9, 0.8))
    short = render_text_image("AB", 2000, 400, [ARIAL_BOLD], fit=(0.9, 0.4))
    _, ty0, _, ty1 = _ink_box(tall)
    _, sy0, _, sy1 = _ink_box(short)
    assert (ty1 - ty0) == pytest.approx(400 * 0.8, rel=0.12)
    assert (sy1 - sy0) == pytest.approx(400 * 0.4, rel=0.12)


def test_text_stays_centred_in_the_canvas_with_tracking():
    img = render_text_image("AI-CCORE", 1600, 400, [ARIAL_BOLD], tracking=0.08)
    x0, y0, x1, y1 = _ink_box(img)
    assert (x0 + x1) / 2 == pytest.approx(800, abs=25)
    assert (y0 + y1) / 2 == pytest.approx(200, abs=15)
