"""Unit tests for meshforge.bake — the UV-space rasterizer and the small
geometry helpers built on top of it. Everything here uses a handful of
triangles and atlases of at most 64 px, so the whole file runs in well
under a second."""
import numpy as np
import pytest

from meshforge.bake import (
    PositionMap,
    bake_position_map,
    barycentric_2d,
    cylindrical_coords,
    dilate_into_padding,
    rasterize_face_ids,
    render_text_image,
)

ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


# --------------------------------------------------------------------------
# rasterize_face_ids
# --------------------------------------------------------------------------

def _lower_left_triangle():
    """One triangle covering the u + v <= 1 half of the atlas."""
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2]])
    return uvs, faces


def test_rasterize_single_triangle_claims_inside_texels_and_leaves_outside_empty():
    uvs, faces = _lower_left_triangle()
    ids = rasterize_face_ids(uvs, faces, 64)

    assert ids.shape == (64, 64)
    assert ids.dtype == np.int32
    # (5, 5) is deep inside the u + v < 1 half, (60, 60) is deep outside it
    assert ids[5, 5] == 0
    assert ids[60, 60] == -1
    # the covered half of a 64x64 atlas is roughly half the texels (the
    # outline makes coverage slightly conservative, never less than half)
    covered = (ids == 0).sum()
    assert 64 * 64 * 0.45 <= covered <= 64 * 64 * 0.6


def test_rasterize_ids_never_exceed_face_count():
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    ids = rasterize_face_ids(uvs, faces, 32)

    assert ids.max() <= len(faces) - 1
    assert ids.min() >= -1
    # a two-triangle quad over the whole atlas leaves no empty texel
    assert (ids >= 0).all()
    assert set(np.unique(ids)) == {0, 1}


# --------------------------------------------------------------------------
# barycentric_2d
# --------------------------------------------------------------------------

def _unit_triangle():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    c = np.array([[0.0, 1.0]])
    return a, b, c


def test_barycentric_centroid_is_one_third_each():
    a, b, c = _unit_triangle()
    bary = barycentric_2d((a + b + c) / 3.0, a, b, c)
    assert bary.shape == (1, 3)
    assert np.allclose(bary, 1.0 / 3.0, atol=1e-12)


def test_barycentric_vertices_are_one_hot():
    a, b, c = _unit_triangle()
    assert np.allclose(barycentric_2d(a, a, b, c), [[1.0, 0.0, 0.0]], atol=1e-12)
    assert np.allclose(barycentric_2d(b, a, b, c), [[0.0, 1.0, 0.0]], atol=1e-12)
    assert np.allclose(barycentric_2d(c, a, b, c), [[0.0, 0.0, 1.0]], atol=1e-12)


def test_barycentric_is_clamped_and_normalized_slightly_outside_the_triangle():
    """Outline texels can sit a hair outside their triangle; the result
    must still be a valid convex combination, not an extrapolation."""
    a, b, c = _unit_triangle()
    outside = np.array([[-0.1, 0.5], [0.6, 0.6], [0.5, -0.05]])
    bary = barycentric_2d(outside, a, b, c)

    assert bary.shape == (3, 3)
    assert np.all(bary >= 0.0)
    assert np.all(bary <= 1.0)
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(bary))


def test_barycentric_is_vectorized_over_many_points():
    a, b, c = _unit_triangle()
    rng = np.random.default_rng(0)
    pts = rng.random((50, 2)) * 0.5  # all inside the triangle
    bary = barycentric_2d(pts, np.repeat(a, 50, 0), np.repeat(b, 50, 0), np.repeat(c, 50, 0))
    # inside the triangle the coordinates reconstruct the point exactly
    recon = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    assert np.allclose(recon, pts, atol=1e-12)


# --------------------------------------------------------------------------
# bake_position_map
# --------------------------------------------------------------------------

def _unit_quad():
    """A unit quad in the z = 0 plane whose uv equals its (x, y), so the
    baked position of any texel is simply its uv center."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = vertices[:, :2].copy()
    return vertices, faces, uvs


def test_bake_position_map_interpolates_linearly_across_the_quad():
    vertices, faces, uvs = _unit_quad()
    size = 64
    pm = bake_position_map(vertices, faces, uvs, size=size)

    assert isinstance(pm, PositionMap)
    assert pm.size == size
    assert pm.position.shape == (size, size, 3)
    assert pm.position.dtype == np.float32

    # texel (x=16, y=48) has its center at uv = (16.5, 48.5) / 64 ~ (0.25, 0.75)
    x, y = 16, 48
    expected = np.array([(x + 0.5) / size, (y + 0.5) / size, 0.0])
    assert pm.mask[y, x]
    assert np.allclose(pm.position[y, x], expected, atol=1e-5)

    # and the same holds for every covered texel (position == texel-center uv)
    ys, xs = np.nonzero(pm.mask)
    centers = np.stack([(xs + 0.5) / size, (ys + 0.5) / size, np.zeros(len(xs))], axis=1)
    assert np.allclose(pm.position[ys, xs], centers, atol=1e-5)


def test_bake_position_map_covers_the_whole_atlas_for_a_full_quad():
    vertices, faces, uvs = _unit_quad()
    pm = bake_position_map(vertices, faces, uvs, size=64)
    assert pm.mask.mean() > 0.95
    assert pm.mask.dtype == bool
    assert (pm.face_id[pm.mask] >= 0).all()
    assert (pm.face_id[~pm.mask] == -1).all()


def test_bake_position_map_normals_are_unit_length_where_covered():
    vertices, faces, uvs = _unit_quad()
    # deliberately non-unit, tilted input normals: the baked ones must be renormalized
    vertex_normals = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, 0.5, 2.0]])
    pm = bake_position_map(vertices, faces, uvs, size=32, vertex_normals=vertex_normals)

    lengths = np.linalg.norm(pm.normal[pm.mask], axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)
    # uncovered texels (there are none here, but the contract is zeros) and
    # the normal map is zero when no vertex normals are passed at all
    pm_no_normals = bake_position_map(vertices, faces, uvs, size=32)
    assert not pm_no_normals.normal.any()


def test_bake_position_map_rejects_wrong_uv_shape():
    vertices, faces, _ = _unit_quad()
    with pytest.raises(ValueError, match="uvs must be"):
        bake_position_map(vertices, faces, np.zeros((3, 2)), size=16)


# --------------------------------------------------------------------------
# dilate_into_padding
# --------------------------------------------------------------------------

def test_dilate_spreads_known_texel_into_manhattan_rings():
    tex = np.zeros((9, 9, 3), dtype=np.float32)
    mask = np.zeros((9, 9), dtype=bool)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tex[4, 4] = red
    mask[4, 4] = True

    out = dilate_into_padding(tex, mask, padding=2)

    assert out.shape == tex.shape
    # ring 1: the four direct neighbours
    for y, x in ((3, 4), (5, 4), (4, 3), (4, 5)):
        assert np.allclose(out[y, x], red), (y, x)
    # ring 2: Manhattan distance 2, including the diagonals reached via two rings
    for y, x in ((2, 4), (6, 4), (4, 2), (4, 6), (3, 3), (3, 5), (5, 3), (5, 5)):
        assert np.allclose(out[y, x], red), (y, x)
    # Manhattan distance 3 is beyond padding=2
    for y, x in ((1, 4), (7, 4), (4, 1), (4, 7), (2, 3)):
        assert np.allclose(out[y, x], 0.0), (y, x)
    # far corners untouched
    for y, x in ((0, 0), (0, 8), (8, 0), (8, 8)):
        assert np.allclose(out[y, x], 0.0), (y, x)
    # exactly the Manhattan-radius-2 diamond (13 texels) is filled
    filled = out[..., 0] > 0
    assert filled.sum() == 13


def test_dilate_never_alters_known_texels_and_does_not_mutate_input():
    rng = np.random.default_rng(1)
    tex = rng.random((9, 9, 3)).astype(np.float32)
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:5, 3:6] = True
    original = tex.copy()

    out = dilate_into_padding(tex, mask, padding=3)

    assert np.array_equal(out[mask], original[mask])
    assert np.array_equal(tex, original)  # input untouched (the function copies)
    # padding=0 is a no-op
    assert np.array_equal(dilate_into_padding(tex, mask, padding=0), original)


def test_dilate_does_not_wrap_around_image_edges():
    """np.roll wraps; the implementation must mask the wrapped row/column so
    a texel on the left edge never bleeds into the right edge."""
    tex = np.zeros((9, 9, 3), dtype=np.float32)
    mask = np.zeros((9, 9), dtype=bool)
    tex[4, 0] = [0.0, 1.0, 0.0]
    mask[4, 0] = True

    out = dilate_into_padding(tex, mask, padding=1)
    assert np.allclose(out[4, 1], [0.0, 1.0, 0.0])
    assert np.allclose(out[4, 8], 0.0)


# --------------------------------------------------------------------------
# cylindrical_coords
# --------------------------------------------------------------------------

def test_cylindrical_coords_around_y_axis():
    """Y-axis cylinder, forward = +Z. Note on the sign convention: the
    implementation builds `side = cross(axis, forward)` = +X, so theta is
    positive toward **axis x forward** (+X here) — the docstring's
    "positive toward forward x axis" would be -X. We pin the behaviour
    the code actually produces."""
    axis_point = np.array([0.0, 0.0, 0.0])
    axis_dir = np.array([0.0, 1.0, 0.0])
    forward = np.array([0.0, 0.0, 1.0])
    points = np.array([
        [0.0, 0.0, 1.0],    # forward
        [1.0, 0.0, 0.0],    # +X
        [-1.0, 0.0, 0.0],   # -X
        [0.0, 0.0, -1.0],   # backward
        [0.0, 2.0, 3.0],    # forward, raised by 2, radius 3
        [3.0, -1.5, 4.0],   # radius 5 off-axis, below the axis point
    ])
    theta, height, radius = cylindrical_coords(points, axis_point, axis_dir, forward)

    assert np.isclose(theta[0], 0.0, atol=1e-12)
    assert np.isclose(theta[1], +np.pi / 2, atol=1e-12)
    assert np.isclose(theta[2], -np.pi / 2, atol=1e-12)
    assert np.isclose(abs(theta[3]), np.pi, atol=1e-12)
    assert np.isclose(theta[4], 0.0, atol=1e-12)

    assert np.allclose(height, points[:, 1], atol=1e-12)
    assert np.allclose(radius, np.hypot(points[:, 0], points[:, 2]), atol=1e-12)
    assert np.all(theta > -np.pi) and np.all(theta <= np.pi)


def test_cylindrical_coords_is_relative_to_axis_point_and_projects_forward():
    """A forward vector with a component along the axis is projected out,
    and height/radius are measured from axis_point, not the origin."""
    axis_point = np.array([1.0, 2.0, 3.0])
    axis_dir = np.array([0.0, 2.0, 0.0])        # non-unit on purpose
    forward = np.array([0.0, 5.0, 1.0])         # mostly along the axis; projects to +Z
    points = axis_point + np.array([[0.0, 0.5, 2.0], [2.0, -1.0, 0.0]])
    theta, height, radius = cylindrical_coords(points, axis_point, axis_dir, forward)

    assert np.allclose(theta, [0.0, np.pi / 2], atol=1e-12)
    assert np.allclose(height, [0.5, -1.0], atol=1e-12)
    assert np.allclose(radius, [2.0, 2.0], atol=1e-12)


# --------------------------------------------------------------------------
# render_text_image
# --------------------------------------------------------------------------

def test_render_text_image_with_real_font_produces_opaque_glyph_pixels():
    img = render_text_image("OWL", 128, 64, [ARIAL_BOLD], fill=(255, 255, 255, 255))

    assert img.mode == "RGBA"
    assert img.size == (128, 64)
    alpha = np.asarray(img)[..., 3]
    opaque = alpha > 0
    # glyphs are sized to ~80 % of the box, so they cover a meaningful area
    assert 0.05 < opaque.mean() < 0.9
    # transparent background: the corners are untouched
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0
    # centered: the glyph bounding box straddles the middle of the image
    ys, xs = np.nonzero(opaque)
    assert xs.min() < 64 < xs.max()
    assert ys.min() < 32 < ys.max()


def test_render_text_image_falls_back_when_no_font_loads():
    img = render_text_image("OWL", 96, 40, [])
    assert img.mode == "RGBA"
    assert img.size == (96, 40)
    assert (np.asarray(img)[..., 3] > 0).any()

    # an unreadable path is skipped, not fatal, and the real font still wins
    img2 = render_text_image("OWL", 96, 40, ["/nonexistent/font.ttf", ARIAL_BOLD])
    assert img2.size == (96, 40)
    assert (np.asarray(img2)[..., 3] > 0).sum() > (np.asarray(img)[..., 3] > 0).sum()
