"""Tests for meshforge.kente — the garment decal.

Built on a synthetic torso (a UV sphere, labelled the way regions.classify
would label a real one) rather than the owl, so this doesn't depend on the
hi-res source asset — same reasoning as test_collar.py's synthetic collar.
"""
import numpy as np
import pytest
import trimesh

from meshforge.kente import build_kente_decal, build_kente_sash_decal
from meshforge.regions import RegionMap
from meshforge.textile import ASANTE_GOLD, WeaveParams


def make_torso(subdivisions: int = 3, radius: float = 1.0):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return mesh


def make_region_map(mesh: trimesh.Trimesh, chest_below_equator_frac: float = 0.65) -> RegionMap:
    """Label everything below `chest_below_equator_frac` of the height as
    "chest", the rest "head" — a stand-in for regions.classify's real
    per-vertex labels."""
    V = np.asarray(mesh.vertices)
    bmin, bmax = mesh.bounds
    f = (V - bmin) / (bmax - bmin)
    labels = np.where(f[:, 1] < chest_below_equator_frac, "chest", "head").astype(object)
    collar = {"centre_xz": np.array([0.5, 0.5])}
    return RegionMap(labels=labels, joints=[], fractions=f, collar=collar, stats={})


@pytest.fixture(scope="module")
def torso_fixture():
    mesh = make_torso()
    return mesh, make_region_map(mesh)


SMALL_WEAVE = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=2, block_px=12, seed=0)


def test_decal_covers_only_the_selected_region(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=64)
    # every source vertex the decal reuses must actually be labelled "chest"
    assert np.all(rm.labels[decal.source_vertex_index] == "chest")
    assert decal.info["vertices"] == len(decal.source_vertex_index)
    assert decal.primitive.vertices.shape == (len(decal.source_vertex_index), 3)


def test_decal_is_offset_outward_along_normals(torso_fixture):
    mesh, rm = torso_fixture
    offset = 0.01
    # smoothing is on by default and deliberately moves vertices (that's
    # its whole job) — isolate the pure offset behaviour here with it off,
    # the smoothing effect itself gets its own test below.
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), offset=offset,
                              texture_size=64, smooth_iterations=0)
    # unwrapped.mesh.vertices (pre-offset) are gathered from the *sub-mesh*
    # by source_index, i.e. exactly mesh.vertices[decal.source_vertex_index]
    # before the offset is applied — verify that identity, then that the
    # offset moved every vertex out along its own normal by the exact amount.
    src = mesh.vertices[decal.source_vertex_index]
    normals = decal.primitive.normals
    assert np.allclose(decal.primitive.vertices, src + normals * offset, atol=1e-9)
    # a sphere's own surface normal points radially outward, so the offset
    # decal must sit strictly farther from the centre than its source
    centre = mesh.bounds.mean(axis=0)
    d_src = np.linalg.norm(src - centre, axis=1)
    d_decal = np.linalg.norm(decal.primitive.vertices - centre, axis=1)
    assert np.all(d_decal > d_src)


def test_smoothing_removes_fine_bumps_but_keeps_the_overall_shape(torso_fixture):
    """Reproduces, on a controlled synthetic surface, the real bug found on
    the owl: its belly carries a raised circuit-trace emboss (part of the
    brand mark), and a decal that's just an offset duplicate of the raw
    surface faithfully traces those bumps — rendered, that looked like odd
    veins running through the kente fabric, not draped cloth. Build a
    bumpy sphere (base radius 1, plus a known high-frequency sinusoidal
    bump), and confirm smoothing pulls the decal's radius back toward the
    smooth base radius while an unsmoothed decal keeps the bump."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    V = mesh.vertices
    theta = np.arctan2(V[:, 0], V[:, 2])
    bump = 0.15 * np.sin(theta * 24.0)  # high-frequency: 24 ripples around the equator
    bumpy = trimesh.Trimesh(vertices=V * (1.0 + bump[:, None]), faces=mesh.faces, process=False)
    rm = make_region_map(bumpy, chest_below_equator_frac=1.1)  # everything is "chest"

    unsmoothed = build_kente_decal(bumpy, rm, SMALL_WEAVE, regions_included=("chest",), offset=0.0,
                                   texture_size=64, smooth_iterations=0, erode_rings=0, hem_percentile=None)
    smoothed = build_kente_decal(bumpy, rm, SMALL_WEAVE, regions_included=("chest",), offset=0.0,
                                 texture_size=64, smooth_iterations=15, smooth_factor=0.5,
                                 erode_rings=0, hem_percentile=None)

    centre = bumpy.bounds.mean(axis=0)
    r_unsmoothed = np.linalg.norm(unsmoothed.primitive.vertices - centre, axis=1)
    r_smoothed = np.linalg.norm(smoothed.primitive.vertices - centre, axis=1)
    # unsmoothed keeps roughly the full bump amplitude (radius swings from
    # ~0.85 to ~1.15); smoothing must shrink that spread substantially
    assert r_unsmoothed.std() > 0.05
    assert r_smoothed.std() < r_unsmoothed.std() * 0.5


def test_uvs_are_a_low_distortion_unwrap_not_a_projection(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=64)
    uvs = decal.primitive.uvs
    # xatlas packs into [0, 1] but doesn't have to touch every edge exactly
    # (padding, multiple charts) — the contract is "valid and well spread",
    # not "touches all four corners" the way a direct bbox projection would.
    assert uvs.min() >= 0.0 and uvs.max() <= 1.0
    assert (uvs.max(axis=0) - uvs.min(axis=0)).min() > 0.5  # uses most of the square, both axes


def test_baked_image_matches_requested_texture_size(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=96)
    assert decal.info["image"] == (96, 96)
    assert decal.primitive.material.base_color_image.size == (96, 96)


def test_baked_texture_has_broad_coverage_and_no_uncovered_holes_after_dilation(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=96, dilate=8)
    # coverage should be a healthy fraction of the atlas, not a sliver
    assert decal.info["coverage"] > 0.15
    img = np.asarray(decal.primitive.material.base_color_image)
    # dilation fills every texel's alpha to fully opaque or fully transparent
    # in this synthetic case there should be no lingering half-filled fringe
    assert set(np.unique(img[..., 3])) <= {0, 255}


def test_normal_map_is_on_by_default(torso_fixture):
    # bake_normal defaults to True: an earlier version defaulted it off
    # because dilate_into_padding was blending unrelated normal directions
    # across xatlas chart gaps (fine for color, wrong for a vector field),
    # producing incoherent shading on the real owl. Fixed by
    # _nearest_valid_fill (exact nearest-neighbor propagation instead of
    # ring-averaging) and verified against the real owl, not just this
    # synthetic torso — see the module docstring and test_nearest_valid_fill_*.
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=96)
    assert decal.primitive.material.normal_image is not None


def test_normal_map_is_a_valid_tangent_space_image(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=96)
    normal_img = decal.primitive.material.normal_image
    assert normal_img.size == (96, 96)
    arr = np.asarray(normal_img).astype(np.float64)
    assert arr.shape == (96, 96, 4)
    # decode back to a unit-ish vector and confirm every covered texel's Z
    # (the "mostly facing outward" component) dominates — a tangent-space
    # normal map that had its basis wrong would still decode to *some*
    # vector, but a correct one stays close to (0, 0, 1) since bump_strength
    # is small; this is the property a basis mix-up would break first.
    n = arr[..., :3] / 255.0 * 2.0 - 1.0
    z = n[..., 2]
    assert np.median(z) > 0.9


def test_normal_map_alpha_is_opaque(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=96)
    arr = np.asarray(decal.primitive.material.normal_image)
    assert np.all(arr[..., 3] == 255)


def test_normal_map_can_still_be_disabled(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=96,
                              bake_normal=False)
    assert decal.primitive.material.normal_image is None


def test_nearest_valid_fill_copies_the_nearest_value_not_a_blend():
    from meshforge.kente import _nearest_valid_fill

    # two unrelated "islands" (like two different xatlas charts) at
    # opposite corners of a small canvas, everything else unknown
    tex = np.zeros((10, 10, 3), dtype=np.float64)
    mask = np.zeros((10, 10), dtype=bool)
    tex[0, 0] = [1.0, 0.0, 0.0]
    mask[0, 0] = True
    tex[9, 9] = [0.0, 0.0, 1.0]
    mask[9, 9] = True

    filled = _nearest_valid_fill(tex, mask)
    # a point roughly equidistant between the two islands must still take
    # on EXACTLY one island's value, not an average of both — that average
    # (a plausible-looking but meaningless blend) is exactly the bug this
    # function replaces dilate_into_padding to avoid for vector data
    midpoint = filled[4, 5]
    is_pure_red = np.allclose(midpoint, [1.0, 0.0, 0.0])
    is_pure_blue = np.allclose(midpoint, [0.0, 0.0, 1.0])
    assert is_pure_red or is_pure_blue
    assert not np.allclose(midpoint, [0.5, 0.0, 0.5])


def test_nearest_valid_fill_is_a_noop_when_fully_covered():
    from meshforge.kente import _nearest_valid_fill

    tex = np.random.default_rng(0).random((6, 6, 3))
    mask = np.ones((6, 6), dtype=bool)
    assert np.array_equal(_nearest_valid_fill(tex, mask), tex)


def test_weave_color_at_is_position_not_layout_dependent():
    """This is the actual invariant build_kente_decal's fix depends on:
    baking by real 3D position (via weave_color_at on a scattered,
    xatlas-ordered set of texel positions) must agree exactly with what a
    plain dense-grid render (generate_weave) produces at the same
    coordinates — i.e. the color at a point never depends on what order or
    grouping it was evaluated in, only on the point itself. That is what
    makes the pattern unbroken across arbitrary chart boundaries."""
    from meshforge.textile import generate_weave, weave_color_at

    grid = generate_weave(64, 48, SMALL_WEAVE)
    ys, xs = np.meshgrid(np.arange(48), np.arange(64), indexing="ij")
    # shuffle into an arbitrary (non-grid-ordered) flat scatter, exactly
    # the kind bake_position_map hands weave_color_at for real texels
    rng = np.random.default_rng(0)
    order = rng.permutation(xs.size)
    scattered = weave_color_at(xs.ravel()[order].astype(np.float64), ys.ravel()[order].astype(np.float64), SMALL_WEAVE)
    expected = grid[ys.ravel()[order], xs.ravel()[order]]
    assert np.array_equal(scattered, expected)


def test_raises_when_region_is_too_small(torso_fixture):
    mesh, rm = torso_fixture
    with pytest.raises(ValueError):
        build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("wing_left",))


def test_faces_reindex_within_bounds(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_decal(mesh, rm, SMALL_WEAVE, regions_included=("chest",), texture_size=64)
    n = len(decal.source_vertex_index)
    assert decal.primitive.faces.min() >= 0
    assert decal.primitive.faces.max() < n


# --------------------------------------------------------------------------
# build_kente_sash_decal — the diagonal-stole alternative
# --------------------------------------------------------------------------

SASH_WEAVE = WeaveParams(colorway=ASANTE_GOLD, strip_px=8, strip_cycle=2, block_px=12, seed=0)


def _sash_kwargs(**overrides):
    # band_half_width_frac=0.4: this coarse synthetic torso (a few hundred
    # vertices) needs a wider fraction than the real owl mesh's default
    # (0.16) to reliably capture >=10 full-triangle faces in the band —
    # a fixture-resolution matter, not something about the sash logic
    # itself (confirmed: the same fractions on the real ~60k-vertex owl
    # mesh capture plenty of faces at the production default).
    base = dict(shoulder_regions=("chest",), hip_regions=("chest",), texture_size=64,
               erode_rings=0, smooth_iterations=0, band_half_width_frac=0.4)
    base.update(overrides)
    return base


def test_sash_anchors_are_diagonal_from_each_other(torso_fixture):
    from meshforge.kente import _sash_anchors

    mesh, rm = torso_fixture
    shoulder, hip = _sash_anchors(mesh, rm, ("chest",), ("chest",), "left", "right", 85.0, 15.0)
    # shoulder: left side (x < centre), higher up. hip: right side, lower.
    assert shoulder[0] < hip[0]
    assert shoulder[1] > hip[1]


def test_sash_selects_only_points_within_half_width(torso_fixture):
    from meshforge.kente import _point_segment_distance, _sash_anchors

    mesh, rm = torso_fixture
    decal = build_kente_sash_decal(mesh, rm, SASH_WEAVE, **_sash_kwargs(band_half_width_frac=0.4))
    shoulder, hip = _sash_anchors(mesh, rm, ("chest",), ("chest",), "left", "right", 85.0, 15.0)
    segment_len = np.linalg.norm(hip - shoulder)
    half_width = segment_len * 0.4
    src = mesh.vertices[decal.source_vertex_index]
    dist, _t = _point_segment_distance(src, shoulder, hip)
    # a little slack: erode_rings=0 here, but source vertices are the
    # *pre-smoothed* raw selection, which is exactly what was thresholded
    assert np.all(dist <= half_width + 1e-9)


def test_sash_is_narrower_than_the_vest_by_default(torso_fixture):
    mesh, rm = torso_fixture
    sash = build_kente_sash_decal(mesh, rm, SASH_WEAVE, **_sash_kwargs())
    vest = build_kente_decal(mesh, rm, SASH_WEAVE, regions_included=("chest",), texture_size=64,
                             erode_rings=0, hem_percentile=None, smooth_iterations=0)
    assert sash.info["vertices"] < vest.info["vertices"]


def test_sash_material_and_image_are_valid(torso_fixture):
    mesh, rm = torso_fixture
    decal = build_kente_sash_decal(mesh, rm, SASH_WEAVE, **_sash_kwargs())
    assert decal.primitive.material.base_color_image.size == (64, 64)
    assert decal.primitive.material.normal_image is not None  # bake_normal defaults True
    assert decal.info["coverage"] > 0.0


def test_sash_raises_when_band_is_too_narrow():
    mesh, rm = None, None
    import trimesh as _trimesh
    from meshforge.regions import RegionMap as _RegionMap

    mesh = _trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    V = np.asarray(mesh.vertices)
    bmin, bmax = mesh.bounds
    f = (V - bmin) / (bmax - bmin)
    labels = np.where(f[:, 1] < 0.65, "chest", "head").astype(object)
    rm = _RegionMap(labels=labels, joints=[], fractions=f, collar={"centre_xz": np.array([0.5, 0.5])}, stats={})
    with pytest.raises(ValueError):
        build_kente_sash_decal(mesh, rm, SASH_WEAVE, **_sash_kwargs(band_half_width_frac=0.001))


def test_sash_raises_when_a_required_region_is_missing(torso_fixture):
    mesh, rm = torso_fixture
    with pytest.raises(ValueError):
        build_kente_sash_decal(mesh, rm, SASH_WEAVE, **_sash_kwargs(hip_regions=("nonexistent",)))


def test_drop_degenerate_faces_removes_zero_area_triangles_and_fixes_normals():
    """Reproduces the real bug found on the owl: a near-zero-area sliver
    face (three vertices within 1e-9 of each other) makes trimesh compute
    a zero vertex normal for its (otherwise unshared) vertices, which the
    exporter's own "normals unit length" validator catches. Engineer that
    exact case — a good quad plus one deliberately degenerate triangle —
    and confirm the cleanup removes it and the mapping back to the
    original mesh stays correct."""
    from meshforge.kente import _drop_degenerate_faces

    # a simple quad (2 good triangles) at indices 0-3, plus a degenerate
    # sliver (indices 4, 5, 6, nearly coincident) touched by no other face
    V = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # good quad
        [5.0, 5.0, 5.0], [5.0 + 1e-9, 5.0, 5.0], [5.0, 5.0 + 1e-9, 5.0],     # degenerate sliver
    ])
    used = np.arange(7)  # pretend every vertex maps to itself in "the original mesh"
    sub_faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])

    new_verts, new_used, new_faces = _drop_degenerate_faces(V, used, sub_faces)

    # the degenerate triangle's 3 vertices must be gone
    assert len(new_used) == 4
    assert set(new_used.tolist()) == {0, 1, 2, 3}
    # the returned local vertex positions must match what `used` implies
    assert np.array_equal(new_verts, V[new_used])
    # the two good faces must survive, correctly reindexed into the new,
    # smaller vertex array
    assert len(new_faces) == 2
    result_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    lens = np.linalg.norm(result_mesh.vertex_normals, axis=1)
    assert np.allclose(lens, 1.0)
    # `new_used` must still correctly map each surviving *local* vertex
    # back to its *original* mesh index — not just have the right count
    assert np.array_equal(np.sort(new_used), np.array([0, 1, 2, 3]))


def test_repair_zero_normals_fixes_only_the_bad_rows():
    from meshforge.kente import _repair_zero_normals

    positions = np.array([
        [1.0, 0.0, 0.0],   # good, valid normal
        [0.0, 1.0, 0.0],   # bad — will get a zero-length normal
        [-1.0, 0.0, 0.0],  # good, valid normal
    ])
    normals = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],   # the defect: an averaged-to-zero vertex normal
        [-1.0, 0.0, 0.0],
    ])
    repaired = _repair_zero_normals(normals, positions)

    # untouched rows must be byte-identical, not just "close"
    assert np.array_equal(repaired[0], normals[0])
    assert np.array_equal(repaired[2], normals[2])
    # the bad row must now be unit length and point outward from the
    # positions' own centroid (here, the centroid is [0, 1/3, 0], so
    # position [0,1,0] relative to it points in +y)
    assert np.isclose(np.linalg.norm(repaired[1]), 1.0)
    assert repaired[1][1] > 0


def test_repair_zero_normals_is_a_noop_when_nothing_is_bad():
    from meshforge.kente import _repair_zero_normals

    positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.array_equal(_repair_zero_normals(normals, positions), normals)
