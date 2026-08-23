"""Unit tests for meshforge.retopo — weld / debris removal / decimation /
xatlas unwrap / 3D surface-sample color transfer, all on tiny meshes."""
import numpy as np
import pytest
import trimesh
from PIL import Image

from meshforge.retopo import (
    SurfaceColorSampler,
    UnwrappedMesh,
    bake_texture,
    decimate,
    drop_small_components,
    unwrap,
    weld,
)

RED = np.array([1.0, 0.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])


def boundary_edge_count(mesh):
    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def half_red_half_blue_image(size=8):
    img = Image.new("RGB", (size, size), (0, 0, 255))
    for x in range(size // 2):
        for y in range(size):
            img.putpixel((x, y), (255, 0, 0))
    return img


def textured_quad():
    """A unit quad in z = 0 whose uv equals its (x, y); the left half of
    the texture is red, the right half blue, so color depends only on x."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = vertices[:, :2].copy()
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=half_red_half_blue_image())
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


# --------------------------------------------------------------------------
# weld
# --------------------------------------------------------------------------

def test_weld_merges_duplicate_seam_into_closed_tetrahedron(duplicate_seam_mesh):
    assert len(duplicate_seam_mesh.vertices) == 6
    welded = weld(duplicate_seam_mesh, min_component_faces=1)

    assert len(welded.vertices) == 4
    assert len(welded.faces) == 4
    assert boundary_edge_count(welded) == 0
    assert welded.is_watertight
    # the input is not mutated
    assert len(duplicate_seam_mesh.vertices) == 6


def test_weld_raises_on_a_genuinely_open_surface():
    open_triangle = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    with pytest.raises(ValueError, match="boundary"):
        weld(open_triangle, min_component_faces=1)


def test_weld_drops_debris_below_min_component_faces(debris_mesh):
    welded = weld(debris_mesh, min_component_faces=2)
    assert len(welded.faces) == 12
    assert boundary_edge_count(welded) == 0


# --------------------------------------------------------------------------
# drop_small_components
# --------------------------------------------------------------------------

def test_drop_small_components_keeps_the_box_and_drops_the_shard(debris_mesh):
    assert len(debris_mesh.faces) == 13
    kept = drop_small_components(debris_mesh, min_faces=2)
    assert len(kept.faces) == 12
    assert len(kept.vertices) == 8
    assert kept.bounds.max() < 2.0  # the shard at (10, 10) is gone


def test_drop_small_components_is_a_noop_when_everything_qualifies(debris_mesh):
    same = drop_small_components(debris_mesh, min_faces=1)
    assert same is debris_mesh


def test_drop_small_components_raises_when_nothing_survives(debris_mesh):
    with pytest.raises(ValueError, match="every component"):
        drop_small_components(debris_mesh, min_faces=100)


# --------------------------------------------------------------------------
# decimate
# --------------------------------------------------------------------------

def test_decimate_icosphere_stays_a_single_closed_shell():
    ico = trimesh.creation.icosphere(subdivisions=4)
    assert len(ico.faces) == 5120
    out = decimate(ico, 400)

    assert 300 <= len(out.faces) <= 450
    assert boundary_edge_count(out) == 0
    assert out.is_watertight
    assert len(out.split(only_watertight=False)) == 1
    # still a sphere: every surviving vertex is near the unit radius (quadric
    # placement may push vertices ~1 % outside the original surface)
    radii = np.linalg.norm(out.vertices, axis=1)
    assert radii.min() > 0.9 and radii.max() < 1.05
    # outward-facing after fix_normals
    assert out.volume > 0
    # the input is untouched
    assert len(ico.faces) == 5120


def test_decimate_with_target_at_or_above_face_count_returns_a_copy():
    ico = trimesh.creation.icosphere(subdivisions=1)
    out = decimate(ico, len(ico.faces), min_component_faces=1)
    assert out is not ico
    assert len(out.faces) == len(ico.faces)
    assert np.allclose(out.vertices, ico.vertices)


# --------------------------------------------------------------------------
# unwrap
# --------------------------------------------------------------------------

def test_unwrap_icosphere_gives_valid_uvs_and_source_mapping():
    ico = trimesh.creation.icosphere(subdivisions=2)
    un = unwrap(ico, resolution=256, padding=2)

    assert isinstance(un, UnwrappedMesh)
    assert un.source is ico
    n_split = len(un.mesh.vertices)
    assert n_split >= len(ico.vertices)          # seams duplicate vertices
    assert len(un.mesh.faces) == len(ico.faces)  # topology count preserved
    assert un.uvs.shape == (n_split, 2)
    assert un.uvs.min() >= 0.0 and un.uvs.max() <= 1.0
    assert un.uvs.max() > 0.5                    # the atlas is actually used
    assert un.source_index.shape == (n_split,)
    assert un.source_index.min() >= 0 and un.source_index.max() < len(ico.vertices)
    assert set(np.unique(un.source_index)) == set(range(len(ico.vertices)))
    assert np.allclose(un.mesh.vertices, ico.vertices[un.source_index])
    assert un.normals.shape == (n_split, 3)
    assert np.allclose(np.linalg.norm(un.normals, axis=1), 1.0, atol=1e-6)
    assert np.allclose(un.normals, ico.vertex_normals[un.source_index])
    assert un.mesh.faces.min() >= 0 and un.mesh.faces.max() < n_split


# --------------------------------------------------------------------------
# SurfaceColorSampler + bake_texture
# --------------------------------------------------------------------------

def test_surface_color_sampler_returns_texture_color_at_3d_points():
    quad = textured_quad()
    sampler = SurfaceColorSampler(quad, n_samples=20_000, seed=0)

    assert sampler.points.shape == (20_000, 3)
    assert sampler.colors.shape == (20_000, 3)
    left = sampler.query(np.array([[0.2, 0.5, 0.0], [0.1, 0.9, 0.0]]))
    right = sampler.query(np.array([[0.8, 0.5, 0.0], [0.9, 0.1, 0.0]]))
    assert np.allclose(left, RED, atol=0.05)
    assert np.allclose(right, BLUE, atol=0.05)
    # k=1 takes the single nearest sample
    assert np.allclose(sampler.query(np.array([[0.2, 0.5, 0.0]]), k=1), RED, atol=0.05)
    # right on the seam the bilinear filter produces a red/blue mix, never another hue
    mid = sampler.query(np.array([[0.5, 0.5, 0.0]]))[0]
    assert mid[1] < 0.05 and np.isclose(mid[0] + mid[2], 1.0, atol=0.05)


def test_surface_color_sampler_rejects_meshes_without_texture():
    plain = trimesh.creation.box()
    with pytest.raises(ValueError, match="TextureVisuals"):
        SurfaceColorSampler(plain, n_samples=100)
    no_image = textured_quad()
    no_image.visual.material.baseColorTexture = None
    with pytest.raises(ValueError, match="baseColorTexture"):
        SurfaceColorSampler(no_image, n_samples=100)


def test_bake_texture_transfers_red_and_blue_through_the_new_atlas():
    quad = textured_quad()
    sampler = SurfaceColorSampler(quad, n_samples=20_000, seed=0)
    un = unwrap(quad, resolution=64, padding=2)
    baked = bake_texture(un, sampler, size=64, padding=4)

    color = baked.color
    assert color.shape == (64, 64, 3)
    assert color.dtype == np.float32
    assert baked.position_map.mask.mean() > 0.9

    red = np.linalg.norm(color - RED, axis=-1) < 0.15
    blue = np.linalg.norm(color - BLUE, axis=-1) < 0.15
    total = color.shape[0] * color.shape[1]
    # both colors are present in bulk ...
    assert red.sum() > 0.3 * total
    assert blue.sum() > 0.3 * total
    # ... the overwhelming majority of texels are one or the other ...
    assert (red | blue).sum() > 0.8 * total
    # ... and nothing else appears: every texel is on the red-blue line
    # (the only blends are the bilinear seam and the dilated padding)
    covered_or_padded = np.linalg.norm(color, axis=-1) > 1e-6
    assert color[..., 1].max() < 0.05
    assert np.allclose(color[covered_or_padded, 0] + color[covered_or_padded, 2], 1.0, atol=0.05)
