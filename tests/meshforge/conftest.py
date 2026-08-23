"""Shared fixtures for meshforge tests."""
import numpy as np
import trimesh
import pytest


def two_triangles_with_duplicate_seam():
    """Multiple triangles sharing edges, but the shared vertices are
    stored as separate (duplicate-position) indices — the exact defect
    found in the raw TRELLIS GLB this session (near-identical but distinct
    vertex indices at a seam)."""
    vertices = np.array([
        [0.0, 0.0, 0.0],   # 0: shared vertex A
        [1.0, 0.0, 0.0],   # 1: seam
        [0.0, 1.0, 0.0],   # 2: seam
        [1.0 + 1e-7, 0.0, 0.0],   # 3: seam dup of 1
        [0.0 + 1e-7, 1.0, 0.0],   # 4: seam dup of 2
        [1.0, 1.0, 0.0],   # 5: shared vertex B
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 5],
        [0, 5, 3],
        [1, 4, 5],
    ])
    colors = np.array([
        [200, 30, 30, 255],
        [200, 30, 30, 255],
        [200, 30, 30, 255],
        [205, 32, 28, 255],
        [198, 29, 33, 255],
        [200, 30, 30, 255],
    ], dtype=np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


def mesh_with_debris():
    """A box (12 faces) plus a single-triangle debris shard far away,
    matching the session's finding that debris components are small and
    disconnected from the main surface."""
    main = trimesh.creation.box(extents=[1, 1, 0.1])
    debris_verts = np.array([
        [10.0, 10.0, 0.0],
        [10.1, 10.0, 0.0],
        [10.0, 10.1, 0.0],
    ])
    debris_faces = np.array([[0, 1, 2]])
    debris = trimesh.Trimesh(vertices=debris_verts, faces=debris_faces, process=False)
    combined = trimesh.util.concatenate([main, debris])
    return combined


@pytest.fixture
def duplicate_seam_mesh():
    return two_triangles_with_duplicate_seam()


@pytest.fixture
def debris_mesh():
    return mesh_with_debris()
