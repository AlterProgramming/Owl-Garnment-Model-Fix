"""Weld coincident vertices, drop debris components, denoise vertex color.

Packages the interactive fix discovered against a raw TRELLIS export this
session: the raw mesh is riddled with near-identical duplicate vertices at
every patch seam (never merged by the exporter), which reads as thousands
of disconnected "debris" components and per-patch texture-bake noise.
Welding first is the fix; everything else here is cleanup on top of that.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    edges_sorted = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _face_adjacency_components(mesh: trimesh.Trimesh) -> tuple[int, np.ndarray]:
    adj = mesh.face_adjacency
    n_faces = len(mesh.faces)
    graph = sp.coo_matrix(
        (np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n_faces, n_faces)
    )
    return connected_components(graph, directed=False)


def _sample_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return (N, 3) float colors in [0, 1] for every vertex, whether the
    mesh currently has a texture (sampled via UV) or vertex colors already."""
    if mesh.visual.kind == "texture" and mesh.visual.uv is not None:
        img = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        h, w = img.shape[:2]
        uv = mesh.visual.uv
        u = np.clip(uv[:, 0] * (w - 1), 0, w - 1)
        v = np.clip((1.0 - uv[:, 1]) * (h - 1), 0, h - 1)
        x0 = np.floor(u).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.floor(v).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = (u - x0)[:, None]
        fy = (v - y0)[:, None]
        c00, c10 = img[y0, x0], img[y0, x1]
        c01, c11 = img[y1, x0], img[y1, x1]
        return c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy
    return mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0


def _denoise_colors(mesh: trimesh.Trimesh, colors: np.ndarray) -> np.ndarray:
    n = len(mesh.vertices)
    edges = mesh.edges_unique
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1

    indptr, indices = adjacency.indptr, adjacency.indices
    max_neighbors = int(np.diff(indptr).max()) if len(indptr) > 1 else 1
    neighbor_idx = np.tile(np.arange(n)[:, None], (1, max_neighbors))
    for k in range(max_neighbors):
        deg_k = np.diff(indptr)
        has_k = deg_k > k
        pos = np.clip(indptr[:-1] + k, 0, max(len(indices) - 1, 0))
        vals = indices[pos] if len(indices) else np.zeros(len(pos), dtype=int)
        neighbor_idx[has_k, k] = vals[has_k]
    ring = np.concatenate([np.arange(n)[:, None], neighbor_idx], axis=1)

    result = colors.copy()
    for _ in range(2):
        result = np.median(result[ring], axis=1)
    for _ in range(4):
        result = (adjacency @ result) / degree[:, None]
    return result


def weld_and_clean(
    mesh: trimesh.Trimesh,
    digits_vertex: int = 5,
    min_component_faces: int = 50,
) -> trimesh.Trimesh:
    """Weld coincident vertices, drop small disconnected components, and
    denoise vertex color sampled from any existing texture.

    Raises ValueError if welding does not collapse boundary edges to near
    zero relative to vertex count — that mismatch means the mesh has a
    genuine hole/gap, not the unwelded-duplicate defect this function
    fixes, and silently "cleaning" it would hide a real problem. Also raises
    ValueError if debris filtering removes all components.
    """
    colors_before_weld = _sample_vertex_colors(mesh)
    working = mesh.copy()
    working.visual = trimesh.visual.ColorVisuals(
        mesh=working,
        vertex_colors=np.concatenate(
            [colors_before_weld, np.ones((len(colors_before_weld), 1))], axis=1
        ),
    )
    working.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits_vertex)

    # Filter debris components first (before boundary check), so harmless debris
    # doesn't trip the "genuine hole" check. The boundary check on the filtered
    # mesh then catches real holes in the surviving surface.
    _, labels = _face_adjacency_components(working)
    counts = np.bincount(labels)
    keep = np.where(counts >= min_component_faces)[0]

    if len(keep) == 0:
        raise ValueError(
            f"weld_and_clean: debris filtering with min_component_faces={min_component_faces} "
            f"removed all {len(counts)} component(s) — the entire mesh is below the "
            "component size threshold and would result in an empty mesh."
        )

    keep_mask = np.isin(labels, keep)
    cleaned = working.submesh([keep_mask], append=True)

    # Now check boundary edges on the filtered/retained mesh using the original threshold.
    # A real hole in the surviving surface will still be caught; harmless debris no longer will.
    boundary_after = _boundary_edge_count(cleaned)
    threshold = max(1, int(0.01 * len(cleaned.vertices)))
    if boundary_after > threshold:
        raise ValueError(
            f"weld_and_clean: {boundary_after} boundary edges remain after "
            f"welding and filtering (threshold {threshold}) — this looks like a genuine "
            "hole or gap, not an unwelded-duplicate defect; refusing to "
            "silently ship a mesh with real missing geometry."
        )

    raw_colors = cleaned.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    denoised = _denoise_colors(cleaned, raw_colors)
    rgba = np.concatenate([denoised, np.ones((len(denoised), 1))], axis=1)
    cleaned.visual = trimesh.visual.ColorVisuals(mesh=cleaned, vertex_colors=rgba)
    cleaned.fix_normals()
    return cleaned
