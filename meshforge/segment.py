"""Isolate a named articulated part (e.g. a raised wing) from a cleaned
mesh, and locate the pivot point + rotation axis where it attaches to the
rest of the body.
"""
from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np
import trimesh

from meshforge.clean import _face_adjacency_components


class BBoxRegion(NamedTuple):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


class PartResult(NamedTuple):
    part: trimesh.Trimesh
    body: trimesh.Trimesh
    pivot_point: np.ndarray
    pivot_axis: np.ndarray
    part_face_mask: np.ndarray


def _region_to_absolute(mesh: trimesh.Trimesh, region: BBoxRegion) -> np.ndarray:
    bmin, bmax = mesh.bounds
    size = bmax - bmin
    lo = bmin + size * np.array([region.x[0], region.y[0], region.z[0]])
    hi = bmin + size * np.array([region.x[1], region.y[1], region.z[1]])
    return np.array([lo, hi])


def _ordered_boundary_loop(mesh: trimesh.Trimesh) -> list[int] | None:
    """Return a single closed boundary loop as an ordered list of vertex
    indices, or None if the boundary isn't exactly one simple loop (every
    boundary vertex has degree 2 in the boundary-edge graph)."""
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return None

    adjacency = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None

    start = int(boundary_edges[0][0])
    loop = [start]
    prev, current, visited = None, start, {start}
    while True:
        candidates = [n for n in adjacency[current] if n != prev]
        next_vertex = None
        for candidate in candidates:
            if candidate == start and len(loop) > 2:
                next_vertex = candidate
                break
            if candidate not in visited:
                next_vertex = candidate
                break
        if next_vertex is None:
            return None
        if next_vertex == start:
            break
        loop.append(next_vertex)
        visited.add(next_vertex)
        prev, current = current, next_vertex

    if len(visited) != len(adjacency):
        return None
    return loop


def _fill_body_hole(body: trimesh.Trimesh) -> trimesh.Trimesh:
    """Cropping a part out of a continuous mesh leaves an open hole where
    the cut happened — with nothing behind it, so an animated part moving
    away from the body reveals a visible void ("body looks ripped apart",
    reported directly against this pipeline's own output). Since Stage B
    has no interior/second layer to reveal, patch the hole with a simple
    fan triangulation from its centroid, colored by averaging the loop's
    own vertex colors, rather than leaving it open.

    A no-op (returns the mesh unchanged) if the boundary isn't a single
    simple loop — a multi-hole or non-manifold boundary is a different,
    rarer failure mode this targeted fix doesn't attempt to solve.
    """
    loop = _ordered_boundary_loop(body)
    if loop is None:
        return body

    if body.visual.kind == "vertex":
        colors = body.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    else:
        colors = np.tile(np.array([0.5, 0.5, 0.5]), (len(body.vertices), 1))

    loop_points = body.vertices[loop]
    loop_colors = colors[loop]
    centroid = loop_points.mean(axis=0)
    centroid_color = loop_colors.mean(axis=0)

    center_index = len(body.vertices)
    new_vertices = np.vstack([body.vertices, centroid[None, :]])
    new_colors = np.vstack([colors, centroid_color[None, :]])

    fan_faces = np.array([
        [loop[i], loop[(i + 1) % len(loop)], center_index]
        for i in range(len(loop))
    ])
    all_faces = np.vstack([body.faces, fan_faces])

    patched = trimesh.Trimesh(vertices=new_vertices, faces=all_faces, process=False)
    rgba = np.concatenate([new_colors, np.ones((len(new_colors), 1))], axis=1)
    patched.visual = trimesh.visual.ColorVisuals(mesh=patched, vertex_colors=rgba)
    patched.fix_normals()
    return patched


def find_articulated_part(
    mesh: trimesh.Trimesh,
    region: BBoxRegion,
    min_part_faces: int = 20,
) -> PartResult:
    """Isolate the part by cropping to `region` first, THEN taking the
    largest connected component within that crop — not by looking for a
    component that is already disconnected in the whole mesh.

    This matters: a part can be either a pre-existing separate shell (e.g.
    a cap) or fully fused into a continuous surface (e.g. a wing welded
    into the same connected surface as the body after cleanup). Cropping
    by region and re-deriving connectivity only within the crop handles
    both cases identically, and for the fused case it produces exactly
    the cut a rigid hinge needs — the crop boundary becomes the seam.
    """
    bounds = _region_to_absolute(mesh, region)
    lo, hi = bounds[0], bounds[1]

    centroids = mesh.triangles_center
    inside = np.all((centroids >= lo) & (centroids <= hi), axis=1)
    if inside.sum() < min_part_faces:
        raise ValueError(
            f"find_articulated_part: no component with >= {min_part_faces} "
            f"faces found inside region {region} (found {int(inside.sum())} faces)."
        )

    cropped_face_indices = np.where(inside)[0]
    cropped = mesh.submesh([inside], append=True)
    _, labels = _face_adjacency_components(cropped)
    counts = np.bincount(labels)
    best_label = int(np.argmax(counts))
    if counts[best_label] < min_part_faces:
        raise ValueError(
            f"find_articulated_part: largest component in region has only "
            f"{int(counts[best_label])} faces; need >= {min_part_faces}."
        )

    keep_in_cropped = labels == best_label
    part_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    part_face_mask[cropped_face_indices[keep_in_cropped]] = True
    body_face_mask = ~part_face_mask

    part = mesh.submesh([part_face_mask], append=True)
    body = mesh.submesh([body_face_mask], append=True)
    body = _fill_body_hole(body)

    # pivot: the part's boundary vertices closest to the body surface
    edges_sorted = np.sort(part.edges, axis=1)
    unique_edges, edge_counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[edge_counts == 1]
    if len(boundary_edges) == 0:
        # fall back to the part's closest vertex to the body if it has no
        # open boundary of its own (e.g. it's a closed shell like the cap)
        boundary_verts = np.arange(len(part.vertices))
    else:
        boundary_verts = np.unique(boundary_edges)

    boundary_pts = part.vertices[boundary_verts]
    closest, distance, _ = trimesh.proximity.closest_point(body, boundary_pts)
    nearest_idx = np.argmin(distance)
    pivot_point = boundary_pts[nearest_idx]

    avg_edge_length = float(np.mean(part.edges_unique_length))
    if distance[nearest_idx] > 2.0 * avg_edge_length:
        raise ValueError(
            f"find_articulated_part: closest part-to-body distance "
            f"({distance[nearest_idx]:.5f}) exceeds 2x the part's average "
            f"edge length ({avg_edge_length:.5f}) — this part does not "
            "look attached to the body; refusing to guess a pivot."
        )

    # rotation axis via PCA on the boundary ring's plane normal
    centered = boundary_pts - boundary_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pivot_axis = vt[-1]
    pivot_axis = pivot_axis / np.linalg.norm(pivot_axis)

    return PartResult(
        part=part, body=body, pivot_point=pivot_point, pivot_axis=pivot_axis,
        part_face_mask=part_face_mask,
    )
