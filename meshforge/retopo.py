"""Geometry + color preparation for a textured, riggable base mesh.

Two source assets feed the owl (see the 2026-08-21 design spec): a
closed-but-vertex-colored mesh and a hi-res textured export whose
geometry is a patchwork but whose 2048^2 baseColor is the only real color
record. This module turns a hi-res textured TRELLIS export into a
closed, decimated, freshly-unwrapped mesh with a clean baked texture:

    load -> weld (position-only) -> drop debris -> decimate -> xatlas unwrap
         -> bake: texel -> 3D position -> nearest hi-res surface sample color

The color transfer deliberately goes through 3D surface samples rather
than UV-space resampling: the source atlas is fragmented per patch, so
there is no 2D correspondence to exploit, but the two surfaces coincide
in 3D, and a KD-tree over a few million texture-colored surface samples
gives texel-accurate color for any point on the new surface.
"""
from __future__ import annotations

from typing import NamedTuple

import fast_simplification
import numpy as np
import trimesh
import xatlas
from scipy.spatial import cKDTree

from meshforge.bake import PositionMap, bake_position_map, dilate_into_padding


class UnwrappedMesh(NamedTuple):
    mesh: trimesh.Trimesh        # xatlas-split vertices (seams duplicated), process=False
    uvs: np.ndarray              # (n_split, 2) in [0, 1], glTF convention (v grows downward)
    source_index: np.ndarray     # (n_split,) index into the pre-split decimated mesh's vertices
    normals: np.ndarray          # (n_split, 3) smooth normals computed on the pre-split topology
    source: trimesh.Trimesh      # the pre-split decimated mesh (for welded-topology operations)


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a GLB as one Trimesh, applying scene-node transforms, keeping
    TextureVisuals (uv + material) when the file has them."""
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    edges_sorted = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    return int((counts == 1).sum())


def drop_small_components(mesh: trimesh.Trimesh, min_faces: int) -> trimesh.Trimesh:
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels)
    keep = counts[labels] >= min_faces
    if not keep.any():
        raise ValueError(f"drop_small_components: every component is below {min_faces} faces")
    if keep.all():
        return mesh
    return mesh.submesh([keep], append=True)


def weld(mesh: trimesh.Trimesh, digits_vertex: int = 5, min_component_faces: int = 50,
         max_boundary_fraction: float = 0.01) -> trimesh.Trimesh:
    """Position-only weld (merge regardless of UV/normal splits — that is
    what `merge_tex=True, merge_norm=True` means in trimesh), then drop
    debris components. Returns a plain geometry mesh (no visuals): color
    is transferred later from the *unwelded* source, so nothing is lost.

    Raises if the welded surface still has more boundary edges than
    `max_boundary_fraction` of its vertices — a genuinely open mesh is a
    different problem than unwelded seams and should not be silently
    accepted (same contract as clean.weld_and_clean).
    """
    working = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    working.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits_vertex)
    working.update_faces(working.nondegenerate_faces())
    working = drop_small_components(working, min_component_faces)
    boundary = _boundary_edge_count(working)
    threshold = max(1, int(max_boundary_fraction * len(working.vertices)))
    if boundary > threshold:
        raise ValueError(
            f"weld: {boundary} boundary edges remain after welding (threshold {threshold}) — "
            "this looks like a genuinely open surface, not unwelded seams."
        )
    return working


def decimate(mesh: trimesh.Trimesh, target_faces: int, min_component_faces: int = 30) -> trimesh.Trimesh:
    """Quadric decimation to about `target_faces` triangles (positions only)."""
    if target_faces >= len(mesh.faces):
        out = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    else:
        reduction = 1.0 - target_faces / len(mesh.faces)
        pts, faces = fast_simplification.simplify(
            mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64),
            target_reduction=float(reduction), agg=7,
        )
        out = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
        out.update_faces(out.nondegenerate_faces())
        out.remove_unreferenced_vertices()
    out = drop_small_components(out, min_component_faces)
    out.fix_normals()
    return out


def unwrap(mesh: trimesh.Trimesh, resolution: int = 2048, padding: int = 4) -> UnwrappedMesh:
    """xatlas chart + pack. Returns the split mesh plus the mapping back to
    the pre-split vertices (needed so smooth normals, skin weights and
    region labels can all be authored once on the welded topology and
    simply gathered onto the split vertices)."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    chart = xatlas.ChartOptions()
    pack = xatlas.PackOptions()
    pack.resolution = int(resolution)
    pack.padding = int(padding)
    pack.bilinear = True
    atlas.generate(chart_options=chart, pack_options=pack)
    vmapping, indices, uvs = atlas[0]
    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    if uvs.max() > 1.5:
        # some binding versions return texel units instead of normalized uv
        uvs = uvs / float(max(atlas.width, atlas.height))
    uvs = np.clip(uvs, 0.0, 1.0)

    split_vertices = mesh.vertices[vmapping]
    split = trimesh.Trimesh(vertices=split_vertices, faces=indices, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)[vmapping]
    return UnwrappedMesh(mesh=split, uvs=uvs, source_index=vmapping, normals=normals, source=mesh)


def _bilinear_sample(image: np.ndarray, uv: np.ndarray, flip_v: bool) -> np.ndarray:
    h, w = image.shape[:2]
    u = np.clip(uv[:, 0] * (w - 1), 0, w - 1)
    v = uv[:, 1]
    if flip_v:
        v = 1.0 - v
    v = np.clip(v * (h - 1), 0, h - 1)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    c00, c10 = image[y0, x0], image[y0, x1]
    c01, c11 = image[y1, x0], image[y1, x1]
    return c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy


class SurfaceColorSampler:
    """Texture-colored surface samples of a textured mesh, queryable by 3D
    position. trimesh flips V when it loads a GLB texture (OpenGL
    convention, v=0 at the bottom), hence `flip_v=True` for meshes that
    came through trimesh.load."""

    def __init__(self, source: trimesh.Trimesh, n_samples: int = 3_000_000, seed: int = 0,
                 flip_v: bool = True):
        if source.visual.kind != "texture" or source.visual.uv is None:
            raise ValueError("SurfaceColorSampler: source mesh needs TextureVisuals with uv")
        texture = source.visual.material.baseColorTexture
        if texture is None:
            raise ValueError("SurfaceColorSampler: source material has no baseColorTexture")
        image = np.asarray(texture.convert("RGB"), dtype=np.float32) / 255.0

        points, face_ids = trimesh.sample.sample_surface(source, n_samples, seed=seed)
        bary = trimesh.triangles.points_to_barycentric(source.triangles[face_ids], points)
        uv = (source.visual.uv[source.faces[face_ids]] * bary[..., None]).sum(axis=1)
        self.colors = _bilinear_sample(image, uv, flip_v=flip_v).astype(np.float32)
        self.points = np.asarray(points, dtype=np.float64)
        self.tree = cKDTree(self.points)

    def query(self, points: np.ndarray, k: int = 3) -> np.ndarray:
        """Inverse-distance-weighted color of the k nearest samples."""
        points = np.asarray(points, dtype=np.float64)
        dist, idx = self.tree.query(points, k=k, workers=-1)
        if k == 1:
            return self.colors[idx]
        w = 1.0 / np.maximum(dist, 1e-9)
        w /= w.sum(axis=1, keepdims=True)
        return (self.colors[idx] * w[..., None].astype(np.float32)).sum(axis=1)


class BakedTexture(NamedTuple):
    color: np.ndarray          # (size, size, 3) float32 in [0, 1], sRGB
    position_map: PositionMap


def bake_texture(unwrapped: UnwrappedMesh, sampler: SurfaceColorSampler, size: "int | tuple[int, int]" = 2048,
                 padding: int = 8, k: int = 3) -> BakedTexture:
    posmap = bake_position_map(
        unwrapped.mesh.vertices, unwrapped.mesh.faces, unwrapped.uvs, size=size,
        vertex_normals=unwrapped.normals,
    )
    color = np.zeros((posmap.height, posmap.width, 3), dtype=np.float32)
    color[posmap.mask] = sampler.query(posmap.position[posmap.mask], k=k)
    color = dilate_into_padding(color, posmap.mask, padding=padding)
    return BakedTexture(color=color, position_map=posmap)
