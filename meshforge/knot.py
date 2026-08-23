"""Procedural compact cloth knot for the raised wing's gathered entry.

The knot is a bunch with a cinch: a softly fluted icosphere is gathered at
its tangent-wise waist, a single flat ribbon closes around that waist, and
two short ribbons turn down and toward the front.  Only three small folds
connect the supplied sheet arc to the underside of the bunch; the knot does
not grow a fan of independent strands.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh

from meshforge.rigexport import MaterialSpec, PrimitiveSpec


class Knot(NamedTuple):
    primitive: PrimitiveSpec
    base_points: np.ndarray      # (n_strands, 3) sampled from the supplied arc
    info: dict


def _bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _tube(path: np.ndarray, radius: np.ndarray, segments: int, along: tuple[float, float],
          around: tuple[float, float], flatten: float = 1.0):
    """Sweep a circle (or flattened ellipse) along ``path``.

    ``along`` and ``around`` remain chart coordinates rather than geometric
    coordinates: the caller can use the same woven region of the kente chart
    for every piece.  A closed path is represented by repeating its first
    point as the final station, which keeps this helper's open-path interface
    unchanged while allowing a watertight band in ``build_knot``.
    """
    path = np.asarray(path, dtype=np.float64)
    k = len(path)
    T = np.gradient(path, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    N = np.zeros_like(T)
    a = np.array([0.0, 1.0, 0.0]) if abs(T[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N[0] = np.cross(T[0], a)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, k):
        n = N[i - 1] - T[i] * np.dot(T[i], N[i - 1])
        N[i] = n / max(np.linalg.norm(n), 1e-12)
    B = np.cross(T, N)
    ph = np.arange(segments) * (2 * np.pi / segments)
    ring = np.cos(ph)[:, None, None] * N[None] + flatten * np.sin(ph)[:, None, None] * B[None]
    verts = (path[None] + np.asarray(radius)[None, :, None] * ring).transpose(1, 0, 2).reshape(-1, 3)
    i = np.arange(k - 1)[:, None]
    j = np.arange(segments)[None, :]
    a_ = i * segments + j
    b_ = i * segments + (j + 1) % segments
    c_ = a_ + segments
    d_ = b_ + segments
    faces = np.concatenate([np.stack([a_, b_, d_], -1).reshape(-1, 3), np.stack([a_, d_, c_], -1).reshape(-1, 3)])
    v = along[0] + (along[1] - along[0]) * np.repeat(np.linspace(0.0, 1.0, k), segments)
    u = around[0] + (around[1] - around[0]) * np.tile(np.arange(segments) / segments, k)
    return verts, faces, np.stack([u, v], axis=1)


def _part_from_icosphere(size: float):
    """Return the cinched bunch in local (tangent, up, normal) coordinates."""
    bunch = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    # The requested tuple describes the full bunch dimensions.  The source
    # sphere has radius 1, so its scale is half that tuple.
    # Keep the bunch dominant in the compact silhouette.  The nominal cloth
    # dimensions remain the reference; this mild allowance accounts for the
    # radius-1 sphere's projected silhouette and the waist band around it.
    half_axes = np.array([0.65, 0.55, 0.275], dtype=np.float64) * size
    local = np.asarray(bunch.vertices, dtype=np.float64) * half_axes

    t = local[:, 0]
    pinch = 1.0 - 0.35 * np.exp(-(t / max(0.28 * size, 1e-12)) ** 2)
    phi = np.arctan2(local[:, 2], local[:, 1])
    flutes = 1.0 + 0.06 * np.cos(5.0 * phi)
    local[:, 1] *= pinch * flutes
    local[:, 2] *= pinch * flutes

    uv_u = 0.20 + 0.30 * (phi / (2.0 * np.pi) + 0.5)
    uv_v = 0.20 + 0.30 * (local[:, 0] / max(half_axes[0], 1e-12) + 1.0) * 0.5
    uvs = np.stack([uv_u, uv_v], axis=1)
    return local, np.asarray(bunch.faces, dtype=np.int64), np.clip(uvs, 0.0, 1.0)


def build_knot(frame: dict, arc_points: np.ndarray, size: float, material: MaterialSpec,
               n_strands: int = 7, stations: int = 16, segments: int = 6, seed: int = 0) -> Knot:
    del seed  # The construction is deterministic; retain the public interface.
    o, nrm, tan, up = (np.asarray(frame[k], dtype=np.float64) for k in ("origin", "normal", "tangent", "up"))
    arc = np.asarray(arc_points, dtype=np.float64)
    arc_len = float(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum())
    idx = np.linspace(0.18 * (len(arc) - 1), 0.82 * (len(arc) - 1), n_strands).round().astype(int)
    starts = arc[idx]

    # Work in the knot frame for every construction step.  The world-frame
    # transform is applied exactly once after all pieces have been assembled.
    R = np.stack([tan, up, nrm], axis=1)
    starts_local = (starts - o) @ R
    parts = []

    # --- the bunch: one softly fluted, tangent-pinched surface
    bunch_v, bunch_f, bunch_uv = _part_from_icosphere(size)
    parts.append((bunch_v, bunch_f, bunch_uv))

    # --- one flat ribbon around the pinched waist
    # There are 28 intervals and a repeated first station so the band closes.
    band_stations = 28
    theta = np.linspace(0.0, 2.0 * np.pi, band_stations + 1)
    band_path = np.stack([
        np.zeros_like(theta),
        0.42 * size * np.cos(theta),
        0.26 * size * np.sin(theta),
    ], axis=1)
    parts.append(_tube(band_path, np.full(len(theta), 0.19 * size), segments,
                       along=(0.50, 0.86), around=(0.30, 0.50), flatten=0.35))

    # --- two short ends: down first, then turn toward +normal
    end_stations = max(8, int(stations // 2) + 2)
    for side in (-1.0, 1.0):
        p0 = np.array([side * 0.27 * size, -0.36 * size, 0.00 * size])
        p1 = p0 + np.array([0.00, -0.15 * size, 0.00])
        p3 = p0 + np.array([side * -0.05 * size, -0.24 * size, 0.22 * size])
        p2 = p3 - np.array([0.00, 0.00, 0.14 * size])
        path = _bezier(p0, p1, p2, p3, end_stations)
        half_width = np.linspace(0.18 * size, 0.24 * size, end_stations)
        parts.append(_tube(path, half_width, segments, along=(0.56, 0.98),
                           around=(0.36 if side < 0 else 0.42, 0.42 if side < 0 else 0.48),
                           flatten=0.30))

    # --- three short entry folds tucked under the bunch's lower edge
    # Use only the three central sampled points.  The remaining base_points
    # are retained as the caller-facing samples but do not become legs.
    centre = n_strands // 2
    middle = np.array([max(0, centre - 1), centre, min(n_strands - 1, centre + 1)])
    wedge_stations = max(5, int(stations // 3))
    for j, i in enumerate(middle):
        p0 = starts_local[i]
        p3 = np.array([0.75 * p0[0], -0.24 * size, 0.04 * size * (j - 1)], dtype=np.float64)
        delta = p3 - p0
        p1 = p0 + 0.32 * delta + np.array([0.0, -0.05 * size, 0.01 * size])
        p2 = p0 + 0.82 * delta + np.array([0.0, 0.02 * size, -0.01 * size])
        path = _bezier(p0, p1, p2, p3, wedge_stations)
        radius = np.linspace(0.11 * size, 0.075 * size, wedge_stations)
        parts.append(_tube(path, radius, segments, along=(0.12, 0.40),
                           around=(0.22 + 0.08 * j, 0.28 + 0.08 * j), flatten=0.30))

    local_verts = np.concatenate([p[0] for p in parts])
    offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    faces = np.concatenate([p[1] + offset for p, offset in zip(parts, offsets)])
    uvs = np.clip(np.concatenate([p[2] for p in parts]), 0.0, 1.0)

    mesh = trimesh.Trimesh(vertices=local_verts, faces=faces, process=False)
    local_normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    bad = np.linalg.norm(local_normals, axis=1) < 0.5
    local_normals[bad] = np.array([0.0, 0.0, 1.0])
    local_normals /= np.linalg.norm(local_normals, axis=1, keepdims=True)

    verts = o + local_verts @ R.T
    normals = local_normals @ R.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    prim = PrimitiveSpec(name="kente_knot", vertices=verts, faces=faces,
                         normals=normals, uvs=uvs, material=material)
    info = {"faces": int(len(faces)), "vertices": int(len(verts)), "strands": n_strands,
            "size": round(float(size), 4), "arc_length": round(arc_len, 4),
            "construction": "cinched_bunch_band_two_ends_three_folds"}
    return Knot(primitive=prim, base_points=starts, info=info)
