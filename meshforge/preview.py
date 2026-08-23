"""Software painter's-algorithm rasterizer for quick sanity-check renders,
promoted from the ad hoc version rewritten three times over the course of
this session's interactive debugging into one shared, tested function.
No GPU/browser required — this is what makes a broken rig visible in a
PNG before it ever reaches a real Three.js viewer.
"""
from __future__ import annotations

import numpy as np


def render_orthographic(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    rotation_deg_y: float = 0.0,
    resolution: int = 512,
    background: tuple[float, float, float] = (0.15, 0.15, 0.15),
    bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Render `vertices`/`faces`/`colors` with a painter's-algorithm
    rasterizer.

    By default the mesh is re-centered and re-scaled to fit the frame on
    every call, which is fine for a single still but makes frame-to-frame
    comparisons misleading (the whole model shifts/rescales as the pose's
    bounding box changes). Pass `bounds` — a `(2, 3)` array of
    `[bmin, bmax]` in the same (unrotated) space as `vertices` — to use a
    fixed centering/scale across a batch of renders instead of
    recomputing it per call. `bounds` is rotated by `rotation_deg_y` the
    same way `vertices` is, so a shared box stays correct across
    different viewing angles.
    """
    canvas = np.tile(np.array(background, dtype=np.float32), (resolution, resolution, 1))

    if len(faces) == 0 or len(vertices) == 0:
        return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)

    theta = np.radians(rotation_deg_y)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
    rotated = vertices @ rotation.T

    if bounds is not None:
        bmin, bmax = np.asarray(bounds[0]), np.asarray(bounds[1])
        corners = np.array([
            [bmin[0] if bx == 0 else bmax[0],
             bmin[1] if by == 0 else bmax[1],
             bmin[2] if bz == 0 else bmax[2]]
            for bx in (0, 1) for by in (0, 1) for bz in (0, 1)
        ])
        rotated_corners = corners @ rotation.T
        center = (rotated_corners.min(0) + rotated_corners.max(0)) / 2
        extent = rotated_corners.max(0) - rotated_corners.min(0)
    else:
        center = (rotated.min(0) + rotated.max(0)) / 2
        extent = rotated.max(0) - rotated.min(0)
    rotated = rotated - center
    scale = 0.9 / max(np.max(extent), 1e-9)
    rotated = rotated * scale

    px = (rotated[:, 0] * 0.9 + 0.5) * (resolution - 1)
    py = ((-rotated[:, 1]) * 0.9 + 0.5) * (resolution - 1)

    face_z = rotated[:, 2][faces].mean(axis=1)
    order = np.argsort(face_z)

    tri_px = np.stack([px[faces], py[faces]], axis=2)
    tri_colors = colors[faces]

    for face_idx in order:
        (x0, y0), (x1, y1), (x2, y2) = tri_px[face_idx]
        min_x = max(int(np.floor(min(x0, x1, x2))), 0)
        max_x = min(int(np.ceil(max(x0, x1, x2))), resolution - 1)
        min_y = max(int(np.floor(min(y0, y1, y2))), 0)
        max_y = min(int(np.ceil(max(y0, y1, y2))), resolution - 1)
        if min_x > max_x or min_y > max_y:
            continue

        grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        grid_x = grid_x.ravel() + 0.5
        grid_y = grid_y.ravel() + 0.5

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if denom == 0:
            continue
        w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
        w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
        w2 = 1 - w0 - w1

        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        w0i, w1i, w2i = w0[inside], w1[inside], w2[inside]
        c0, c1, c2 = tri_colors[face_idx]
        pixel_color = w0i[:, None] * c0 + w1i[:, None] * c1 + w2i[:, None] * c2

        cols = grid_x[inside].astype(np.int32)
        rows = grid_y[inside].astype(np.int32)
        canvas[rows, cols] = pixel_color

    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
