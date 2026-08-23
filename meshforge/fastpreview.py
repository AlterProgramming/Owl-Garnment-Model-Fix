"""Fast flat-shaded preview renderer (PIL polygon fill, painter's order).

meshforge.preview.render_orthographic interpolates vertex colors per
pixel in a Python loop, which is exact but takes ~15 s per 512 px view of
a 150k-face mesh. Region-map and skin-weight debugging wants dozens of
views per iteration, so this renderer trades per-pixel interpolation for
one C-speed polygon fill per triangle with a flat color (mean of the
vertex colors, optionally Lambert-shaded by the face normal). ~1 s per
1024 px view. Use render_orthographic for final stills, this for loops.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    return rx @ ry


def render_flat(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    resolution: int = 1024,
    bounds: np.ndarray | None = None,
    shade: bool = True,
    background=(38, 38, 38),
    fill_fraction: float = 0.92,
    return_mapping: bool = False,
):
    """Orthographic flat-shaded render. `colors` is (n_verts, 3) or
    (n_faces, 3) in [0, 1]. Camera looks down -Z after rotating the model
    by yaw about Y then pitch about X (positive pitch = camera looks
    down from above). With `bounds` the framing is fixed across calls.
    `return_mapping` also returns (scale, center) so callers can draw
    overlays in the same pixel space."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.float64)
    R = _rotation(yaw_deg, pitch_deg)
    rotated = vertices @ R.T
    if bounds is not None:
        bmin, bmax = np.asarray(bounds[0]), np.asarray(bounds[1])
        corners = np.array([[bmin[0] if bx == 0 else bmax[0], bmin[1] if by == 0 else bmax[1],
                             bmin[2] if bz == 0 else bmax[2]] for bx in (0, 1) for by in (0, 1) for bz in (0, 1)])
        rc = corners @ R.T
        center = (rc.min(0) + rc.max(0)) / 2
        extent = (rc.max(0) - rc.min(0))
    else:
        center = (rotated.min(0) + rotated.max(0)) / 2
        extent = rotated.max(0) - rotated.min(0)
    scale = fill_fraction * (resolution - 1) / max(np.max(extent[:2]), 1e-9)
    pts = rotated - center
    px = pts[:, 0] * scale + (resolution - 1) / 2
    py = -pts[:, 1] * scale + (resolution - 1) / 2

    tri = pts[faces]
    face_z = tri[:, :, 2].mean(axis=1)
    order = np.argsort(face_z)

    if colors.shape[0] == len(faces) and colors.shape[0] != len(vertices):
        face_colors = colors
    else:
        face_colors = colors[faces].mean(axis=1)
    if shade:
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        light = np.array([0.35, 0.5, 0.8])
        light /= np.linalg.norm(light)
        lambert = np.clip(n @ light, 0, 1)
        face_colors = face_colors * (0.55 + 0.45 * lambert)[:, None]
    rgb = (np.clip(face_colors, 0, 1) * 255).astype(np.uint8)

    img = Image.new("RGB", (resolution, resolution), tuple(int(c) for c in background))
    draw = ImageDraw.Draw(img)
    fx = px[faces]
    fy = py[faces]
    for f in order:
        c = (int(rgb[f, 0]), int(rgb[f, 1]), int(rgb[f, 2]))
        draw.polygon([(fx[f, 0], fy[f, 0]), (fx[f, 1], fy[f, 1]), (fx[f, 2], fy[f, 2])], fill=c, outline=c)
    if return_mapping:
        return img, (scale, center, R)
    return img


def contact_sheet(images: list[Image.Image], cols: int | None = None, pad: int = 6,
                  background=(20, 20, 20)) -> Image.Image:
    if not images:
        raise ValueError("contact_sheet: no images")
    cols = cols or len(images)
    rows = int(np.ceil(len(images) / cols))
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), background)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet
