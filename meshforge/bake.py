"""UV-space baking primitives: rasterize a mesh's triangles into its
texture atlas so every texel knows which 3D point it covers.

Once a texel has a 3D position, "painting" becomes geometry: fill every
texel inside a 3D region, project an image through a cylinder, darken a
ring around a socket — none of it needs to know how fragmented or oddly
laid out the UV atlas is. That is the whole point: TRELLIS-style atlases
(and xatlas output) are a jigsaw of small charts, so authoring anything in
2D UV space directly is hopeless; authoring in 3D and letting the
position map carry it into UV space is not.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw


class PositionMap(NamedTuple):
    face_id: np.ndarray    # (H, W) int32, -1 where no triangle covers the texel
    position: np.ndarray   # (H, W, 3) float32 world-space position per covered texel
    normal: np.ndarray     # (H, W, 3) float32 interpolated vertex normal per covered texel
    mask: np.ndarray       # (H, W) bool, True where covered

    @property
    def size(self) -> int:
        return self.face_id.shape[0]

    @property
    def width(self) -> int:
        return self.face_id.shape[1]

    @property
    def height(self) -> int:
        return self.face_id.shape[0]


def _size_wh(size: "int | tuple[int, int]") -> tuple[int, int]:
    """(width, height) from either a square size or an explicit (w, h)."""
    if isinstance(size, (tuple, list)):
        w, h = size
        return int(w), int(h)
    return int(size), int(size)


def _uv_to_pixel(uv: np.ndarray, size: "int | tuple[int, int]") -> np.ndarray:
    """glTF convention: uv (0, 0) is the top-left texel corner, v grows
    downward — which is also PIL's pixel convention, so no flip."""
    w, h = _size_wh(size)
    return uv * np.array([w, h], dtype=np.float64)


def rasterize_face_ids(uvs: np.ndarray, faces: np.ndarray, size: "int | tuple[int, int]") -> np.ndarray:
    """Return an (height, width) int32 image of face indices (-1 = empty).
    `size` is a square edge or an explicit (width, height) — a garment
    whose UV layout is a long rectangle (meshforge.robe wraps the whole
    circumference into one chart) wastes most of a square atlas.

    Drawn with PIL (C-speed) by encoding each face index as a 24-bit RGB
    color; triangles are also outlined so slivers thinner than a texel
    still claim at least their edge texels (conservative-ish coverage),
    which matters because xatlas charts are full of thin triangles and a
    missed texel is a visible pinhole in the baked texture.
    """
    if len(faces) >= 1 << 24:
        raise ValueError("rasterize_face_ids: more than 2^24 faces cannot be encoded in 24-bit ids")
    w, h = _size_wh(size)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    px = _uv_to_pixel(np.asarray(uvs, dtype=np.float64), size)
    tri = px[faces]  # (F, 3, 2)
    for fid in range(len(faces)):
        c = (fid & 0xFF, (fid >> 8) & 0xFF, (fid >> 16) & 0xFF)
        pts = [(float(x), float(y)) for x, y in tri[fid]]
        draw.polygon(pts, fill=c, outline=c)
    arr = np.asarray(img, dtype=np.int64)
    ids = arr[..., 0] | (arr[..., 1] << 8) | (arr[..., 2] << 16)
    ids[ids == 0xFFFFFF] = -1
    # faces past the real count can only come from the white background
    ids[ids >= len(faces)] = -1
    return ids.astype(np.int32)


def barycentric_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of points p w.r.t. triangles (a, b, c) in
    2D, vectorized over the leading axis; degenerate triangles get the
    nearest-vertex one-hot so they never produce NaN."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    ok = np.abs(denom) > 1e-18
    safe = np.where(ok, denom, 1.0)
    v = np.where(ok, (d11 * d20 - d01 * d21) / safe, 0.0)
    w = np.where(ok, (d00 * d21 - d01 * d20) / safe, 0.0)
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=-1)
    # texels claimed by an outline pixel can sit slightly outside their
    # triangle; clamp + renormalize instead of extrapolating wildly.
    bary = np.clip(bary, 0.0, 1.0)
    s = bary.sum(-1, keepdims=True)
    s[s == 0] = 1.0
    return bary / s


def bake_position_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    size: "int | tuple[int, int]" = 2048,
    vertex_normals: np.ndarray | None = None,
) -> PositionMap:
    """Rasterize the atlas and interpolate a world position (and normal)
    for every covered texel center. `size` is a square edge or (width,
    height); the returned maps are (height, width[, 3])."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    if uvs.shape != (len(vertices), 2):
        raise ValueError(f"bake_position_map: uvs must be ({len(vertices)}, 2), got {uvs.shape}")

    face_id = rasterize_face_ids(uvs, faces, size)
    mask = face_id >= 0
    ys, xs = np.nonzero(mask)
    fids = face_id[ys, xs]
    texel_centers = np.stack([xs + 0.5, ys + 0.5], axis=1)
    px = _uv_to_pixel(uvs, size)
    tri_uv = px[faces[fids]]  # (N, 3, 2)
    bary = barycentric_2d(texel_centers, tri_uv[:, 0], tri_uv[:, 1], tri_uv[:, 2])

    tri_pos = vertices[faces[fids]]  # (N, 3, 3)
    pos = (tri_pos * bary[..., None]).sum(axis=1)

    w, h = _size_wh(size)
    position = np.zeros((h, w, 3), dtype=np.float32)
    position[ys, xs] = pos

    normal = np.zeros((h, w, 3), dtype=np.float32)
    if vertex_normals is not None:
        tri_n = np.asarray(vertex_normals, dtype=np.float64)[faces[fids]]
        n = (tri_n * bary[..., None]).sum(axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        normal[ys, xs] = n

    return PositionMap(face_id=face_id, position=position, normal=normal, mask=mask)


def dilate_into_padding(texture: np.ndarray, mask: np.ndarray, padding: int = 8) -> np.ndarray:
    """Grow covered texel colors outward into the empty gutters so bilinear
    filtering and mipmaps at chart borders sample a neighbour's color
    instead of the background. Pure numpy 4-neighbour dilation, one ring
    per iteration."""
    tex = texture.astype(np.float32).copy()
    known = mask.copy()
    for _ in range(padding):
        if known.all():
            break
        acc = np.zeros_like(tex)
        cnt = np.zeros(known.shape, dtype=np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_known = np.roll(known, (dy, dx), axis=(0, 1))
            shifted_tex = np.roll(tex, (dy, dx), axis=(0, 1))
            # np.roll wraps around; mask out the wrapped row/column
            if dy == 1:
                shifted_known[0, :] = False
            if dy == -1:
                shifted_known[-1, :] = False
            if dx == 1:
                shifted_known[:, 0] = False
            if dx == -1:
                shifted_known[:, -1] = False
            take = shifted_known & ~known
            acc[take] += shifted_tex[take]
            cnt[take] += 1.0
        fill = (cnt > 0) & ~known
        tex[fill] = acc[fill] / cnt[fill][:, None]
        known = known | fill
    return tex


def cylindrical_coords(points: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray,
                       forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta, height, radius) of points around an axis. theta is measured
    from `forward` (projected perpendicular to the axis), positive toward
    axis × forward (for a +Y axis and +Z forward that is +X), in radians in
    (-pi, pi]."""
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    fwd = np.asarray(forward, dtype=np.float64)
    fwd = fwd - axis_dir * np.dot(fwd, axis_dir)
    fwd = fwd / np.linalg.norm(fwd)
    side = np.cross(axis_dir, fwd)
    rel = np.asarray(points, dtype=np.float64) - np.asarray(axis_point, dtype=np.float64)
    height = rel @ axis_dir
    px = rel @ fwd
    py = rel @ side
    theta = np.arctan2(py, px)
    radius = np.sqrt(px * px + py * py)
    return theta, height, radius


def render_text_image(text: str, width: int, height: int, font_paths: list[str],
                      fill=(255, 255, 255, 255), stroke_width: int = 0,
                      stroke_fill=(0, 0, 0, 0), tracking: float = 0.0,
                      fit: tuple[float, float] = (0.84, 0.80)) -> Image.Image:
    """Render `text` as large as fits into `fit` × (width, height),
    centered, RGBA on transparent. Tries each font path in order,
    falling back to PIL's bitmap font only if none load (which would be
    unreadably small — callers should pass real font files).

    `fit` is (width fraction, height fraction); whichever binds first
    sets the size. Callers that care about cap height relative to a
    surface — the collar decal does — size the canvas to that surface and
    pass the target as `fit`.

    `tracking` adds letter-spacing as a fraction of the font size; it is
    drawn glyph by glyph, so pair kerning is dropped. Zero keeps the
    single-call path (kerning intact)."""
    from PIL import ImageFont

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, size=10)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
        draw.text((width // 2, height // 2), text, fill=fill, font=font, anchor="mm")
        return img

    def ink(f):
        """(width, height, top, bottom) of the string as drawn, tracking included."""
        l, t, r, b = draw.textbbox((0, 0), text, font=f, stroke_width=stroke_width)
        w = (r - l) + tracking * f.size * max(len(text) - 1, 0)
        return w, b - t, t, b

    # binary-search the largest size that fits the box with margins
    lo, hi = 8, max(height, 16) * 2
    target_w, target_h = width * fit[0], height * fit[1]
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        w, h, _t, _b = ink(ImageFont.truetype(font.path, size=mid))
        if w <= target_w and h <= target_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    f = ImageFont.truetype(font.path, size=best)
    if tracking <= 0:
        draw.text((width / 2, height / 2), text, fill=fill, font=f, anchor="mm",
                  stroke_width=stroke_width, stroke_fill=stroke_fill)
        return img
    advances = [f.getlength(ch) for ch in text]
    extra = tracking * best
    total = sum(advances) + extra * max(len(text) - 1, 0)
    _w, _h, top, bottom = ink(f)
    x = (width - total) / 2
    y = height / 2 - (top + bottom) / 2          # textbbox is measured from anchor "la"
    for ch, adv in zip(text, advances):
        draw.text((x, y), ch, fill=fill, font=f, stroke_width=stroke_width, stroke_fill=stroke_fill)
        x += adv + extra
    return img
