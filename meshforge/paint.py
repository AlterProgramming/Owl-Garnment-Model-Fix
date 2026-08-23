"""3D-aware paint operations on a baked texture (see meshforge.bake for
why painting goes through texel world positions).

* `paint_collar_text` — flat-fills the collar interior and projects a
  text image cylindrically around the neck axis so the lettering follows
  the band's measured tilt; replaces the hallucinated "ALCOPE".
* `find_eye_discs` / `blank_eye_discs` — measure the painted eyes (dark
  iris cluster + surrounding white disc) and erase them to the local face
  colour with a soft socket shadow, so authored eyeball geometry can sit
  exactly where the painting was.
* `grade` — small global saturation/contrast lift.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from meshforge.bake import PositionMap, render_text_image
from meshforge.regions import RegionMap, collar_bounds_at

DEFAULT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _hsv_sat_val(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return sat, mx


def texel_labels(posmap: PositionMap, mesh_vertices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nearest-vertex region label for every covered texel ('' elsewhere)."""
    out = np.full(posmap.mask.shape, "", dtype=object)
    tree = cKDTree(np.asarray(mesh_vertices, dtype=np.float64))
    _, idx = tree.query(posmap.position[posmap.mask].astype(np.float64), k=1, workers=-1)
    out[posmap.mask] = labels[idx]
    return out


def paint_collar_text(
    color: np.ndarray,
    posmap: PositionMap,
    region_map: RegionMap,
    mesh_bounds: np.ndarray,
    mesh_vertices: np.ndarray,
    text: str | None = "AI-CCORE",
    theta_span_deg: float = 62.0,
    rim: float = 0.010,
    v_pad: float = 0.16,
    font_paths: list[str] | None = None,
    text_color=(255, 255, 255),
    fill_color: np.ndarray | None = None,
    texel_label_map: np.ndarray | None = None,
    frame=None,
    frame_inset: float = 0.02,
    radius_max_frac: float = 1.22,
    piping_frac: float = 0.0,
    frame_labels: tuple[str, ...] = ("neck", "chest", "head"),
    wing_labels: tuple[str, ...] = ("wing_left", "wing_right"),
) -> tuple[np.ndarray, dict]:
    """Returns (new_color, info). Works in bbox-fraction space like
    meshforge.regions so the collar description can be reused directly."""
    color = color.astype(np.float32).copy()
    bmin, bmax = np.asarray(mesh_bounds[0]), np.asarray(mesh_bounds[1])
    size = bmax - bmin
    collar = region_map.collar
    cx, cz = collar["centre_xz"]
    if texel_label_map is None:
        texel_label_map = texel_labels(posmap, mesh_vertices, region_map.labels)

    ys, xs = np.nonzero(posmap.mask)
    pos = posmap.position[ys, xs].astype(np.float64)
    f = (pos - bmin) / size
    theta = np.arctan2(f[:, 0] - cx, f[:, 2] - cz)
    lab = texel_label_map[ys, xs]
    is_neck = lab == "neck"
    front = np.abs(theta) < np.radians(theta_span_deg + 18)
    cur = color[ys, xs]
    sat, val = _hsv_sat_val(cur)
    if frame is None:
        lo, hi = collar_bounds_at(collar, theta)
        band = hi - lo
        interior = is_neck & (f[:, 1] > lo + rim) & (f[:, 1] < hi - rim)
    else:
        # Fill by the fitted rims, not by the "neck" label. The label is a
        # horizontal slab cut cleaned up by island removal, and on this asset
        # its boundary is visibly ragged: red spikes into the white chest and
        # a white notch punched through the middle of the band. Between the
        # fitted rims the band is a clean crescent.
        lo, hi = frame.bounds(theta)
        band = hi - lo
        # The folded right wing owns a strip of the band, and its texels kept
        # their baked colour: that showed as dark speckles on the red next to
        # the final E. Anything close enough to the collar axis is band, so
        # take it whatever the label says — the tablet is the one dark thing
        # that must stay out, and it is held well forward of that radius.
        radius_f = np.sqrt((f[:, 0] - cx) ** 2 + (f[:, 2] - cz) ** 2)
        already_red = (sat > 0.40) & (val > 0.20) & (cur[:, 0] >= cur[:, 1])
        paintable = (np.isin(lab, list(frame_labels))
                     | (radius_f < frame.radius * radius_max_frac)
                     | (np.isin(lab, list(wing_labels)) & already_red))
        interior = paintable & (f[:, 1] > lo + frame_inset * band) & (f[:, 1] < hi - frame_inset * band)
    fill_sel = interior & front

    if fill_color is None:
        reds = fill_sel & (sat > 0.45) & (val > 0.3)
        base = np.median(cur[reds], axis=0) if reds.sum() > 50 else np.array([0.80, 0.12, 0.10], dtype=np.float32)
    else:
        base = np.asarray(fill_color, dtype=np.float32)

    # flat fill with a gentle vertical shading (slightly darker toward the lower rim)
    v = np.clip((f[:, 1] - lo) / np.maximum(band, 1e-6), 0, 1)
    shade = (0.90 + 0.12 * v)[:, None]
    # feather the fill at the angular edges so it blends into the untouched collar sides
    edge = np.clip((np.radians(theta_span_deg + 18) - np.abs(theta)) / np.radians(8), 0, 1)
    new = base[None, :] * shade
    alpha = (fill_sel * edge)[:, None]
    cur = cur * (1 - alpha) + new * alpha

    info = {"fill_color": base.tolist(), "filled_texels": int(fill_sel.sum()), "text_texels": 0}
    if frame is not None and piping_frac > 0:
        # Re-lay the dark piping as a clean ring just outside each fitted rim.
        # The crease behind the chest bulge bakes as a torn, streaky smear
        # because the colour sampler barely sees it; a measured flat ring
        # reads as a scarf edge instead of as damage.
        ring = paintable & front & (
            ((f[:, 1] > lo - piping_frac * band) & (f[:, 1] <= lo + frame_inset * band))
            | ((f[:, 1] >= hi - frame_inset * band) & (f[:, 1] < hi + piping_frac * band)))
        darks = ring & (val < 0.34)
        piping = np.median(cur[darks], axis=0) if darks.sum() > 40 else base * 0.28
        cur[ring] = cur[ring] * 0.15 + piping[None, :] * 0.85
        info.update({"piping_texels": int(ring.sum()), "piping_color": np.round(piping, 4).tolist()})
    if text:
        # text: u along the arc, v across a constant-height box centred on the band centreline
        radius_f = np.sqrt((f[:, 0] - cx) ** 2 + (f[:, 2] - cz) ** 2)
        r_units = np.median(radius_f[fill_sel]) if fill_sel.any() else 0.3
        arc_len = 2 * np.radians(theta_span_deg) * r_units * float(np.mean(size[[0, 2]]))
        h_box = float(np.median(band[fill_sel])) * (1 - 2 * v_pad) if fill_sel.any() else 0.05
        band_h = h_box * size[1]
        aspect = max(arc_len / max(band_h, 1e-6), 1.5)
        th = 256
        tw = int(round(th * aspect))
        text_img = render_text_image(text, tw, th, font_paths or DEFAULT_FONTS, fill=(*text_color, 255))
        tarr = np.asarray(text_img, dtype=np.float32) / 255.0

        u = (theta / np.radians(theta_span_deg) + 1) / 2
        v_centered = f[:, 1] - (lo + hi) / 2
        vv = 0.5 + v_centered / max(h_box, 1e-6)
        in_text = interior & (u >= 0) & (u <= 1) & (vv >= 0) & (vv <= 1)
        px = np.clip((u[in_text] * (tw - 1)).round().astype(int), 0, tw - 1)
        py = np.clip(((1 - vv[in_text]) * (th - 1)).round().astype(int), 0, th - 1)
        ta = tarr[py, px, 3:4]
        trgb = tarr[py, px, :3]
        cur[in_text] = cur[in_text] * (1 - ta) + trgb * ta
        info.update({"text_texels": int((ta > 0.5).sum()), "text_image_size": (tw, th), "arc_len": arc_len, "band_h": band_h})

    color[ys, xs] = cur
    return color, info


class EyeDisc(NamedTuple):
    center: np.ndarray       # 3D centre of the painted iris
    normal: np.ndarray       # outward surface normal at the disc
    iris_radius: float       # radius of the dark painted iris
    disc_radius: float       # radius of the white eye disc around it


def find_eye_discs(points: np.ndarray, colors: np.ndarray, seeds: list[np.ndarray],
                   surface_normals_at: "callable | None" = None, grow_radius: float = 0.012,
                   max_radius: float = 0.16, seed_radius: float = 0.04,
                   disc_ratio_fallback: float = 1.45) -> list[EyeDisc]:
    """Locate the painted eyes on dense colored surface points.

    For each seed: take the dark points within `seed_radius`, then grow
    the set through dark points within `grow_radius` of the current set
    (a connectivity flood, so the separate dark eyebrow is never pulled
    in), capped at `max_radius` from the running centroid. The iris radius
    is the 92nd-percentile distance of that cluster from its centroid; the
    white disc radius is measured on bright, unsaturated points around it
    and falls back to `disc_ratio_fallback × iris` if that ring is not
    clearly separable from the cream face."""
    P = np.asarray(points, dtype=np.float64)
    rgb = np.clip(np.asarray(colors, dtype=np.float64), 0, 1)
    sat, val = _hsv_sat_val(rgb)
    dark_mask = val < 0.38          # the painted iris is dark brown, not black
    white_mask = (sat < 0.10) & (val > 0.86)
    dark_idx = np.where(dark_mask)[0]
    if len(dark_idx) < 20:
        raise ValueError("find_eye_discs: no dark pixels to grow an iris from")
    dark_tree = cKDTree(P[dark_idx])
    white_idx = np.where(white_mask)[0]
    white_tree = cKDTree(P[white_idx]) if len(white_idx) else None

    discs = []
    for seed in seeds:
        seed = np.asarray(seed, dtype=np.float64)
        current = set(dark_tree.query_ball_point(seed, seed_radius))
        if not current:
            # fall back to the nearest dark point
            current = {int(dark_tree.query(seed)[1])}
        frontier = set(current)
        centroid = P[dark_idx[list(current)]].mean(axis=0)
        for _ in range(200):
            if not frontier:
                break
            pts = P[dark_idx[list(frontier)]]
            neigh = dark_tree.query_ball_point(pts, grow_radius)
            new = set()
            for lst in neigh:
                new.update(lst)
            new -= current
            if not new:
                break
            cand = np.array(sorted(new))
            keep = np.linalg.norm(P[dark_idx[cand]] - centroid, axis=1) < max_radius
            cand = cand[keep]
            if len(cand) == 0:
                break
            current.update(cand.tolist())
            frontier = set(cand.tolist())
            centroid = P[dark_idx[list(current)]].mean(axis=0)
        members = P[dark_idx[list(current)]]
        d = np.linalg.norm(members - centroid, axis=1)
        iris_r = float(np.percentile(d, 92))
        disc_r = disc_ratio_fallback * iris_r
        if white_tree is not None:
            ring = white_tree.query_ball_point(centroid, 2.4 * iris_r)
            if len(ring) > 30:
                dw = np.linalg.norm(P[white_idx[ring]] - centroid, axis=1)
                cand_r = float(np.percentile(dw, 93))
                if 1.15 * iris_r < cand_r < 2.2 * iris_r:
                    disc_r = cand_r
        if surface_normals_at is not None:
            n = np.asarray(surface_normals_at(centroid), dtype=np.float64)
        else:
            # estimate from the local point cloud (PCA smallest axis, oriented outward from the seed side)
            local = P[dark_idx[list(current)]] - centroid
            _, _, vt = np.linalg.svd(local, full_matrices=False)
            n = vt[-1]
            if n[2] < 0:
                n = -n
        n = n / max(np.linalg.norm(n), 1e-9)
        discs.append(EyeDisc(center=centroid, normal=n, iris_radius=iris_r, disc_radius=disc_r))
    return discs


def blank_eye_discs(color: np.ndarray, posmap: PositionMap, discs: list[EyeDisc], radius_scale: float = 1.08,
                    shadow: float = 0.22, feather: float = 0.18) -> np.ndarray:
    """Paint over each eye disc with the median colour of the face ring
    just outside it, darkened toward the rim (a soft socket shadow) and
    feathered outward so there is no hard edge."""
    color = color.astype(np.float32).copy()
    ys, xs = np.nonzero(posmap.mask)
    pos = posmap.position[ys, xs].astype(np.float64)
    cur = color[ys, xs]
    for disc in discs:
        r = disc.disc_radius * radius_scale
        d = np.linalg.norm(pos - disc.center, axis=1)
        ring = (d > r * 1.05) & (d < r * 1.35)
        sat, val = _hsv_sat_val(cur[ring])
        cream = ring.copy()
        cream[ring] = (sat < 0.35) & (val > 0.55)
        base = np.median(cur[cream], axis=0) if cream.sum() > 30 else np.array([0.95, 0.92, 0.86], dtype=np.float32)
        inside = d < r * (1 + feather)
        # shadow strongest at the rim, fading toward the centre and outward
        t_in = np.clip(d / r, 0, 1)
        rim_shadow = shadow * (t_in ** 2)
        outer = np.clip((d - r) / (r * feather), 0, 1)
        alpha = np.where(d <= r, 1.0, 1.0 - outer)
        new = base[None, :] * (1 - rim_shadow * (1 - outer))[:, None]
        a = (inside * alpha)[:, None]
        cur = cur * (1 - a) + new * a
    color[ys, xs] = cur
    return color


def grade(color: np.ndarray, saturation: float = 1.06, contrast: float = 1.03, gamma: float = 1.0) -> np.ndarray:
    c = np.clip(color.astype(np.float32), 0, 1)
    lum = (0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2])[..., None]
    c = lum + (c - lum) * saturation
    c = 0.5 + (c - 0.5) * contrast
    if gamma != 1.0:
        c = np.clip(c, 0, 1) ** gamma
    return np.clip(c, 0, 1)


def to_image(color: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(color, 0, 1) * 255 + 0.5).astype(np.uint8))
