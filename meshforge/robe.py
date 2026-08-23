"""A kente robe: a real garment mesh with air between the cloth and the
owl, not a print on its skin.

`meshforge.kente`'s vest and sash are *decals* — the body's own triangles
duplicated 6 mm out along their normals and painted with the weave. That
is the right tool for the collar lettering (ink on a scarf) and the wrong
one for clothing: it traces every bump of the belly, ends in the region
map's jagged label boundaries, and reads as body paint. A worn kente
cloth is ample. It hangs from wherever it is held, falls straight past
the widest point of the body instead of following the body back in, and
stands off the skin by a hand's width at the hem.

This module builds that garment from measurements of the owl's own mesh
rather than from a guessed silhouette:

  radial_scan        cast one ray per (angle, height) cell from a vertical
                     axis through the torso and record every surface it
                     crosses, with the region label of the face it hit.
  torso_envelope     the body's own radius per cell — first crossing where
                     that crossing is torso, taken from the −X half and
                     mirrored (the +X half is a solid block of folded wing,
                     tablet and mislabelled leg; the body under it can't be
                     measured, but it is the same body).
  obstacles          everything else the rays cross — wing roots, the
                     folded wing, the hand, the tablet — as (inner radius)
                     per cell, classified purely by *geometry* (farther out
                     than the envelope plus a tolerance), so a wing that the
                     region map mislabelled `leg_right` is still a wing.
  neckline           the highest the cloth reaches per angle: just under
                     the collar's measured lower rim all round — the neck
                     base is the one place on this body narrower than what
                     is below it, i.e. the only thing a garment can hang
                     from. Both wings' fused roots are draped over (the
                     raised wing's free part wears a sleeve, meshforge.sleeve);
                     a hand held out in front gets a slit, the cloth hangs
                     behind it.
  gravity_hull       the cumulative-from-the-top maximum of the envelope
                     below the top edge: cloth supported at the top edge
                     falls vertically from the widest point it passes.
  build_kente_robe   hull + a clearance that grows with the fall + vertical
                     pleats that deepen toward the hem → a (theta, t) grid
                     mesh with one rectangular UV chart, the weave baked by
                     position exactly like the decals (`kente.bake_weave_maps`).

What the cloth clears is then *measured*, not assumed: `clearance` reports
the minimum distance between the robe and the body surface and the
fraction of robe vertices that ended up inside it, and `posed_clearance`
repeats that through the baked clips, because a hem that clears the legs
at rest can still be kicked through on the hop.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.spatial import cKDTree
from trimesh.intersections import mesh_plane

from meshforge import fk
from meshforge.bake import bake_position_map
from meshforge.kente import bake_weave_maps
from meshforge.regions import RegionMap, collar_bounds_at
from meshforge.rigexport import AnimationSpec, MaterialSpec, PrimitiveSpec
from meshforge.textile import WeaveParams, to_image

TORSO_LABELS: tuple[str, ...] = ("chest", "body", "neck", "leg_left", "leg_right", "foot_left", "foot_right")


@dataclasses.dataclass(frozen=True)
class RobeParams:
    """Every length is in mesh units unless the name ends in `_frac`
    (fraction of the mesh's bounding-box height). Defaults were tuned on
    the AI-CCORE owl (1.37 × 1.90 × 1.18) by rendering, not by numbers."""
    n_theta: int = 144                  # columns around the axis (also the scan resolution)
    n_rows: int = 64                    # rows from the top edge to the hem
    hem_frac: float = 0.12              # hem height; the feet start at ~0.077
    top_gap_frac: float = 0.005         # how far under the collar's measured lower rim the neckline stops
    obstacle_margin_frac: float = 0.02  # vertical margin an armhole / slit keeps around what passes through it
    hole_margin_deg: float = 3.0        # ... and the angular margin
    hole_smooth_cells: float = 1.5      # rounding of a hole's edge, in scan cells (never shrinks the hole)
    root_tolerance: float = 0.015       # a wing vertex this close to the fused surface belongs to the root that is swept
    clearance_top: float = 0.008        # air between cloth and body at the neckline (tucked under the scarf)
    clearance_hem: float = 0.10         # ... and at the hem, after the full fall
    clearance_power: float = 1.0        # how the clearance grows with the fall (1 = linear)
    clearance_band_frac: float = 0.06   # fall distance over which the cloth leaves the shoulders for the global clearance
    edge_band: float = 0.035            # width (mesh units) of the plain binding along every edge; 0 = none
    edge_band_color: tuple[float, float, float] = (0.80, 0.62, 0.05)   # gold
    drape_clearance: float = 0.05       # extra air over a draped-over limb: it moves under the cloth (tablet_show, wave)
    drape_merge_gap: float = 0.13       # a drape-over solid this close behind the first one is part of the same shoulder (the folded wing's upper arm pressed to the chest), not something held out
    foreign_tolerance: float = 0.08     # a torso-labelled surface this far outside the mirrored torso is an appendage
    surface_fuzz: float = 0.02          # crossings this close together are one solid (the belly's embossed circuit trace)
    obstacle_near: float = 0.03         # radial margin the cloth keeps from anything it passes behind or over
    drape_over: tuple[str, ...] = ("wing_right", "wing_left", "wing_left_tip", "tail")   # fused surfaces the cloth falls over (both wings' roots, the tail)
    drape_always: tuple[str, ...] = ("tail",)              # ... even when the ray met something else first (the legs)
    stop_below: tuple[str, ...] = ()                       # the cloth ends under these instead (an armhole; the raised wing wears a sleeve now)
    pleats: int = 10
    pleat_depth: float = 0.035          # radial amplitude of the pleats at the hem
    pleat_power: float = 1.5            # pleats deepen with the fall
    hem_wave: float = 0.010             # hem height undulation in step with the pleats
    edge_smooth_deg: float = 10.0       # neckline smoothing (the coarse collar rim zigzags)
    hull_smooth_deg: float = 6.0
    strips_around: int = 18             # kente strips around the circumference; keep a multiple of strip_cycle
    blocks_tall: float = 5.0            # weave blocks over the robe's full height
    texture_size: tuple[int, int] = (2048, 512)   # (width, height): one chart, 2π around × the fall down
    bake_normal: bool = False
    bump_strength: float = 0.15
    roughness: float = 0.88
    material_name: str = "owl_kente_robe"


class RadialScan(NamedTuple):
    theta: np.ndarray          # (n_theta,) cell-centre angles, 0 = +Z (front), +ve toward +X, radians
    y: np.ndarray              # (n_rows,) ascending world heights
    axis_xz: np.ndarray        # (2,) the vertical axis every ray starts on
    radius: list               # radius[row][col] -> (k,) ascending crossing radii
    label: list                # label[row][col] -> list[str] region label of each crossing


def _majority_label(face_labels: np.ndarray) -> np.ndarray:
    """Per face, the label two or more of its vertices share (else vertex 0's)."""
    a, b, c = face_labels[:, 0], face_labels[:, 1], face_labels[:, 2]
    out = a.copy()
    bc = (b == c) & (a != b)
    out[bc] = b[bc]
    return out


def radial_scan(mesh: trimesh.Trimesh, labels: np.ndarray, axis_xz: np.ndarray,
                y_levels: np.ndarray, n_theta: int) -> RadialScan:
    """Slice the mesh at every height in `y_levels`, shoot `n_theta` rays
    outward from the axis in the slice plane and record every segment each
    ray crosses, nearest first, with the label of the face that segment
    came from."""
    labels = np.asarray(labels)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    axis_xz = np.asarray(axis_xz, dtype=np.float64)
    theta = -np.pi + (np.arange(n_theta) + 0.5) * (2 * np.pi / n_theta)
    d = np.stack([np.sin(theta), np.cos(theta)], axis=1)           # (T, 2) ray directions in (x, z)
    radius: list = []
    label: list = []
    for y in y_levels:
        dots = V[:, 1] - y
        segs, fids = mesh_plane(mesh, np.array([0.0, 1.0, 0.0]), np.array([0.0, y, 0.0]),
                                return_faces=True, cached_dots=dots)
        row_r: list = [np.zeros(0)] * n_theta
        row_l: list = [[]] * n_theta
        if len(segs):
            p = segs[:, 0][:, [0, 2]] - axis_xz                        # (S, 2)
            e = segs[:, 1][:, [0, 2]] - segs[:, 0][:, [0, 2]]          # (S, 2)
            seg_label = _majority_label(labels[mesh.faces[fids]])
            # ray: t*d = p + s*e  ->  t = (p x e)/(d x e), s = (p x d)/(d x e)
            dxe = d[:, 0, None] * e[None, :, 1] - d[:, 1, None] * e[None, :, 0]       # (T, S)
            pxe = p[:, 0] * e[:, 1] - p[:, 1] * e[:, 0]                                # (S,)
            pxd = p[None, :, 0] * d[:, 1, None] - p[None, :, 1] * d[:, 0, None]       # (T, S)
            ok = np.abs(dxe) > 1e-12
            safe = np.where(ok, dxe, 1.0)
            t = np.where(ok, pxe[None, :] / safe, -1.0)
            s = np.where(ok, pxd / safe, -1.0)
            hit = ok & (s >= 0.0) & (s <= 1.0) & (t > 1e-9)
            for k in range(n_theta):
                cols = np.where(hit[k])[0]
                if len(cols) == 0:
                    continue
                # a ray through a vertex of the slice polyline hits both
                # segments sharing it at the same radius; keep one, or the
                # duplicate flips every enter/exit pair beyond it
                order = np.argsort(t[k, cols], kind="stable")
                tt = t[k, cols][order]
                keep = np.concatenate([[True], np.diff(tt) > 1e-9])
                row_r[k] = tt[keep]
                row_l[k] = [str(seg_label[c]) for c in cols[order][keep]]
        radius.append(row_r)
        label.append(row_l)
    return RadialScan(theta=theta, y=np.asarray(y_levels, dtype=np.float64), axis_xz=axis_xz,
                      radius=radius, label=label)


def _circular_interp_nan(row: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(row)
    if ok.all() or not ok.any():
        return row
    n = len(row)
    xs = np.arange(n)
    x_ok = np.concatenate([xs[ok] - n, xs[ok], xs[ok] + n])
    v_ok = np.tile(row[ok], 3)
    out = row.copy()
    out[~ok] = np.interp(xs[~ok], x_ok, v_ok)
    return out


def _fill_nan_grid(grid: np.ndarray) -> np.ndarray:
    """Circular interpolation along theta per row, then linear along rows
    per column (edge rows held), so every cell carries a value."""
    g = np.array([_circular_interp_nan(r) for r in grid])
    for c in range(g.shape[1]):
        col = g[:, c]
        ok = ~np.isnan(col)
        if ok.all():
            continue
        if not ok.any():
            raise ValueError("_fill_nan_grid: a column has no measured value at any height")
        idx = np.arange(len(col))
        col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
        g[:, c] = col
    return g


def torso_envelope(scan: RadialScan, torso_labels: Sequence[str] = TORSO_LABELS,
                   mirror: bool = True) -> np.ndarray:
    """(n_rows, n_theta) radius of the body itself. Per cell the first
    crossing counts only if it is labelled torso — the fused root of the
    raised wing is not the body, even where the region map's crop box
    labelled flank under it `wing_left`; those cells are interpolated from
    their neighbours instead. With `mirror`, the +X half is replaced by the
    −X half reflected, because on this asset the +X half is not measurable
    (see the module docstring)."""
    n_rows, n_theta = len(scan.y), len(scan.theta)
    env = np.full((n_rows, n_theta), np.nan)
    torso = set(torso_labels)
    for r in range(n_rows):
        for c in range(n_theta):
            rad = scan.radius[r][c]
            if len(rad) and scan.label[r][c][0] in torso:
                env[r, c] = rad[0]
    if mirror:
        # theta_k = -pi + 2pi(k+0.5)/n  ->  -theta_k is column n-1-k, exactly
        left = scan.theta < 0
        env[:, ~left] = env[:, ::-1][:, ~left]
    return _fill_nan_grid(env)


class CellKinds(NamedTuple):
    """What each (row, column) ray met, reduced to what the cloth needs."""
    kind: np.ndarray        # (rows, cols) str: "torso" | "drape" | "stop" | "none"
    first_r: np.ndarray     # (rows, cols) radius of the first crossing (nan if none)
    drape_r: np.ndarray     # (rows, cols) outermost radius the cloth must fall over (-inf if nothing)
    held: list              # held[row][col] -> list of (r_in, r_out) solids met *after* the first crossing


def classify_cells(scan: RadialScan, envelope: np.ndarray, params: RobeParams,
                   torso_labels: Sequence[str] = TORSO_LABELS) -> CellKinds:
    """The first crossing is the solid the cloth must clear: the body
    (`torso`), something fused onto it the cloth falls over (`drape`: the
    folded wing, the tail), or something it must end below (`stop`: the
    wing that waves). A torso-labelled first crossing far outside the
    mirrored body is an appendage the region map mislabelled (the folded
    wing's tip below the crop box is `leg_right` on this asset) and is
    draped over. Later crossings are solids held away from the body — the
    hand, the tablet — recorded as radial intervals: the cloth may pass
    behind them but not through them. Except a drape-over solid that starts
    within `drape_merge_gap` of the solid before it: that is the same
    shoulder (the folded wing's upper arm lies against the chest with a
    finger of air between — measured 0.05–0.12 on the owl, the forearm and
    hand 0.13–0.38), and the cloth falls over it too."""
    n_rows, n_theta = envelope.shape
    kind = np.full((n_rows, n_theta), "none", dtype=object)
    first_r = np.full((n_rows, n_theta), np.nan)
    drape_r = np.full((n_rows, n_theta), -np.inf)
    held: list = []
    torso, drape, stop, always = set(torso_labels), set(params.drape_over), set(params.stop_below), set(params.drape_always)
    for r in range(n_rows):
        held_row = []
        for c in range(n_theta):
            rad = scan.radius[r][c]
            lab = scan.label[r][c]
            cell_held = []
            if len(rad):
                # a ray grazing raised surface detail (the belly's embossed
                # circuit trace, feather edges) exits and re-enters the same
                # solid within a hair: merge crossings closer than
                # `surface_fuzz` into one solid ending at the last of them
                end, k = rad[0], 1
                while k + 1 < len(rad) and rad[k] - end < params.surface_fuzz:
                    end, k = rad[k + 1], k + 2
                first_r[r, c] = end
                l0 = lab[0]
                if l0 in stop:
                    kind[r, c] = "stop"
                elif l0 in drape:
                    kind[r, c] = "drape"
                    drape_r[r, c] = end
                elif l0 in torso and end > envelope[r, c] + params.foreign_tolerance:
                    kind[r, c] = "drape"
                    drape_r[r, c] = end
                elif l0 in torso:
                    kind[r, c] = "torso"
                else:
                    kind[r, c] = "drape"
                    drape_r[r, c] = end
                # everything after the first solid comes in enter/exit pairs
                held_label = None
                for k in range(k, len(rad), 2):
                    r_in = rad[k]
                    r_out = rad[k + 1] if k + 1 < len(rad) else rad[k]
                    if lab[k] in always:
                        drape_r[r, c] = max(drape_r[r, c], r_out)
                        held_label = None           # a held solid beyond it is a new solid
                    elif lab[k] in stop:
                        kind[r, c] = "stop"
                        held_label = None
                    elif lab[k] in drape and r_in - end < params.drape_merge_gap and kind[r, c] != "stop" and held_label is None:
                        # measured from the *first* solid, not from the last
                        # merged one: chaining would swallow the hand and the
                        # tablet behind the upper arm (measured: the cloth then
                        # fell outside the tablet and covered it)
                        kind[r, c] = "drape"
                        drape_r[r, c] = max(drape_r[r, c], r_out)
                    elif cell_held and lab[k] == held_label:
                        # the same appendage again further out (the palm,
                        # then the tablet it grips): one solid, the cloth
                        # does not thread between them
                        cell_held[-1] = (cell_held[-1][0], float(r_out))
                    else:
                        cell_held.append((float(r_in), float(r_out)))
                        held_label = lab[k]
            held_row.append(cell_held)
        held.append(held_row)
    return CellKinds(kind=kind, first_r=first_r, drape_r=drape_r, held=held)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def clearance_profile(y: np.ndarray, y_ref: float, y_hem: float, params: RobeParams) -> np.ndarray:
    """Global clearance at height `y`: `clearance_top` at the highest
    support `y_ref`, growing to `clearance_hem` at the hem."""
    fall = np.clip((y_ref - y) / max(y_ref - y_hem, 1e-9), 0.0, 1.0)
    return params.clearance_top + (params.clearance_hem - params.clearance_top) * fall ** params.clearance_power


def gravity_hull(envelope: np.ndarray, cells: CellKinds, y: np.ndarray, y_top: np.ndarray,
                 params: RobeParams) -> np.ndarray:
    """Cumulative maximum, from each column's own top edge downward, of
    the surface the cloth must clear: the mirrored body where the body is
    exposed, the fused surface where something is draped over. Lightly
    smoothed around the axis and never pulled inside either."""
    # the mirrored envelope is a floor, not the truth: where the ray met a
    # real surface (the body's own, or something fused onto it) the cloth
    # clears *that* — the body is not perfectly symmetric (measured: up to
    # 0.07 wider on the +X side of the back) and a fused surface is what
    # is actually there. Only `stop` cells keep the envelope, and the cloth
    # ends below them anyway.
    measured = np.where(np.isnan(cells.first_r), -np.inf, cells.first_r)
    measured = np.where(cells.kind == "stop", -np.inf, measured)
    base = np.maximum(np.maximum(envelope, cells.drape_r), measured)
    hull = np.where(y[:, None] <= y_top[None, :] + 1e-9, base, -np.inf)
    hull = np.maximum.accumulate(hull[::-1], axis=0)[::-1]
    hull = np.where(np.isfinite(hull), hull, base)
    sig = params.hull_smooth_deg / 360.0 * envelope.shape[1]
    if sig > 0:
        grown = maximum_filter1d(hull, size=2 * int(round(sig)) + 1, mode="wrap", axis=1)
        hull = gaussian_filter1d(grown, sigma=sig, mode="wrap", axis=1)
        hull = np.maximum(hull, base)
    return hull


def cloth_radius(hull: np.ndarray, y: np.ndarray, y_top: np.ndarray, y_ref: float, y_hem: float,
                 params: RobeParams, size_y: float | None = None) -> np.ndarray:
    """Hull plus the clearance (no pleats): `clearance_top` at each
    column's own top edge, reaching the global profile `clearance_band_frac`
    of the height below it."""
    size_y = (y_ref - y_hem) / max(1.0 - params.hem_frac, 1e-9) if size_y is None else size_y
    fall_local = np.clip(y_top[None, :] - y[:, None], 0.0, None)
    c_global = clearance_profile(y, y_ref, y_hem, params)[:, None]
    blend = _smoothstep(fall_local / max(params.clearance_band_frac * size_y, 1e-9))
    return hull + params.clearance_top + (c_global - params.clearance_top) * blend


def _interp_columns(values_at_cells: np.ndarray, theta_cells: np.ndarray, theta_query: np.ndarray) -> np.ndarray:
    """Circular linear interpolation of per-cell values (last axis) to arbitrary angles."""
    n = len(theta_cells)
    period = 2 * np.pi
    x = np.concatenate([theta_cells - period, theta_cells, theta_cells + period])
    if values_at_cells.ndim == 1:
        return np.interp(theta_query, x, np.tile(values_at_cells, 3))
    return np.stack([np.interp(theta_query, x, np.tile(row, 3)) for row in values_at_cells])


def _neckline(scan: RadialScan, envelope: np.ndarray, region_map: RegionMap, band_frame,
              bmin: np.ndarray, size: np.ndarray, params: RobeParams) -> np.ndarray:
    """Per scan column, the height of the neckline: the collar band's lower
    rim minus `top_gap_frac`, so the cloth tucks under the scarf all the
    way round — the neck base is the one place on this body that is
    narrower than what is below it, i.e. the only thing a garment can
    hang from. `regions.measure_collar` describes the rim per angle around
    *its own* centre (in bbox fractions), not the scan axis — on the owl
    the two differ by 0.12 in depth — so each scan ray's collar-height hit
    is expressed in the collar's own angle before the rim is read. With a
    `collar.BandFrame` the fitted rim of the red facing is used across
    the front (the coarse rim zigzags) and blended into the coarse one
    past the frame's span."""
    collar = region_map.collar
    cx, cz = collar["centre_xz"]
    size_y = float(size[1])
    y_collar = bmin[1] + float(np.nanmean(collar["y_lo"])) * size_y
    row = int(np.argmin(np.abs(scan.y - y_collar)))
    r = envelope[row]
    px = scan.axis_xz[0] + r * np.sin(scan.theta)
    pz = scan.axis_xz[1] + r * np.cos(scan.theta)
    fx = (px - bmin[0]) / size[0]
    fz = (pz - bmin[2]) / size[2]
    theta_collar = np.arctan2(fx - cx, fz - cz)
    lo, _ = collar_bounds_at(collar, theta_collar)
    if band_frame is not None:
        lo_frame, _ = band_frame.bounds(theta_collar)
        inside = np.clip((band_frame.span - np.abs(theta_collar)) / 0.25, 0.0, 1.0)
        w = inside * inside * (3.0 - 2.0 * inside)
        lo = w * lo_frame + (1.0 - w) * lo
    y_top = bmin[1] + (lo - params.top_gap_frac) * size_y
    sig = max(params.edge_smooth_deg / 360.0 * len(scan.theta) / 2.0, 0.5)
    y_top = gaussian_filter1d(y_top, sigma=sig, mode="wrap")
    return np.minimum(y_top, scan.y.max() - 1e-6)


def _dilate_periodic(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Binary dilation by an ellipse of `rows` x `cols` cells, periodic in theta."""
    ry, rx = max(int(rows), 0), max(int(cols), 0)
    if ry == 0 and rx == 0:
        return mask.copy()
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    se = (yy / max(ry, 1)) ** 2 + (xx / max(rx, 1)) ** 2 <= 1.0 + 1e-9
    if rx:
        padded = np.concatenate([mask[:, -rx:], mask, mask[:, :rx]], axis=1)
    else:
        padded = mask
    out = binary_dilation(padded, structure=se)
    return out[:, rx:rx + mask.shape[1]] if rx else out


def _components_periodic(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components (4-connectivity), merging those that touch
    across the theta seam."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return lab, int(n)
    parent = list(range(n + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(lab[:, 0], lab[:, -1]):
        if a and b and find(a) != find(b):
            parent[find(a)] = find(b)
    ids: dict = {}
    remap = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        root = find(i)
        ids.setdefault(root, len(ids) + 1)
        remap[i] = ids[root]
    return remap[lab], len(ids)


class HoleField(NamedTuple):
    """A hole in the cloth as a signed-distance field over the scan grid,
    in the metric plane (theta * `s_scale`, height): positive outside the
    hole, negative inside, zero exactly on the dilated mask's boundary.
    A star-shaped curve (the first version of this) inflated every
    elongated mask — the diagonal slit for the forearm and tablet grew
    toward the neckline by the difference between neighbouring polar
    bins — where a field is exact whatever the shape."""
    centre: np.ndarray     # (theta_c, y_c) of the mask
    extent: np.ndarray     # (theta_lo, theta_hi, y_lo, y_hi), theta unwrapped around the centre
    s_scale: float
    field: object          # RegularGridInterpolator over (y, theta), periodic columns extended
    y_range: tuple         # (y0, y1) of the scanned rows

    def signed_distance(self, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Heights outside the scanned rows read as the nearest row (a hole
        on the top row continues upward, never extrapolates)."""
        theta = np.angle(np.exp(1j * np.asarray(theta, dtype=np.float64)))
        y = np.clip(np.asarray(y, dtype=np.float64), self.y_range[0], self.y_range[1])
        pts = np.stack([y, theta], axis=-1)
        return self.field(pts)


def _hole_field(mask: np.ndarray, theta: np.ndarray, y: np.ndarray, s_scale: float,
                smooth_cells: float = 1.0) -> HoleField:
    n_rows, n_theta = mask.shape
    if n_rows < 2:
        raise ValueError("_hole_field: needs at least two scanned rows")
    tiled = np.concatenate([mask, mask, mask], axis=1)          # distances wrap around the seam
    dy = float(y[1] - y[0])
    dth = float(theta[1] - theta[0]) * float(s_scale)
    d_out = distance_transform_edt(~tiled, sampling=(dy, dth))
    d_in = distance_transform_edt(tiled, sampling=(dy, dth))
    F = (d_out - d_in)[:, n_theta:2 * n_theta]
    if smooth_cells > 0:
        # round the staircase without ever shrinking the hole (a blur pulls
        # convex corners inward; the minimum keeps every cut cell cut)
        F = np.minimum(gaussian_filter(F, sigma=smooth_cells, mode=("nearest", "wrap")), F)
    th_ext = np.concatenate([[theta[-1] - 2 * np.pi], theta, [theta[0] + 2 * np.pi]])
    F_ext = np.concatenate([F[:, -1:], F, F[:, :1]], axis=1)
    field = RegularGridInterpolator((y, th_ext), F_ext, bounds_error=False, fill_value=None)
    rows, cols = np.nonzero(mask)
    th, yy = theta[cols], y[rows]
    thc = float(np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th))))
    dth_c = np.angle(np.exp(1j * (th - thc)))
    extent = np.array([thc + dth_c.min(), thc + dth_c.max(), yy.min(), yy.max()])
    return HoleField(centre=np.array([thc, float(yy.mean())]), extent=extent, s_scale=float(s_scale), field=field,
                     y_range=(float(y[0]), float(y[-1])))


def _cell_of(theta: np.ndarray, y: np.ndarray, scan: RadialScan) -> tuple[np.ndarray, np.ndarray]:
    n_theta = len(scan.theta)
    col = np.floor((theta + np.pi) / (2 * np.pi) * n_theta).astype(int) % n_theta
    row = np.clip(np.rint((y - scan.y[0]) / (scan.y[1] - scan.y[0])).astype(int), 0, len(scan.y) - 1)
    return row, col


def hole_masks(scan: RadialScan, envelope: np.ndarray, cells: CellKinds, hull: np.ndarray, y_top: np.ndarray,
               y_hem: float, y_ref: float, size_y: float, mesh: trimesh.Trimesh, labels: np.ndarray,
               params: RobeParams, torso_labels: Sequence[str] = TORSO_LABELS,
               sweep_vertices: Sequence[np.ndarray] | None = None,
               obstacle_points: Sequence[np.ndarray] | None = None,
               surface: "_Surface | None" = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(armhole cells, slit cells, combined dilated mask) on the scan grid.

    Armholes: every `stop` cell — the fused root of the wing that waves —
    plus, for each posed copy of the body in `sweep_vertices` (the wave
    clip sampled through its range), the cells that root's vertices move
    into. Slits: every cell where a held-out solid (the hand, the tablet)
    sits inside the radii the cloth would span there, plus the cells those
    vertices move into when posed (`tablet_show`). Both dilated by
    `hole_margin_deg` x `obstacle_margin_frac`. `obstacle_points` are
    world points of things that are not part of the body at all but must
    not be cut through either — the sleeve where it passes out through the
    tunic's shoulder — and are cut around wherever they reach the cloth's
    surface from inside or stand beyond it.

    With `surface` (the cloth as it will actually be placed: hull,
    clearance, drape slack and pleats) the tests use the real radius per
    cell ± `obstacle_near`; without it, a conservative band around the
    clearance-only radius (pleats and drape slack both ways), which on the
    owl cut the hand slit up to the neckline — the hand is held 0.09 off
    the shoulder cloth there, and the cloth passes behind it."""
    n_rows, n_theta = envelope.shape
    in_fabric = (scan.y >= y_hem - 1e-9)[:, None] & (scan.y[:, None] <= y_top[None, :] + params.obstacle_margin_frac * size_y)
    # [lo, hi] is the band of radii the cloth may occupy per cell, already
    # including `obstacle_near` on both sides
    if surface is not None:
        TH, Y = np.meshgrid(scan.theta, scan.y)
        R = surface.radius(TH, Y)
        lo = R - params.obstacle_near
        hi = R + params.obstacle_near
    else:
        R = cloth_radius(hull, scan.y, y_top, y_ref, y_hem, params, size_y=size_y)
        lo = (minimum_filter1d(minimum_filter1d(R, size=3, mode="wrap", axis=1), size=3, mode="nearest", axis=0)
              - params.pleat_depth - params.obstacle_near)
        hi = (maximum_filter1d(maximum_filter1d(R, size=3, mode="wrap", axis=1), size=3, mode="nearest", axis=0)
              + params.pleat_depth + params.drape_clearance + params.obstacle_near)

    arm = (cells.kind == "stop") & in_fabric
    slit = np.zeros_like(arm)
    for r in range(n_rows):
        for c in range(n_theta):
            if not in_fabric[r, c] or arm[r, c]:
                continue
            for r_in, r_out in cells.held[r][c]:
                if r_in <= hi[r, c] and r_out >= lo[r, c]:
                    slit[r, c] = True
                    break

    if sweep_vertices:
        V = np.asarray(mesh.vertices, dtype=np.float64)
        th0 = np.arctan2(V[:, 0] - scan.axis_xz[0], V[:, 2] - scan.axis_xz[1])
        r0 = np.hypot(V[:, 0] - scan.axis_xz[0], V[:, 2] - scan.axis_xz[1])
        row0, col0 = _cell_of(th0, V[:, 1], scan)
        first = np.where(np.isnan(cells.first_r), np.inf, cells.first_r)
        stop_lab = np.isin(labels, list(params.stop_below))
        root_set = stop_lab & (cells.kind[row0, col0] == "stop") & (r0 <= first[row0, col0] + params.root_tolerance)
        held_set = ~np.isin(labels, list(torso_labels)) & ~stop_lab & (r0 > first[row0, col0] + params.obstacle_near)
        for P in sweep_vertices:
            P = np.asarray(P, dtype=np.float64)
            th = np.arctan2(P[:, 0] - scan.axis_xz[0], P[:, 2] - scan.axis_xz[1])
            rr = np.hypot(P[:, 0] - scan.axis_xz[0], P[:, 2] - scan.axis_xz[1])
            row, col = _cell_of(th, P[:, 1], scan)
            ok = in_fabric[row, col]
            sel = root_set & ok
            arm[row[sel], col[sel]] = True
            sel = held_set & ok & (rr >= lo[row, col]) & (rr <= hi[row, col])
            slit[row[sel], col[sel]] = True

    for P in (obstacle_points or []):
        P = np.asarray(P, dtype=np.float64)
        if len(P) == 0:
            continue
        th = np.arctan2(P[:, 0] - scan.axis_xz[0], P[:, 2] - scan.axis_xz[1])
        rr = np.hypot(P[:, 0] - scan.axis_xz[0], P[:, 2] - scan.axis_xz[1])
        row, col = _cell_of(th, P[:, 1], scan)
        sel = in_fabric[row, col] & (rr >= lo[row, col])
        slit[row[sel], col[sel]] = True

    rows = int(np.ceil(params.obstacle_margin_frac * size_y / max(scan.y[1] - scan.y[0], 1e-9)))
    cols = int(np.ceil(params.hole_margin_deg / (360.0 / n_theta)))
    combined = _dilate_periodic(arm | slit, rows, cols)
    return arm, slit, combined


def cut_grid(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Triangulate a (rows, cols) parameter grid and cut it along the zero
    level of the signed distance `F` (positive inside the cloth). Faces
    with no inside vertex are dropped; every outside vertex a kept face
    still uses is snapped onto the edge by walking toward its most-inside
    8-neighbour to the zero crossing. Returns (faces indexed row*cols+col,
    src_a, src_b, frac, snapped count): a vertex's parameters after the
    cut are `param[src_a] + frac * (param[src_b] - param[src_a])`
    (unmoved vertices have src_a == src_b == themselves, frac 0)."""
    F = np.asarray(F, dtype=np.float64)
    rows, cols = F.shape
    r_idx, c_idx = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    v00 = (r_idx * cols + c_idx).ravel()
    v10, v11, v01 = v00 + cols, v00 + cols + 1, v00 + 1
    faces = np.stack([np.stack([v00, v10, v11], axis=1), np.stack([v00, v11, v01], axis=1)], axis=1).reshape(-1, 3)
    Fv = F.ravel()
    faces = faces[(Fv[faces] >= 0.0).any(axis=1)]
    n = rows * cols
    src_a, src_b, frac = np.arange(n), np.arange(n), np.zeros(n)
    used = np.unique(faces)
    snapped = 0
    for v in used[Fv[used] < 0.0]:
        r, c = divmod(int(v), cols)
        best, best_f = None, -np.inf
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if F[rr, cc] > best_f:
                    best, best_f = (rr, cc), F[rr, cc]
        if best is None or best_f < 0.0:
            continue
        src_a[v] = best[0] * cols + best[1]
        src_b[v] = v
        frac[v] = best_f / max(best_f - F[r, c], 1e-12)      # fraction from the inside neighbour toward v
        snapped += 1
    return faces, src_a, src_b, frac, snapped


class _Surface:
    """The cloth surface as a function of (theta, height): hull + clearance
    + slack over draped limbs + pleats, interpolated from the scan cells so
    vertices snapped to a hole boundary land on the same surface as the
    grid."""

    def __init__(self, scan: RadialScan, hull: np.ndarray, draped: np.ndarray, y_top: np.ndarray,
                 y_hem: float, y_ref: float, size_y: float, axis_xz: np.ndarray, params: RobeParams):
        th = scan.theta
        th_ext = np.concatenate([[th[-1] - 2 * np.pi], th, [th[0] + 2 * np.pi]])
        ext = lambda a: np.concatenate([a[:, -1:], a, a[:, :1]], axis=1)
        self._hull = RegularGridInterpolator((scan.y, th_ext), ext(hull), bounds_error=False, fill_value=None)
        self._draped = RegularGridInterpolator((scan.y, th_ext), ext(draped), bounds_error=False, fill_value=None)
        self.scan, self.y_top_cells = scan, y_top
        self.y_hem, self.y_ref, self.size_y, self.axis_xz, self.params = y_hem, y_ref, size_y, axis_xz, params

    def y_top(self, theta):
        return _interp_columns(self.y_top_cells, self.scan.theta, np.asarray(theta, dtype=np.float64))

    def hem(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        return self.y_hem + self.params.hem_wave * self.size_y * np.cos(self.params.pleats * theta)

    def radius(self, theta, y):
        theta = np.asarray(theta, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        pts = np.stack([np.clip(y, self.scan.y.min(), self.scan.y.max()), theta], axis=-1)
        hull = self._hull(pts)
        draped = np.clip(self._draped(pts), 0.0, 1.0)
        p = self.params
        fall = np.clip(self.y_top(theta) - y, 0.0, None)
        c_global = clearance_profile(y, self.y_ref, self.y_hem, p)
        blend = _smoothstep(fall / max(p.clearance_band_frac * self.size_y, 1e-9))
        clearance = p.clearance_top + (c_global - p.clearance_top) * blend + p.drape_clearance * draped
        fall_norm = np.clip(fall / max(self.y_ref - self.y_hem, 1e-9), 0.0, 1.0)
        pleat = p.pleat_depth * fall_norm ** p.pleat_power * np.cos(p.pleats * theta)
        return hull + clearance + pleat

    def position(self, theta, y):
        theta = np.asarray(theta, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        r = self.radius(theta, y)
        return np.stack([self.axis_xz[0] + r * np.sin(theta), y, self.axis_xz[1] + r * np.cos(theta)], axis=-1)

    def uv(self, theta, y):
        theta = np.asarray(theta, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        top, hem = self.y_top(theta), self.hem(theta)
        v = np.clip((top - y) / np.maximum(top - hem, 1e-9), 0.0, 1.0)
        return np.stack([(theta + np.pi) / (2 * np.pi), v], axis=-1)


def fabric_signed_distance(surface: _Surface, holes: Sequence[HoleField], theta: np.ndarray, y: np.ndarray,
                           include_hem: bool = False) -> np.ndarray:
    """Positive inside the cloth, zero on an edge, negative in a hole or
    above the neckline (and below the hem, with `include_hem`). Heights
    are in world units, the holes' radial distance in their param plane."""
    theta = np.asarray(theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    f = surface.y_top(theta) - y
    if include_hem:
        f = np.minimum(f, y - surface.hem(theta))
    for h in holes:
        f = np.minimum(f, h.signed_distance(theta, y))
    return f


class KenteRobe(NamedTuple):
    primitive: PrimitiveSpec
    grid_shape: tuple[int, int]      # (rows, columns incl. the duplicated seam column) of the grid it was cut from
    grid_index: np.ndarray           # (n_vertices,) row * columns + column of each surviving vertex
    axis_xz: np.ndarray
    y_top: np.ndarray                # (n_theta,) neckline height at the scan's cell angles
    holes: list                      # HoleField per armhole / slit
    info: dict


def build_kente_robe(
    mesh: trimesh.Trimesh,
    region_map: RegionMap,
    weave_params: WeaveParams,
    params: RobeParams = RobeParams(),
    torso_labels: Sequence[str] = TORSO_LABELS,
    band_frame=None,
    sweep_vertices: Sequence[np.ndarray] | None = None,
    obstacle_points: Sequence[np.ndarray] | None = None,
) -> KenteRobe:
    """Measure the body, hang the cloth from the neckline, cut the
    armhole and the hand slit, bake the weave. See the module docstring
    for the stages; every intermediate is in `info`. `band_frame` is the
    collar's fitted `collar.BandFrame` (front rim); `sweep_vertices` are
    posed copies of `mesh.vertices` through the clips that move the wings
    (the holes are cut around where the wings *go*, not just where they
    rest); `obstacle_points` are world points the cloth must also be cut
    around (the sleeve's outer surface where it leaves the tunic)."""
    if params.n_theta < 8 or params.n_rows < 2:
        raise ValueError(f"build_kente_robe: need n_theta >= 8 and n_rows >= 2, got {params.n_theta}x{params.n_rows}")
    if min(params.texture_size) < 4:
        raise ValueError(f"build_kente_robe: texture_size {params.texture_size} is too small")
    if params.strips_around % weave_params.strip_cycle != 0:
        raise ValueError(f"build_kente_robe: strips_around ({params.strips_around}) must be a multiple of the "
                         f"weave's strip_cycle ({weave_params.strip_cycle}) so the weave meets itself at the seam")
    clash = (set(params.drape_over) | set(params.drape_always)) & set(params.stop_below)
    if clash:
        raise ValueError(f"build_kente_robe: {sorted(clash)} cannot be both draped over and stopped under")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bmin, bmax = np.array(mesh.bounds)
    size = bmax - bmin
    size_y = float(size[1])
    labels = np.asarray(region_map.labels)

    core_labels = [l for l in ("chest", "body") if l in torso_labels] or list(torso_labels)
    torso_mask = np.isin(labels, core_labels)
    if torso_mask.sum() < 50:
        raise ValueError(f"build_kente_robe: the region map has no torso ({core_labels}) to dress")
    # the axis sits on the body's symmetry plane (x = bbox centre) at the
    # torso's own mean depth; the envelope is mirrored about it
    axis_xz = np.array([0.5 * (bmin[0] + bmax[0]), V[torso_mask][:, 2].mean()])

    y_hem = bmin[1] + params.hem_frac * size_y
    theta = -np.pi + (np.arange(params.n_theta) + 0.5) * (2 * np.pi / params.n_theta)
    # provisional cap (angles as if the collar were measured around the scan
    # axis) only to size the scan; the real cap is read after the scan, at
    # the angle the collar measurement itself uses
    provisional_lo, _ = collar_bounds_at(region_map.collar, theta)
    y_cap0 = bmin[1] + (np.nanmax(provisional_lo) - params.top_gap_frac) * size_y

    # scan a little past both ends so the hull and the obstacle test have context
    y_levels = np.linspace(y_hem - 0.04 * size_y, min(float(y_cap0) + 0.05 * size_y, bmax[1]), params.n_rows + 8)
    scan = radial_scan(mesh, labels, axis_xz, y_levels, params.n_theta)
    envelope = torso_envelope(scan, torso_labels=torso_labels, mirror=True)
    y_top_cells = _neckline(scan, envelope, region_map, band_frame, bmin, size, params)
    cells = classify_cells(scan, envelope, params, torso_labels=torso_labels)
    hull_cells = gravity_hull(envelope, cells, scan.y, y_top_cells, params)
    y_ref = float(y_top_cells.max())
    # the surface the cloth will actually be placed on (no holes yet): the
    # cuts are decided against *it*, not against a band around the hull
    draped_cells = (cells.kind == "drape").astype(np.float64)
    sig = params.hull_smooth_deg / 360.0 * params.n_theta
    if sig > 0:
        draped_cells = gaussian_filter1d(maximum_filter1d(draped_cells, size=2 * int(round(sig)) + 1, mode="wrap", axis=1),
                                         sigma=sig, mode="wrap", axis=1)
    surface = _Surface(scan, hull_cells, draped_cells, y_top_cells, y_hem, y_ref, size_y, axis_xz, params)
    arm_mask, slit_mask, hole_mask = hole_masks(scan, envelope, cells, hull_cells, y_top_cells, y_hem, y_ref, size_y,
                                                mesh, labels, params, torso_labels, sweep_vertices, obstacle_points,
                                                surface=surface)
    hole_labels, n_holes = _components_periodic(hole_mask)
    s_scale = float(np.nanmean(hull_cells[np.isfinite(hull_cells)]))
    holes = [_hole_field(hole_labels == k, scan.theta, scan.y, s_scale, smooth_cells=params.hole_smooth_cells)
             for k in range(1, n_holes + 1)]

    # ---- the grid: rows t in [0, 1] from the neckline to the hem, then cut
    n_cols = params.n_theta + 1                     # seam column duplicated for the UV chart
    theta_v = -np.pi + np.arange(n_cols) * (2 * np.pi / params.n_theta)
    rows, cols = params.n_rows, n_cols
    t = np.linspace(0.0, 1.0, rows)[:, None]
    TH = np.tile(theta_v[None, :], (rows, 1))
    Y = surface.y_top(theta_v)[None, :] + t * (surface.hem(theta_v)[None, :] - surface.y_top(theta_v)[None, :])
    F = fabric_signed_distance(surface, holes, TH, Y)          # (rows, cols)
    # cut along the edges and snap the outside vertices onto them, then
    # re-evaluate the surface at the snapped parameters
    faces, src_a, src_b, frac, n_snapped = cut_grid(F)
    TH_flat, Y_flat = TH.ravel(), Y.ravel()
    TH_flat = TH_flat[src_a] + frac * (TH_flat[src_b] - TH_flat[src_a])
    Y_flat = Y_flat[src_a] + frac * (Y_flat[src_b] - Y_flat[src_a])
    verts_all = surface.position(TH_flat, Y_flat)
    uvs_all = surface.uv(TH_flat, Y_flat)

    # drop faces the snapping collapsed, then unreferenced vertices
    tri = verts_all[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = faces[area2 > 1e-12]
    if len(faces) < 10:
        raise ValueError("build_kente_robe: the cuts left no cloth")

    # normals on the welded (periodic) grid so the seam column shades seamlessly
    last = cols - 1
    welded_faces = faces.copy()
    seam = (welded_faces % cols) == last
    welded_faces[seam] -= last
    welded = trimesh.Trimesh(vertices=verts_all, faces=welded_faces, process=False)
    normals_all = np.asarray(welded.vertex_normals, dtype=np.float64).copy().reshape(rows, cols, 3)
    normals_all[:, last] = normals_all[:, 0]
    normals_all = normals_all.reshape(-1, 3)
    radial = np.stack([np.sin(TH_flat), np.zeros_like(TH_flat), np.cos(TH_flat)], axis=1)
    used = np.unique(faces)
    if np.mean(np.sum(normals_all[used] * radial[used], axis=1)) < 0:
        faces = faces[:, ::-1]
        normals_all = -normals_all
    bad = np.linalg.norm(normals_all, axis=1) < 0.5
    normals_all[bad] = radial[bad]
    normals_all /= np.maximum(np.linalg.norm(normals_all, axis=1, keepdims=True), 1e-9)

    remap = -np.ones(rows * cols, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts, normals, uvs, faces = verts_all[used], normals_all[used], uvs_all[used], remap[faces]
    grid_index = used

    # ---- the weave, baked by position like the decals
    posmap = bake_position_map(verts, faces, uvs, size=tuple(params.texture_size), vertex_normals=normals)
    covered = posmap.position[posmap.mask].astype(np.float64)
    th = np.arctan2(covered[:, 0] - axis_xz[0], covered[:, 2] - axis_xz[1])
    px_per_radian = weave_params.strip_px * params.strips_around / (2 * np.pi)
    px_per_height = weave_params.block_px * params.blocks_tall / max(y_ref - y_hem, 1e-9)
    px = th * px_per_radian
    py = (y_ref - covered[:, 1]) * px_per_height           # blocks count down from the neckline
    t_dir = np.stack([np.cos(th), np.zeros_like(th), -np.sin(th)], axis=1)
    img, normal_img = bake_weave_maps(posmap, px, py, t_dir, np.array([0.0, -1.0, 0.0]), weave_params,
                                      verts, faces, uvs, bake_normal=params.bake_normal,
                                      bump_strength=params.bump_strength, dilate=4)
    if params.edge_band > 0:
        # a plain binding along every edge: neckline, armhole, slit, hem
        f_tex = fabric_signed_distance(surface, holes, th, covered[:, 1], include_hem=True)
        band = f_tex < params.edge_band
        arr = np.asarray(img).astype(np.float64) / 255.0
        color = np.array([*params.edge_band_color, 1.0])
        ys, xs = np.nonzero(posmap.mask)
        arr[ys[band], xs[band]] = color
        # a dark stitch line where the binding meets the weave
        stitch = (f_tex >= params.edge_band) & (f_tex < params.edge_band * 1.15)
        arr[ys[stitch], xs[stitch]] = np.array([*weave_params.colorway.motif, 1.0])
        img = to_image(arr)

    mat = MaterialSpec(name=params.material_name, base_color_image=img, image_format="PNG",
                       roughness=params.roughness, metallic=0.0, double_sided=True,
                       normal_image=normal_img)
    prim = PrimitiveSpec(name="kente_robe", vertices=verts, faces=faces, normals=normals, uvs=uvs, material=mat)
    y0 = bmin[1]
    info = {
        "faces": int(len(faces)), "vertices": int(len(verts)), "colorway": weave_params.colorway.name,
        "image": img.size, "axis_xz": [round(float(a), 4) for a in axis_xz],
        "hem_y": round(float(y_hem), 4),
        "neckline_frac_range": [round(float((y_top_cells.min() - y0) / size_y), 3), round(float((y_top_cells.max() - y0) / size_y), 3)],
        "hull_radius_range": [round(float(hull_cells.min()), 4), round(float(hull_cells.max()), 4)],
        "strips_around": params.strips_around, "pleats": params.pleats,
        "stop_cells": int(arm_mask.sum()), "slit_cells": int(slit_mask.sum()),
        "holes": [{"centre_theta_deg": round(float(np.degrees(h.centre[0])), 1),
                   "centre_y_frac": round(float((h.centre[1] - y0) / size_y), 3),
                   "theta_deg_range": [round(float(np.degrees(h.extent[0])), 1), round(float(np.degrees(h.extent[1])), 1)],
                   "y_frac_range": [round(float((h.extent[2] - y0) / size_y), 3), round(float((h.extent[3] - y0) / size_y), 3)]}
                  for h in holes],
        "obstacle_points": int(sum(len(np.asarray(o)) for o in (obstacle_points or []))),
        "snapped_vertices": int(n_snapped),
    }
    return KenteRobe(primitive=prim, grid_shape=(rows, cols), grid_index=grid_index, axis_xz=axis_xz,
                     y_top=y_top_cells, holes=holes, info=info)


# --------------------------------------------------------------------------
# skinning + measurement
# --------------------------------------------------------------------------

def transfer_body_weights(garment_vertices: np.ndarray, body_vertices: np.ndarray, pool_mask: np.ndarray,
                          body_weights: np.ndarray, k: int = 8,
                          allowed_joints: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dense skin weights for a garment from the `k` nearest body vertices
    in `pool_mask`, inverse-distance blended — the garment is not the body's
    own geometry (unlike a decal), so it has no `source_vertex_index` to
    copy from. `allowed_joints` (a boolean mask over joint columns) keeps
    the garment on the torso's joints only: body vertices next to a wing
    seam carry a share of that wing's weight, and a hem that inherits it
    follows the folded wing into the (static, mislabelled) wing tip below
    it on `tablet_show` — measured, 14 vertices. Returns (weights
    (n, n_joints), nearest pool vertex index)."""
    pool = np.where(np.asarray(pool_mask, dtype=bool))[0]
    if len(pool) == 0:
        raise ValueError("transfer_body_weights: empty pool")
    tree = cKDTree(np.asarray(body_vertices, dtype=np.float64)[pool])
    k = int(min(k, len(pool)))
    dist, idx = tree.query(np.asarray(garment_vertices, dtype=np.float64), k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    W = np.einsum("nk,nkj->nj", w, np.asarray(body_weights, dtype=np.float64)[pool[idx]])
    if allowed_joints is not None:
        allowed = np.asarray(allowed_joints, dtype=bool)
        if not allowed.any():
            raise ValueError("transfer_body_weights: no allowed joints")
        W[:, ~allowed] = 0.0
        empty = W.sum(axis=1) <= 1e-12
        if empty.any():
            # nothing allowed nearby: fall back to the allowed joint with the
            # most mass across the whole pool, rather than a zero row
            W[np.ix_(empty, [int(np.argmax(np.asarray(body_weights)[pool].sum(axis=0) * allowed))])] = 1.0
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    return W, pool[idx[:, 0]]


def smooth_grid_weights(weights: np.ndarray, grid_index: np.ndarray, grid_shape: tuple[int, int],
                        sigma: float = 1.0, periodic: bool = True) -> np.ndarray:
    """Gaussian-smooth per-vertex skin weights over the (rows, cols) grid a
    garment was cut from, so that where the nearest body vertex flips from
    a wing to the chest between two neighbouring cloth vertices the cloth
    does not crease — weights transferred from the nearest skin are only
    as smooth as the skin's labels. `grid_index` is each vertex's
    row * cols + col; with `periodic` the last column is the seam copy of
    the first. Missing cells (cut away) do not pull on their neighbours.
    Rows are renormalised to sum to one."""
    W = np.asarray(weights, dtype=np.float64)
    rows, cols = grid_shape
    G = np.zeros((rows * cols, W.shape[1]))
    M = np.zeros(rows * cols)
    G[grid_index] = W
    M[grid_index] = 1.0
    G = G.reshape(rows, cols, -1)
    M = M.reshape(rows, cols)
    if periodic:
        G, M = G[:, :-1], M[:, :-1]
    mode = "wrap" if periodic else "nearest"
    GM = gaussian_filter1d(gaussian_filter1d(G * M[..., None], sigma, axis=0, mode="nearest"), sigma, axis=1, mode=mode)
    MM = gaussian_filter1d(gaussian_filter1d(M, sigma, axis=0, mode="nearest"), sigma, axis=1, mode=mode)
    out = np.where(MM[..., None] > 1e-9, GM / np.maximum(MM[..., None], 1e-9), G)
    if periodic:
        out = np.concatenate([out, out[:, :1]], axis=1)
    out = out.reshape(rows * cols, -1)[grid_index]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def garment_overlap(inner_vertices: np.ndarray, outer_vertices: np.ndarray, outer_faces: np.ndarray,
                    n_samples: int = 100_000, seed: int = 0) -> dict:
    """Whether one garment hides under another: for every vertex of the
    inner one, the signed distance to the outer garment's surface (its
    face normals point away from the body), positive = poking *outside*
    the outer cloth. Reports how many do and how deep the rest sit."""
    P = np.asarray(inner_vertices, dtype=np.float64).reshape(-1, 3)
    if len(P) == 0:
        return {"count": 0, "outside_count": 0, "max_outside": 0.0, "min_depth": 0.0, "median_depth": 0.0, "max_depth": 0.0}
    outer = trimesh.Trimesh(vertices=np.asarray(outer_vertices, dtype=np.float64),
                            faces=np.asarray(outer_faces), process=False)
    samples = sample_surface(outer, n_samples, seed=seed)
    S, N = samples.evaluate(outer.vertices, outer.faces)
    _, si = cKDTree(S).query(P, workers=-1)
    signed = np.sum((P - S[si]) * N[si], axis=1)
    return {"count": int(len(P)), "outside_count": int((signed > 0).sum()),
            "max_outside": float(max(signed.max(), 0.0)), "min_depth": float(-signed.max()),
            "median_depth": float(-np.median(signed)), "max_depth": float(-signed.min())}


class SurfaceSamples(NamedTuple):
    """Dense samples of a body's surface, kept as (face, barycentric) so the
    same samples can be re-evaluated on a posed copy of the body."""
    face: np.ndarray       # (n,) face index
    bary: np.ndarray       # (n, 3)

    def evaluate(self, vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(points, unit face normals) on the body given by `vertices`."""
        tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)[self.face]]      # (n, 3, 3)
        points = np.einsum("nk,nkj->nj", self.bary, tri)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        return points, normals


def sample_surface(body: trimesh.Trimesh, n: int = 200_000, seed: int = 0) -> SurfaceSamples:
    points, face = trimesh.sample.sample_surface(body, int(n), seed=seed)
    bary = trimesh.triangles.points_to_barycentric(body.triangles[face], points)
    return SurfaceSamples(face=np.asarray(face, dtype=np.int64), bary=np.asarray(bary, dtype=np.float64))


def clearance(garment_vertices: np.ndarray, body: trimesh.Trimesh,
              surface_tree: cKDTree | None = None, samples: SurfaceSamples | None = None,
              n_samples: int = 200_000) -> dict:
    """How much air the garment keeps. `min`/`p05`/`median` are distances
    to the nearest body surface sample (`surface_tree`, e.g. the pipeline's
    3 M hi-res surface samples; otherwise dense samples of `body` itself),
    `inside_count` the number of garment vertices that sit *behind* the
    surface: each is compared against the nearest dense surface sample and
    the outward normal of the face that sample lies on. Nearest *vertex*
    normals (the first version of this) are not a geometric test — on a
    coarse mesh the nearest vertex is a corner whose averaged normal points
    diagonally away, and a point hovering outside a face reads as inside."""
    P = np.asarray(garment_vertices, dtype=np.float64)
    samples = samples if samples is not None else sample_surface(body, n_samples)
    S, N = samples.evaluate(body.vertices, body.faces)
    stree = cKDTree(S)
    d_samples, si = stree.query(P, workers=-1)
    if surface_tree is not None:
        d, _ = surface_tree.query(P, workers=-1)
    else:
        d = d_samples
    behind = np.sum((P - S[si]) * N[si], axis=1) < 0
    # the pseudo-normal sign is exact at the closest surface point but a
    # nearest *sample* in a concave crease (the rotated leg against the
    # rump on the hop) can belong to a face that faces away from a point
    # that is well outside — confirm every suspect by ray parity, which
    # fails differently (at the mesh's few boundary edges), so a vertex
    # counts as inside only when both agree
    inside = behind.copy()
    if behind.any():
        inside[behind] = np.asarray(body.contains(P[behind]), dtype=bool)
    return {"min": float(d.min()), "p05": float(np.percentile(d, 5)), "median": float(np.median(d)),
            "inside_fraction": float(inside.mean()), "inside_count": int(inside.sum()),
            "suspect_count": int(behind.sum())}


def posed_overlap(inner_vertices: np.ndarray, inner_weights: np.ndarray, outer_vertices: np.ndarray,
                  outer_faces: np.ndarray, outer_weights: np.ndarray, joints: list[fk.Joint], joint_names: list[str],
                  clips: Sequence[AnimationSpec], samples_per_clip: int = 6, n_samples: int = 60_000) -> dict:
    """`garment_overlap` through every clip, both garments posed with their
    own shipped weights; keeps the reading with the most inner vertices
    outside the outer garment (then the shallowest)."""
    worst = None
    per_clip = {}
    for anim in clips:
        duration = max(float(np.max(t.times)) for t in anim.tracks) if anim.tracks else 0.0
        clip_worst = None
        for tm in np.linspace(0.0, duration, samples_per_clip):
            rot, tr = clip_pose_at(anim, tm)
            skin = fk.skin_matrices(joints, fk.world_matrices(joints, rotations=rot, translations=tr))
            inner = fk.pose_vertices(inner_vertices, inner_weights, joint_names, skin)
            outer = fk.pose_vertices(outer_vertices, outer_weights, joint_names, skin)
            o = garment_overlap(inner, outer, outer_faces, n_samples=n_samples)
            o["time"] = round(float(tm), 3)
            key = (o["outside_count"], -o["min_depth"])
            if clip_worst is None or key > (clip_worst["outside_count"], -clip_worst["min_depth"]):
                clip_worst = o
        per_clip[anim.name] = clip_worst
        if clip_worst and (worst is None or (clip_worst["outside_count"], -clip_worst["min_depth"]) > (worst["outside_count"], -worst["min_depth"])):
            worst = dict(clip_worst, clip=anim.name)
    return {"worst": worst, "per_clip": per_clip}


def clip_pose_at(anim: AnimationSpec, time: float) -> tuple[dict, dict]:
    """(rotations, translation offsets) per joint at `time`, nearest keyframe."""
    rotations, translations = {}, {}
    for track in anim.tracks:
        i = int(np.argmin(np.abs(np.asarray(track.times) - time)))
        if track.path == "rotation":
            rotations[track.joint] = np.asarray(track.values[i], dtype=np.float64)
        elif track.path == "translation":
            translations[track.joint] = np.asarray(track.values[i], dtype=np.float64)
    return rotations, translations


def posed_clearance(garment_vertices: np.ndarray, garment_weights: np.ndarray,
                    body: trimesh.Trimesh, body_weights: np.ndarray, joints: list[fk.Joint],
                    joint_names: list[str], clips: Sequence[AnimationSpec],
                    samples_per_clip: int = 6) -> dict:
    """`clearance` through every clip: pose the body and the garment with
    the same skin (LBS through meshforge.fk) at a few times each and keep
    the worst reading. Sampled by the body's own vertices, so a garment
    vertex that ends up inside the posed body is counted as inside."""
    worst = None
    per_clip = {}
    body_V = np.asarray(body.vertices, dtype=np.float64)
    samples = sample_surface(body, 150_000)   # sampled once; re-evaluated on every posed body
    for anim in clips:
        duration = max(float(np.max(t.times)) for t in anim.tracks) if anim.tracks else 0.0
        times = np.linspace(0.0, duration, samples_per_clip)
        clip_worst = None
        for tm in times:
            rot, tr = clip_pose_at(anim, tm)
            world = fk.world_matrices(joints, rotations=rot, translations=tr)
            skin = fk.skin_matrices(joints, world)
            posed_body = trimesh.Trimesh(vertices=fk.pose_vertices(body_V, body_weights, joint_names, skin),
                                         faces=body.faces, process=False)
            posed_garment = fk.pose_vertices(garment_vertices, garment_weights, joint_names, skin)
            c = clearance(posed_garment, posed_body, samples=samples)
            c["time"] = round(float(tm), 3)
            if clip_worst is None or c["min"] < clip_worst["min"]:
                clip_worst = c
        per_clip[anim.name] = clip_worst
        if clip_worst and (worst is None or clip_worst["min"] < worst["min"]):
            worst = dict(clip_worst, clip=anim.name)
    return {"worst": worst, "per_clip": per_clip}
