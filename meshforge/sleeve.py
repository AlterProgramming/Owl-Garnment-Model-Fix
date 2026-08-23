"""A kente sleeve for the wing that waves: a tube built from the wing's own
cross-sections, wide where the wing leaves the body and snug at the cuff.

The tunic (`meshforge.robe`) hangs from the neckline and drapes over the
fused root of both wings; it cannot follow the part of the raised wing
that stands free of the body, and a sleeve cut from the tunic's (angle,
height) cylinder would not either — the wing points up and out at 60°,
nowhere near the tunic's axis. So the sleeve has its own axis: the wing's
principal axis from the shoulder pivot, and its own scan:

  wing_frame      the axis `u` (away from the pivot), the wing's chord
                  direction `w` and its flat-face normal `n`.
  separation      along `u`, where the wing stops touching the body: the
                  sleeve's root ring sits a little before that point, under
                  the tunic's cloth over the root, and its cuff
                  `length_frac` of the free wing beyond it.
  section_scan    at every station along `u`, slice the mesh, cast rays in
                  the section plane from the wing's own centroid and record
                  the wing's radius (`wing_r`) and the first thing beyond it
                  that is not the wing (`body_r`: the collar, the head).
  build_sleeve    radius = min(wing_r + clearance(s), body_r - margin),
                  clearance `clearance_root` under the tunic, widening to
                  `clearance_wide` just past the separation and tapering
                  to `clearance_cuff` at the cuff ("wide around the joint,
                  tighter toward the tip"); cells where the wing is fused to
                  the body are cut away (the tunic covers that side).
                  One rectangular UV chart, kente strips *along* the sleeve,
                  a gold cuff binding, the weave baked by position.

What the sleeve clears is measured with the robe's `clearance` /
`posed_clearance`, and whether its root really hides under the tunic with
`robe.garment_overlap`.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter
from scipy.spatial import cKDTree
from trimesh.intersections import mesh_plane

from meshforge.bake import bake_position_map
from meshforge.kente import bake_weave_maps
from meshforge.rigexport import MaterialSpec, PrimitiveSpec
from meshforge.robe import _circular_interp_nan, _majority_label, _smoothstep, cut_grid
from meshforge.textile import WeaveParams, to_image

WING_LABELS: tuple[str, ...] = ("wing_left", "wing_left_tip")


@dataclasses.dataclass(frozen=True)
class SleeveParams:
    """Lengths in mesh units. Defaults tuned on the AI-CCORE owl (raised
    wing 0.8 long from the shoulder pivot, free of the body past 0.22)."""
    n_phi: int = 48                     # columns around the wing
    station_spacing: float = 0.01       # rows along the wing
    length_frac: float = 0.72           # of the free wing the sleeve covers (the feathered tip shows past the cuff)
    root_overlap: float = 0.04          # how far the root ring reaches back under the tunic, before the separation
    separation_gap: float = 0.02        # the wing is free of the body from where it keeps this much air all the way out
    clearance_root: float = 0.03        # air at the root ring (it must stay *inside* the tunic's cloth over the root)
    clearance_wide: float = 0.06        # ... at the widest, `wide_at` past the separation
    wide_at: float = 0.06
    clearance_cuff: float = 0.03        # ... at the cuff (the wing tapers through the cuff ring, so the true air is ~2/3 of this, and the cloth's blended weights lag the skin's by a few % at the peak of the wave)
    min_clearance: float = 0.012        # never closer than this to the wing along the ray, whatever the body allows (~0.008 of true air where the ray is oblique to the wing's surface)
    body_margin: float = 0.04           # air kept from whatever is not the wing (collar, head), along the section ray; the head turns at runtime (±25° yaw moves the collar ~0.025 at the root ring)
    section_fuzz: float = 0.02          # crossings this close are one solid (feather edges)
    smooth_phi_deg: float = 15.0
    smooth_stations: float = 1.5
    centroid_smooth_stations: float = 2.0
    cuff_band: float = 0.03             # gold binding at the cuff; 0 = none
    band_color: tuple[float, float, float] = (0.80, 0.62, 0.05)
    strips_around: int = 6              # kente strips run along the sleeve; this many around
    blocks_long: float = 2.0            # weave blocks over the sleeve's length
    texture_size: tuple[int, int] = (1024, 256)   # (around, along)
    bake_normal: bool = False
    bump_strength: float = 0.15
    roughness: float = 0.88
    material_name: str = "owl_kente_sleeve"


class WingFrame(NamedTuple):
    pivot: np.ndarray      # (3,) the shoulder joint
    u: np.ndarray          # (3,) along the wing, away from the pivot
    w: np.ndarray          # (3,) across the wing (its chord)
    n: np.ndarray          # (3,) the wing's flat-face normal
    s_lo: float            # extent of the wing along u (1st / 99th percentile)
    s_hi: float

    def station_origin(self, s: float) -> np.ndarray:
        return self.pivot + float(s) * self.u


class LabelSamples(NamedTuple):
    """Surface samples split by label, so every measurement is of the
    *surface* — a coarse wing (the synthetic test fin is a box with eight
    corners) has almost no vertices, but just as much surface."""
    wing: np.ndarray       # (n, 3) points on faces whose majority label is a wing label
    other: np.ndarray      # (m, 3) points on every other face


def label_samples(mesh: trimesh.Trimesh, labels: np.ndarray, wing_labels: Sequence[str] = WING_LABELS,
                  n_samples: int = 150_000, seed: int = 0) -> LabelSamples:
    labels = np.asarray(labels)
    face_lab = _majority_label(labels[np.asarray(mesh.faces)])
    wing_faces = np.isin(face_lab, list(wing_labels))
    if wing_faces.sum() < 4:
        raise ValueError(f"label_samples: too few faces labelled {list(wing_labels)} ({int(wing_faces.sum())})")
    if (~wing_faces).sum() < 4:
        raise ValueError("label_samples: the mesh is all wing")
    points, fid = trimesh.sample.sample_surface(mesh, int(n_samples), seed=seed)
    points = np.asarray(points, dtype=np.float64)
    is_wing = wing_faces[np.asarray(fid)]
    if is_wing.sum() < 50:
        # the wing is a small part of the surface: sample it on its own
        sub = mesh.submesh([np.where(wing_faces)[0]], append=True)
        extra, _ = trimesh.sample.sample_surface(sub, max(2000, int(n_samples) // 10), seed=seed + 1)
        return LabelSamples(wing=np.asarray(extra, dtype=np.float64), other=points[~is_wing])
    return LabelSamples(wing=points[is_wing], other=points[~is_wing])


def wing_frame(samples: LabelSamples, pivot: np.ndarray) -> WingFrame:
    """Principal axes of the wing's surface: the long axis oriented away
    from the pivot, the thinnest axis as the face normal."""
    pivot = np.asarray(pivot, dtype=np.float64)
    P = np.asarray(samples.wing, dtype=np.float64)
    c = P.mean(axis=0)
    axes = np.linalg.svd(P - c, full_matrices=False)[2]
    u = axes[0] if np.dot(c - pivot, axes[0]) >= 0 else -axes[0]
    n = axes[2]
    w = np.cross(n, u)
    w /= max(np.linalg.norm(w), 1e-12)
    n = np.cross(u, w)
    s = (P - pivot) @ u
    return WingFrame(pivot=pivot, u=u, w=w, n=n, s_lo=float(np.percentile(s, 1)), s_hi=float(np.percentile(s, 99)))


def separation(samples: LabelSamples, frame: WingFrame, spacing: float, gap: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Where the wing leaves the body: the first station along `u` past
    which the wing's surface never comes within `gap` of a surface that is
    not the wing. Returns (s_sep, stations, min gap per station)."""
    d, _ = cKDTree(samples.other).query(samples.wing, workers=-1)
    s = (samples.wing - frame.pivot) @ frame.u
    stations = np.arange(frame.s_lo, frame.s_hi, spacing)
    gaps = np.full(len(stations), np.nan)
    for i, st in enumerate(stations):
        sel = np.abs(s - st) < spacing
        if sel.any():
            gaps[i] = d[sel].min()
    touching = np.where(np.nan_to_num(gaps, nan=np.inf) < gap)[0]
    if len(touching) == 0:
        s_sep = float(stations[0])
    elif touching[-1] + 1 >= len(stations):
        raise ValueError("separation: the wing never leaves the body")
    else:
        s_sep = float(stations[touching[-1] + 1])
    return s_sep, stations, gaps


class SectionScan(NamedTuple):
    s: np.ndarray          # (S,) stations along u
    phi: np.ndarray        # (K,) ray angles in the (w, n) plane, -pi..pi
    centre: np.ndarray     # (S, 2) the wing's centroid per station, (w, n) offsets from the axis line
    wing_r: np.ndarray     # (S, K) radius of the wing's surface along each ray from the centroid
    body_r: np.ndarray     # (S, K) radius of the first solid beyond the wing that is not the wing (inf if none)
    fused: np.ndarray      # (S, K) bool: the ray left the solid through a surface that is not the wing (wing_r/body_r nan)
    open: np.ndarray       # (S, K) bool: the ray crossed nothing at all (a cavity of a concave section; wing_r 0)


def dedupe_crossings(t: np.ndarray, labels: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Sorted crossings with coincident ones collapsed: a ray through a
    vertex of the section polyline hits both segments that share it, at
    the same distance, and the duplicate flips the inside/outside parity
    of everything beyond."""
    t = np.asarray(t, dtype=np.float64)
    labels = np.asarray(labels)
    if len(t) == 0:
        return t, labels
    order = np.argsort(t, kind="stable")
    t, labels = t[order], labels[order]
    keep = np.concatenate([[True], np.diff(t) > tol])
    return t[keep], labels[keep]


def _occupied_intervals(t: np.ndarray, is_wing: np.ndarray, fuzz: float) -> list[tuple[float, float, bool]]:
    """Crossings along a ray from a point -> solid intervals (start, end,
    exit surface is wing). An odd crossing count means the ray started
    inside a solid. Intervals closer than `fuzz` are merged; a merged
    interval counts as wing only if both parts did (a body solid a hair
    behind the wing is the body, not more wing)."""
    t = np.asarray(t, dtype=np.float64)
    is_wing = np.asarray(is_wing, dtype=bool)
    if len(t) == 0:
        return []
    if len(t) % 2 == 1:
        raw = [(0.0, float(t[0]), bool(is_wing[0]))]
        start = 1
    else:
        raw, start = [], 0
    for i in range(start, len(t) - 1, 2):
        raw.append((float(t[i]), float(t[i + 1]), bool(is_wing[i + 1])))
    merged: list[tuple[float, float, bool]] = []
    for a, b, lw in raw:
        if merged and a - merged[-1][1] < fuzz:
            pa, _, plw = merged[-1]
            merged[-1] = (pa, b, plw and lw)
        else:
            merged.append((a, b, lw))
    return merged


def section_scan(mesh: trimesh.Trimesh, labels: np.ndarray, frame: WingFrame, stations: np.ndarray,
                 n_phi: int, params: SleeveParams, wing_labels: Sequence[str] = WING_LABELS) -> SectionScan:
    labels = np.asarray(labels)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    wing_set = list(wing_labels)
    phi = -np.pi + (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    d = np.stack([np.cos(phi), np.sin(phi)], axis=1)

    # pass 1: slice, keep the segments, measure the wing's centroid per station
    slices = []
    centres = np.zeros((len(stations), 2))
    for i, s in enumerate(stations):
        origin = frame.station_origin(s)
        segs, fids = mesh_plane(mesh, frame.u, origin, return_faces=True, cached_dots=(V - origin) @ frame.u)
        if len(segs) == 0:
            raise ValueError(f"section_scan: nothing to slice at station {s:.3f}")
        seg_lab = _majority_label(labels[mesh.faces[fids]])
        is_wing = np.isin(seg_lab, wing_set)
        if not is_wing.any():
            raise ValueError(f"section_scan: no wing at station {s:.3f} (labels {sorted(set(seg_lab))})")
        a2 = np.stack([(segs[:, 0] - origin) @ frame.w, (segs[:, 0] - origin) @ frame.n], axis=1)
        b2 = np.stack([(segs[:, 1] - origin) @ frame.w, (segs[:, 1] - origin) @ frame.n], axis=1)
        length = np.linalg.norm(b2 - a2, axis=1)
        mid = 0.5 * (a2 + b2)
        wgt = length[is_wing] / max(length[is_wing].sum(), 1e-12)
        centres[i] = (mid[is_wing] * wgt[:, None]).sum(axis=0)
        slices.append((a2, b2, is_wing))
    if params.centroid_smooth_stations > 0 and len(stations) > 1:
        centres = gaussian_filter1d(centres, sigma=params.centroid_smooth_stations, axis=0, mode="nearest")

    # pass 2: rays from the smoothed centroid
    wing_r = np.zeros((len(stations), n_phi))
    body_r = np.full((len(stations), n_phi), np.inf)
    fused = np.zeros((len(stations), n_phi), dtype=bool)
    open_ = np.zeros((len(stations), n_phi), dtype=bool)
    for i, (a2, b2, is_wing) in enumerate(slices):
        p = a2 - centres[i]
        e = b2 - a2
        dxe = d[:, 0, None] * e[None, :, 1] - d[:, 1, None] * e[None, :, 0]
        pxe = p[:, 0] * e[:, 1] - p[:, 1] * e[:, 0]
        pxd = p[None, :, 0] * d[:, 1, None] - p[None, :, 1] * d[:, 0, None]
        ok = np.abs(dxe) > 1e-12
        safe = np.where(ok, dxe, 1.0)
        t = np.where(ok, pxe[None, :] / safe, -1.0)
        sfrac = np.where(ok, pxd / safe, -1.0)
        hit = ok & (sfrac >= 0.0) & (sfrac <= 1.0) & (t > 1e-9)
        for k in range(n_phi):
            cols = np.where(hit[k])[0]
            tt, lw = dedupe_crossings(t[k, cols], is_wing[cols])
            intervals = _occupied_intervals(tt, lw, params.section_fuzz)
            if not intervals:
                open_[i, k] = True                # nothing in this direction: a cavity of the section
                continue
            end, j = 0.0, 0
            while j < len(intervals) and intervals[j][2]:
                end = intervals[j][1]
                j += 1
            if j == 0:
                # the ray leaves the solid through a surface that is not the
                # wing: the wing is fused to the body here; no radius is
                # meaningful, the cell is cut
                fused[i, k] = True
                wing_r[i, k] = np.nan
                body_r[i, k] = np.nan
                continue
            wing_r[i, k] = end
            if j < len(intervals):
                body_r[i, k] = intervals[j][0]
    return SectionScan(s=np.asarray(stations, dtype=np.float64), phi=phi, centre=centres,
                       wing_r=wing_r, body_r=body_r, fused=fused, open=open_)


def clearance_schedule(s: np.ndarray, s_sep: float, s_cuff: float, params: SleeveParams) -> np.ndarray:
    """`clearance_root` under the tunic, `clearance_wide` `wide_at` past
    the separation, `clearance_cuff` at the cuff."""
    s = np.asarray(s, dtype=np.float64)
    if s_cuff <= s_sep + params.wide_at:
        # too short for the plateau: straight from the root air to the cuff air
        t = _smoothstep((s - s_sep) / max(s_cuff - s_sep, 1e-9))
        return params.clearance_root + (params.clearance_cuff - params.clearance_root) * t
    s_wide = s_sep + params.wide_at
    rise = _smoothstep((s - s_sep) / max(s_wide - s_sep, 1e-9))
    fall = _smoothstep((s - s_wide) / max(s_cuff - s_wide, 1e-9))
    c = params.clearance_root + (params.clearance_wide - params.clearance_root) * rise
    return np.where(s <= s_wide, c, params.clearance_wide + (params.clearance_cuff - params.clearance_wide) * fall)


class KenteSleeve(NamedTuple):
    primitive: PrimitiveSpec
    grid_shape: tuple[int, int]
    grid_index: np.ndarray
    frame: WingFrame
    scan: SectionScan
    info: dict


def build_sleeve(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    pivot: np.ndarray,
    weave_params: WeaveParams,
    params: SleeveParams = SleeveParams(),
    wing_labels: Sequence[str] = WING_LABELS,
) -> KenteSleeve:
    """Measure the wing, pick the root and the cuff, scan the sections,
    wrap the cloth, bake the weave. Every intermediate is in `info`."""
    if params.n_phi < 8:
        raise ValueError(f"build_sleeve: need n_phi >= 8, got {params.n_phi}")
    if params.station_spacing <= 0 or params.length_frac <= 0 or params.length_frac > 1:
        raise ValueError("build_sleeve: station_spacing must be positive and length_frac in (0, 1]")
    if min(params.texture_size) < 4:
        raise ValueError(f"build_sleeve: texture_size {params.texture_size} is too small")
    if params.strips_around % weave_params.strip_cycle != 0:
        raise ValueError(f"build_sleeve: strips_around ({params.strips_around}) must be a multiple of the "
                         f"weave's strip_cycle ({weave_params.strip_cycle}) so the weave meets itself at the seam")
    labels = np.asarray(labels)
    samples = label_samples(mesh, labels, wing_labels)
    frame = wing_frame(samples, pivot)
    s_sep, probe_stations, gaps = separation(samples, frame, params.station_spacing, params.separation_gap)
    s_cuff = s_sep + params.length_frac * (frame.s_hi - s_sep)
    s_root = max(s_sep - params.root_overlap, frame.s_lo)
    n_stations = max(2, int(round((s_cuff - s_root) / params.station_spacing)) + 1)
    stations = np.linspace(s_root, s_cuff, n_stations)
    scan = section_scan(mesh, labels, frame, stations, params.n_phi, params, wing_labels)

    # ---- the radius per cell, and what is cut
    c = clearance_schedule(scan.s, s_sep, s_cuff, params)[:, None]
    body_cap = np.minimum(scan.body_r, 10.0) - params.body_margin        # nan on fused cells
    floor = np.where(scan.open, 0.0, scan.wing_r + params.min_clearance)   # nan on fused cells
    F = body_cap - floor                                   # positive: room for cloth
    # a fused cell is cut outright (F far below zero, so the edge snaps to
    # the last cell with room rather than part-way into the body); a cell
    # with merely too little room keeps its real (small, negative) F and
    # the edge lands at the zero crossing. An open cell (the ray met
    # nothing) has no radius of its own and takes its neighbours'.
    F = np.where(scan.fused, -1.0, F)
    radius = np.where((F >= 0) & ~scan.open, np.minimum(scan.wing_r + c, body_cap), np.nan)
    filled = np.array([_circular_interp_nan(row) for row in radius])
    if np.isnan(filled).any():
        raise ValueError("build_sleeve: a whole station is fused to the body; move the root past the separation")
    sig_phi = params.smooth_phi_deg / 360.0 * params.n_phi
    if sig_phi > 0:
        grown = maximum_filter1d(filled, size=3, mode="wrap", axis=1)
        filled = gaussian_filter1d(grown, sigma=sig_phi, mode="wrap", axis=1)
    if params.smooth_stations > 0 and len(stations) > 1:
        filled = gaussian_filter1d(filled, sigma=params.smooth_stations, mode="nearest", axis=0)
    # clamp back between the wing and the body — only where there is cloth:
    # a cut cell's own bounds are meaningless (its ray left the solid
    # through the body's far side) and must not leak into the neighbours
    # it is interpolated with
    kept = F >= 0
    radius = np.where(kept, np.minimum(np.maximum(filled, floor), body_cap), filled)
    # a cut cell takes the smallest radius of the kept cells around it as
    # the seed the edge vertices interpolate toward; every vertex is then
    # clamped between the *interpolated* wing floor and body cap at its own
    # parameters (below), which is what keeps a snapped edge vertex out of
    # both the wing and the body at once
    with_inf = np.where(kept, radius, np.inf)
    min_kept = minimum_filter(with_inf, size=3, mode=("nearest", "wrap"))
    radius = np.where(kept, radius, np.where(np.isfinite(min_kept), min_kept, radius))
    if not np.isfinite(radius).all():
        raise ValueError("build_sleeve: the section scan left a cell without a radius")
    floor_grid = np.array([_circular_interp_nan(row) for row in np.where(scan.fused, np.nan, floor)])
    cap_grid = np.array([_circular_interp_nan(row) for row in np.where(scan.fused, np.nan, body_cap)])
    floor_grid = np.where(np.isfinite(floor_grid), floor_grid, 0.0)
    cap_grid = np.where(np.isfinite(cap_grid), cap_grid, 10.0)

    # ---- the grid: rows at the stations, columns at the scan's ray angles
    # (so every vertex sits exactly on a measured cell), seam duplicated
    rows, cols = len(stations), params.n_phi + 1
    phi_v = np.concatenate([scan.phi, [scan.phi[0] + 2 * np.pi]])
    phi_ext = np.concatenate([[scan.phi[-1] - 2 * np.pi], scan.phi, [scan.phi[0] + 2 * np.pi]])
    ext = lambda a: np.concatenate([a[:, -1:], a, a[:, :1]], axis=1)
    radius_at = RegularGridInterpolator((scan.s, phi_ext), ext(radius), bounds_error=False, fill_value=None)
    floor_at = RegularGridInterpolator((scan.s, phi_ext), ext(floor_grid), bounds_error=False, fill_value=None)
    cap_at = RegularGridInterpolator((scan.s, phi_ext), ext(cap_grid), bounds_error=False, fill_value=None)
    F_at = RegularGridInterpolator((scan.s, phi_ext), ext(F), bounds_error=False, fill_value=None)
    S = np.tile(scan.s[:, None], (1, cols))
    PH = np.tile(phi_v[None, :], (rows, 1))
    Fg = F_at(np.stack([S, PH], axis=-1))
    faces, src_a, src_b, frac, n_snapped = cut_grid(Fg)
    S_flat, PH_flat = S.ravel(), PH.ravel()
    S_flat = S_flat[src_a] + frac * (S_flat[src_b] - S_flat[src_a])
    PH_flat = PH_flat[src_a] + frac * (PH_flat[src_b] - PH_flat[src_a])

    def position(s, ph):
        s = np.asarray(s, dtype=np.float64)
        ph = np.asarray(ph, dtype=np.float64)
        q = np.stack([np.clip(s, scan.s[0], scan.s[-1]), ph], axis=-1)
        r = np.minimum(np.maximum(radius_at(q), floor_at(q)), cap_at(q))
        cw = np.interp(s, scan.s, scan.centre[:, 0])
        cn = np.interp(s, scan.s, scan.centre[:, 1])
        off_w = cw + r * np.cos(ph)
        off_n = cn + r * np.sin(ph)
        return frame.pivot + s[..., None] * frame.u + off_w[..., None] * frame.w + off_n[..., None] * frame.n

    verts_all = position(S_flat, PH_flat)
    uvs_all = np.stack([(PH_flat - phi_v[0]) / (2 * np.pi), (S_flat - s_root) / max(s_cuff - s_root, 1e-9)], axis=1)
    tri = verts_all[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = faces[area2 > 1e-12]
    if len(faces) < 10:
        raise ValueError("build_sleeve: the cuts left no cloth")

    last = cols - 1
    welded_faces = faces.copy()
    seam = (welded_faces % cols) == last
    welded_faces[seam] -= last
    welded = trimesh.Trimesh(vertices=verts_all, faces=welded_faces, process=False)
    normals_all = np.asarray(welded.vertex_normals, dtype=np.float64).copy().reshape(rows, cols, 3)
    normals_all[:, last] = normals_all[:, 0]
    normals_all = normals_all.reshape(-1, 3)
    radial = np.cos(PH_flat)[:, None] * frame.w + np.sin(PH_flat)[:, None] * frame.n
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

    # ---- the weave: strips along the wing, baked by position
    posmap = bake_position_map(verts, faces, uvs, size=tuple(params.texture_size), vertex_normals=normals)
    covered = posmap.position[posmap.mask].astype(np.float64)
    s_tex = (covered - frame.pivot) @ frame.u
    cw = np.interp(s_tex, scan.s, scan.centre[:, 0])
    cn = np.interp(s_tex, scan.s, scan.centre[:, 1])
    q = covered - frame.pivot - s_tex[:, None] * frame.u
    ph_tex = np.arctan2(q @ frame.n - cn, q @ frame.w - cw)
    px_per_radian = weave_params.strip_px * params.strips_around / (2 * np.pi)
    px_per_length = weave_params.block_px * params.blocks_long / max(s_cuff - s_root, 1e-9)
    px = ph_tex * px_per_radian
    py = (s_tex - s_root) * px_per_length
    t_dir = -np.sin(ph_tex)[:, None] * frame.w + np.cos(ph_tex)[:, None] * frame.n
    img, normal_img = bake_weave_maps(posmap, px, py, t_dir, frame.u, weave_params, verts, faces, uvs,
                                      bake_normal=params.bake_normal, bump_strength=params.bump_strength, dilate=4)
    if params.cuff_band > 0:
        f_edge = s_cuff - s_tex
        arr = np.asarray(img).astype(np.float64) / 255.0
        ys, xs = np.nonzero(posmap.mask)
        band = f_edge < params.cuff_band
        arr[ys[band], xs[band]] = np.array([*params.band_color, 1.0])
        stitch = (f_edge >= params.cuff_band) & (f_edge < params.cuff_band * 1.15)
        arr[ys[stitch], xs[stitch]] = np.array([*weave_params.colorway.motif, 1.0])
        img = to_image(arr)

    mat = MaterialSpec(name=params.material_name, base_color_image=img, image_format="PNG",
                       roughness=params.roughness, metallic=0.0, double_sided=True, normal_image=normal_img)
    prim = PrimitiveSpec(name="kente_sleeve", vertices=verts, faces=faces, normals=normals, uvs=uvs, material=mat)
    mean_r = np.array([radius[i, kept[i]].mean() if kept[i].any() else np.nan for i in range(rows)])
    i_sep = int(np.argmin(np.abs(scan.s - s_sep)))
    info = {
        "faces": int(len(faces)), "vertices": int(len(verts)), "colorway": weave_params.colorway.name,
        "image": img.size, "pivot": [round(float(v), 4) for v in frame.pivot],
        "axis": [round(float(v), 4) for v in frame.u],
        "wing_extent": [round(frame.s_lo, 4), round(frame.s_hi, 4)],
        "s_separation": round(s_sep, 4), "s_root": round(float(s_root), 4), "s_cuff": round(float(s_cuff), 4),
        "stations": int(rows), "cut_cells": int((F < 0).sum()), "fused_cells": int(scan.fused.sum()),
        "body_capped_cells": int(((scan.wing_r + c) > body_cap).sum()), "open_cells": int(scan.open.sum()),
        "mean_radius": {"root": round(float(mean_r[0]), 4), "separation": round(float(mean_r[i_sep]), 4),
                        "cuff": round(float(mean_r[-1]), 4)},
        "clearance": {"root": params.clearance_root, "wide": params.clearance_wide, "cuff": params.clearance_cuff},
        "snapped_vertices": int(n_snapped),
    }
    return KenteSleeve(primitive=prim, grid_shape=(rows, cols), grid_index=grid_index, frame=frame, scan=scan, info=info)
