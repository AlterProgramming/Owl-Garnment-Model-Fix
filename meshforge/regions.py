"""Semantic region map + skeleton for the AI-CCORE owl (OwlV1).

The rigging guide's Phase 2: a small amount of semantic information that
no distance-based skinner can infer — which vertices are "head", which
are "the wing resting against the flank", which are "the tablet". Rules
are geometric (fractions of the mesh bounding box, measured against the
real asset on calibrated grid renders, see front_grid.png /
side_grid.png in the session notes) with color hints where geometry is
ambiguous (the collar rims, the tassel, the cap), followed by a
connectivity clean-up so a rule never leaves stray islands inside another
region.

Names are viewer-perspective: `*_left` is mesh −X (screen-left when the
owl faces the camera), matching the existing eye joints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import trimesh

from meshforge.fk import Joint

OWL_JOINT_PARENTS: dict[str, str | None] = {
    "body": None,
    "chest": "body",
    "neck": "chest",
    "head": "neck",
    "cap": "head",
    "tassel": "cap",
    "eye_left": "head",
    "eye_right": "head",
    "lid_left": "head",
    "lid_right": "head",
    "wing_left": "chest",
    "wing_left_tip": "wing_left",   # the waving wing bends; see classify()
    "wing_right": "chest",
    "leg_left": "body",
    "leg_right": "body",
    "foot_left": "leg_left",
    "foot_right": "leg_right",
    "tail": "body",
}

# joints that own a body-mesh region (the eyes/lids are separate geometry)
BODY_REGION_JOINTS = [
    "body", "chest", "neck", "head", "cap", "tassel", "wing_left", "wing_left_tip",
    "wing_right", "leg_left", "leg_right", "foot_left", "foot_right", "tail",
]

# how many edge rings around a seam may blend, per region (guide §6: keep
# rigid things rigid, blend only at declared boundaries)
DEFAULT_BLEND_RINGS: dict[str, int] = {
    "body": 5, "chest": 5, "neck": 5, "head": 4, "cap": 2, "tassel": 2,
    # the waving wing is the one thing on this owl that should read as soft:
    # a wide band at the shoulder and a wider one at the elbow, so the bend
    # is spread over the wing instead of creasing at two hard seams
    "wing_left": 8, "wing_left_tip": 10, "wing_right": 3, "leg_left": 4, "leg_right": 4,
    "foot_left": 3, "foot_right": 3, "tail": 3,
}


@dataclass
class OwlRegionParams:
    """All thresholds are fractions of the mesh bounding box unless the
    name says `_abs` (mesh units). Defaults measured on owl-mascot /
    AI-CCORE Owl (same frame: 1.367 × 1.899 × 1.176)."""
    foot_top: float = 0.075
    leg_top: float = 0.215
    body_top: float = 0.30
    collar_lo: float = 0.38          # search window for the collar rims
    collar_hi: float = 0.58
    collar_fallback: tuple[float, float] = (0.41, 0.52)
    head_top: float = 0.905
    # raised wing (−X): piecewise x-limit so the crop stops at the wing and
    # not at the cheek above it — (y_from, y_to, x_max) rows
    wing_left_bands: tuple[tuple[float, float, float], ...] = ((0.32, 0.45, 0.31), (0.45, 0.56, 0.285), (0.56, 0.73, 0.25))
    # folded wing (+X): (y_from, y_to, x_min) rows; the shoulder top is narrower than the wing body
    wing_right_bands: tuple[tuple[float, float, float], ...] = ((0.25, 0.50, 0.72), (0.50, 0.575, 0.785))
    tablet_z_min: float = 0.84       # the held tablet: z fraction above this, x fraction above tablet_x_min
    tablet_x_min: float = 0.62
    tablet_y: tuple[float, float] = (0.25, 0.60)
    tail_z_max: float = 0.12
    tail_y_max: float = 0.22
    tassel_x_min: float = 0.80
    tassel_z_min: float = 0.55
    tassel_y: tuple[float, float] = (0.77, 0.94)
    dark_value: float = 0.30         # HSV value below which a vertex is "dark"
    red_sat: float = 0.45
    leg_split_x: float | None = None  # x fraction; None = k-means on the leg slice
    wing_tip_split: float = 0.46      # fraction along the left wing where the tip joint starts


class RegionMap(NamedTuple):
    labels: np.ndarray            # (n_verts,) str (object) joint name per vertex
    joints: list[Joint]           # full OwlV1 skeleton with measured pivots (eyes/lids pivots filled later)
    fractions: np.ndarray         # (n_verts, 3) bbox fractions, handy for downstream paint ops
    collar: dict                  # collar band description: {"theta": ..., "y_lo": ..., "y_hi": ..., "center": ..., "axis": ...}
    stats: dict


def color_classes(colors: np.ndarray, params: OwlRegionParams) -> dict[str, np.ndarray]:
    c = np.clip(np.asarray(colors, dtype=np.float64), 0, 1)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    mx = c.max(axis=1)
    mn = c.min(axis=1)
    d = np.maximum(mx - mn, 1e-6)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return {
        "red": (sat > params.red_sat) & (mx > 0.25) & ((hue < 22) | (hue > 340)),
        "orange": (sat > params.red_sat) & (mx > 0.5) & (hue >= 22) & (hue < 58),
        "cream": (sat < 0.32) & (mx > 0.6),
        "dark": mx < params.dark_value,
    }


def _largest_component_mask(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest face-adjacency component inside face_mask."""
    if not face_mask.any():
        return face_mask
    idx = np.where(face_mask)[0]
    sub = mesh.submesh([face_mask], append=True)
    labels = trimesh.graph.connected_component_labels(sub.face_adjacency, node_count=len(sub.faces))
    best = np.argmax(np.bincount(labels))
    out = np.zeros_like(face_mask)
    out[idx[labels == best]] = True
    return out


def _faces_to_vertex_mask(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> np.ndarray:
    vm = np.zeros(len(mesh.vertices), dtype=bool)
    vm[mesh.faces[face_mask].ravel()] = True
    return vm


def _vertex_to_face_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray, min_verts: int = 2) -> np.ndarray:
    return vertex_mask[mesh.faces].sum(axis=1) >= min_verts


def measure_collar(fractions: np.ndarray, classes: dict[str, np.ndarray], params: OwlRegionParams,
                   exclude: np.ndarray, n_bins: int = 36) -> dict:
    """Locate the collar band from its dark rims. Returns per-angle (around
    the vertical axis through the head centre) lower/upper rim heights as
    bbox fractions, interpolated over gaps, plus the axis used."""
    fy = fractions[:, 1]
    window = (fy > params.collar_lo) & (fy < params.collar_hi) & ~exclude
    dark = window & classes["dark"]
    # axis: vertical line through the centroid of the whole collar window
    centre = fractions[window][:, [0, 2]].mean(axis=0) if window.any() else np.array([0.5, 0.5])
    theta = np.arctan2(fractions[:, 0] - centre[0], fractions[:, 2] - centre[1])  # 0 = front (+z), +ve toward +x
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    which = np.digitize(theta, bins) - 1
    for b in range(n_bins):
        sel = dark & (which == b)
        if sel.sum() < 8:
            continue
        ys = fy[sel]
        # the two rims are the two densest height clusters: take robust low/high quantiles
        lo[b] = np.percentile(ys, 8)
        hi[b] = np.percentile(ys, 92)
    valid = ~np.isnan(lo)
    if valid.sum() < 4:
        lo[:] = params.collar_fallback[0]
        hi[:] = params.collar_fallback[1]
    else:
        centres = (bins[:-1] + bins[1:]) / 2
        # circular interpolation over missing bins
        for arr in (lo, hi):
            ok = ~np.isnan(arr)
            xs = np.concatenate([centres[ok] - 2 * np.pi, centres[ok], centres[ok] + 2 * np.pi])
            vs = np.concatenate([arr[ok], arr[ok], arr[ok]])
            arr[~ok] = np.interp(centres[~ok], xs, vs)
        # light circular smoothing
        for arr in (lo, hi):
            sm = arr.copy()
            for i in range(n_bins):
                sm[i] = (arr[(i - 1) % n_bins] + 2 * arr[i] + arr[(i + 1) % n_bins]) / 4
            arr[:] = sm
    return {"theta_bins": bins, "y_lo": lo, "y_hi": hi, "centre_xz": centre, "theta": theta}


def collar_bounds_at(collar: dict, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = collar["theta_bins"]
    n = len(bins) - 1
    centres = (bins[:-1] + bins[1:]) / 2
    xs = np.concatenate([centres - 2 * np.pi, centres, centres + 2 * np.pi])
    lo = np.interp(theta, xs, np.tile(collar["y_lo"], 3))
    hi = np.interp(theta, xs, np.tile(collar["y_hi"], 3))
    return lo, hi


def classify(mesh: trimesh.Trimesh, colors: np.ndarray, params: OwlRegionParams | None = None) -> RegionMap:
    """Label every vertex of `mesh` (a welded, closed owl mesh) with the
    joint that owns it. `colors` are (n_verts, 3) sRGB in [0, 1]."""
    params = params or OwlRegionParams()
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bmin, bmax = mesh.bounds
    size = bmax - bmin
    f = (V - bmin) / size
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    classes = color_classes(colors, params)
    n = len(V)

    # --- rigid appendages first (they override the horizontal slabs) ---
    tablet = (fz > params.tablet_z_min) & (fx > params.tablet_x_min) & (fy > params.tablet_y[0]) & (fy < params.tablet_y[1])
    wing_r = np.zeros(n, dtype=bool)
    for y0, y1, x_min in params.wing_right_bands:
        wing_r |= (fx > x_min) & (fy >= y0) & (fy < y1)
    wing_right_v = tablet | wing_r
    wing_right_f = _largest_component_mask(mesh, _vertex_to_face_mask(mesh, wing_right_v))
    wing_right_v = _faces_to_vertex_mask(mesh, wing_right_f)

    wing_l = np.zeros(n, dtype=bool)
    for y0, y1, x_max in params.wing_left_bands:
        wing_l |= (fx < x_max) & (fy >= y0) & (fy < y1)
    wing_left_f = _largest_component_mask(mesh, _vertex_to_face_mask(mesh, wing_l))
    wing_left_v = _faces_to_vertex_mask(mesh, wing_left_f)

    tail_v = (fz < params.tail_z_max) & (fy < params.tail_y_max) & ~wing_right_v
    tail_f = _largest_component_mask(mesh, _vertex_to_face_mask(mesh, tail_v))
    tail_v = _faces_to_vertex_mask(mesh, tail_f)

    tassel_v = (fx > params.tassel_x_min) & (fz > params.tassel_z_min) & (fy > params.tassel_y[0]) & (fy < params.tassel_y[1]) & classes["red"]
    if tassel_v.sum() > 20:
        tassel_f = _largest_component_mask(mesh, _vertex_to_face_mask(mesh, tassel_v, min_verts=3))
        tassel_v = _faces_to_vertex_mask(mesh, tassel_f)
    else:
        tassel_v[:] = False

    appendage = wing_right_v | wing_left_v | tail_v | tassel_v

    # --- collar band from its dark rims, then head / neck / chest by it ---
    collar = measure_collar(f, classes, params, exclude=appendage)
    y_lo, y_hi = collar_bounds_at(collar, collar["theta"])
    margin = 0.006
    neck_v = (fy >= y_lo - margin) & (fy <= y_hi + margin) & ~appendage
    head_v = (fy > y_hi + margin) & (fy < params.head_top) & ~appendage
    cap_v = (fy >= params.head_top) & ~appendage
    # dark mortar-board geometry can dip slightly below head_top at the brim
    cap_v |= (fy >= params.head_top - 0.03) & classes["dark"] & ~appendage & ~neck_v

    # --- lower body ---
    foot_v = (fy < params.foot_top) & ~appendage
    leg_v = (fy >= params.foot_top) & (fy < params.leg_top) & ~appendage
    # left/right split from the two orange feet (the only unambiguous
    # bilateral landmark down there — the lower back feathers skew any
    # whole-slice statistic toward +x)
    feet_orange = foot_v & classes["orange"]
    if params.leg_split_x is None and feet_orange.sum() > 20:
        xs = fx[feet_orange]
        c0, c1 = np.percentile(xs, 20), np.percentile(xs, 80)
        for _ in range(30):
            assign = np.abs(xs - c0) <= np.abs(xs - c1)
            if assign.any():
                c0 = xs[assign].mean()
            if (~assign).any():
                c1 = xs[~assign].mean()
        split = (c0 + c1) / 2
    else:
        split = params.leg_split_x if params.leg_split_x is not None else 0.44
    left_side = fx < split

    body_v = (fy >= params.leg_top) & (fy < params.body_top) & ~appendage & ~neck_v
    chest_v = (fy >= params.body_top) & (fy < y_lo - margin) & ~appendage & ~neck_v

    labels = np.full(n, "chest", dtype=object)  # default for anything unclaimed in the middle
    labels[body_v] = "body"
    labels[chest_v] = "chest"
    labels[neck_v] = "neck"
    labels[head_v] = "head"
    labels[cap_v] = "cap"
    labels[leg_v & left_side] = "leg_left"
    labels[leg_v & ~left_side] = "leg_right"
    labels[foot_v & left_side] = "foot_left"
    labels[foot_v & ~left_side] = "foot_right"
    labels[tail_v] = "tail"
    labels[wing_left_v] = "wing_left"
    labels[wing_right_v] = "wing_right"
    labels[tassel_v] = "tassel"

    # Split the waving wing into an inner and an outer joint. One rigid
    # rotation about the shoulder makes a 0.62-long wing on a 1.9-tall body
    # swing like a paddle; a second joint lets the tip trail the base, which
    # is what reads as a wave. Weight painting alone cannot do this — a
    # single joint moves every vertex it owns by the same rotation, however
    # softly the seam is blended.
    tip_v = _wing_tip_mask(mesh, labels, wing_left_v, params.wing_tip_split)
    labels[tip_v] = "wing_left_tip"
    # anything above the collar that wasn't claimed (e.g. collar margin gaps) -> head
    unclaimed_high = (labels == "chest") & (fy > y_hi)
    labels[unclaimed_high] = "head"

    labels = _clean_islands(mesh, labels)

    joints = build_skeleton(mesh, labels, f, collar, params)
    stats = {name: int((labels == name).sum()) for name in BODY_REGION_JOINTS}
    stats["leg_split_x_fraction"] = float(split)
    return RegionMap(labels=labels, joints=joints, fractions=f, collar=collar, stats=stats)


def _clean_islands(mesh: trimesh.Trimesh, labels: np.ndarray, min_faces: int = 40) -> np.ndarray:
    """Re-label small disconnected islands of a label to the label that
    surrounds them (by majority of neighbouring vertices)."""
    labels = labels.copy()
    faces = mesh.faces
    adj = mesh.face_adjacency
    for _ in range(3):
        changed = False
        for name in np.unique(labels):
            face_mask = (labels[faces] == name).all(axis=1)
            if not face_mask.any():
                continue
            idx = np.where(face_mask)[0]
            keep_adj = adj[face_mask[adj[:, 0]] & face_mask[adj[:, 1]]]
            remap = -np.ones(len(faces), dtype=np.int64)
            remap[idx] = np.arange(len(idx))
            comp = trimesh.graph.connected_component_labels(remap[keep_adj], node_count=len(idx))
            counts = np.bincount(comp)
            for c in np.where(counts < min_faces)[0]:
                island_faces = idx[comp == c]
                island_verts = np.unique(faces[island_faces])
                # neighbours via faces touching island verts
                touching = np.isin(faces, island_verts).any(axis=1)
                neigh_labels = labels[faces[touching]].ravel()
                neigh_labels = neigh_labels[neigh_labels != name]
                if len(neigh_labels) == 0:
                    continue
                vals, cnts = np.unique(neigh_labels, return_counts=True)
                labels[island_verts] = vals[np.argmax(cnts)]
                changed = True
        if not changed:
            break
    return labels



def _wing_tip_mask(mesh: trimesh.Trimesh, labels: np.ndarray, wing_v: np.ndarray,
                   split: float) -> np.ndarray:
    """Outer `1 - split` of a wing, measured along its own long axis from
    the shoulder seam. Falls back to an empty mask if the wing is too small
    or has no seam to measure from."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    out = np.zeros(len(V), dtype=bool)
    if wing_v.sum() < 50:
        return out
    seam = _seam_centroid(mesh, labels, "wing_left", {"chest", "body", "neck", "head"})
    P = V[wing_v]
    if seam is None:
        seam = P[np.argmin(np.linalg.norm(P - P.mean(axis=0), axis=1))]
    axis = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)[2][0]
    if np.dot(P.mean(axis=0) - seam, axis) < 0:
        axis = -axis
    t = (P - seam) @ axis
    lo, hi = np.percentile(t, 1), np.percentile(t, 99)
    frac = (t - lo) / max(hi - lo, 1e-9)
    out[np.where(wing_v)[0][frac > split]] = True
    return out


def _seam_centroid(mesh: trimesh.Trimesh, labels: np.ndarray, a: str, b_set: set[str]) -> np.ndarray | None:
    """Centroid of vertices of region a that share an edge with any region in b_set."""
    e = mesh.edges_unique
    la, lb = labels[e[:, 0]], labels[e[:, 1]]
    m1 = (la == a) & np.isin(lb, list(b_set))
    m2 = (lb == a) & np.isin(la, list(b_set))
    verts = np.concatenate([e[m1, 0], e[m2, 1]])
    if len(verts) == 0:
        return None
    return mesh.vertices[np.unique(verts)].mean(axis=0)


def build_skeleton(mesh: trimesh.Trimesh, labels: np.ndarray, f: np.ndarray, collar: dict,
                   params: OwlRegionParams) -> list[Joint]:
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bmin, bmax = mesh.bounds
    size = bmax - bmin

    def at(fx: float, fy: float, fz: float) -> np.ndarray:
        return bmin + size * np.array([fx, fy, fz])

    def region_centroid(name: str) -> np.ndarray:
        m = labels == name
        return V[m].mean(axis=0) if m.any() else at(0.5, 0.5, 0.5)

    cx, cz = collar["centre_xz"]
    collar_mid = float(np.nanmean((collar["y_lo"] + collar["y_hi"]) / 2))
    collar_top = float(np.nanmean(collar["y_hi"]))

    body_c = region_centroid("body")
    pivots: dict[str, np.ndarray] = {}
    pivots["body"] = np.array([body_c[0], at(0, params.leg_top, 0)[1], body_c[2]])
    pivots["chest"] = np.array([body_c[0], at(0, params.body_top, 0)[1], body_c[2]])
    pivots["neck"] = at(cx, collar_mid, cz)
    pivots["head"] = at(cx, collar_top, cz)
    head_c = region_centroid("head")
    pivots["cap"] = np.array([head_c[0], at(0, params.head_top, 0)[1], head_c[2]])
    tassel_seam = _seam_centroid(mesh, labels, "tassel", {"cap", "head"})
    if tassel_seam is None:
        tm = labels == "tassel"
        tassel_seam = V[tm][np.argmax(V[tm][:, 1])] if tm.any() else pivots["cap"] + np.array([0.4, 0.0, 0.2])
    pivots["tassel"] = tassel_seam

    for wing in ("wing_left", "wing_right"):
        seam = _seam_centroid(mesh, labels, wing, {"chest", "body", "neck", "head"})
        pivots[wing] = seam if seam is not None else region_centroid(wing)
    elbow = _seam_centroid(mesh, labels, "wing_left_tip", {"wing_left"})
    pivots["wing_left_tip"] = elbow if elbow is not None else region_centroid("wing_left_tip")
    for side in ("left", "right"):
        leg = f"leg_{side}"
        foot = f"foot_{side}"
        hip = _seam_centroid(mesh, labels, leg, {"body", "chest"})
        pivots[leg] = hip if hip is not None else region_centroid(leg)
        ankle = _seam_centroid(mesh, labels, foot, {leg})
        pivots[foot] = ankle if ankle is not None else region_centroid(foot)
    tail_seam = _seam_centroid(mesh, labels, "tail", {"body", "chest", "leg_left", "leg_right", "wing_right"})
    pivots["tail"] = tail_seam if tail_seam is not None else region_centroid("tail")

    # eyes/lids: placeholders on the head pivot; meshforge.eyes fills the real ones
    for name in ("eye_left", "eye_right", "lid_left", "lid_right"):
        pivots[name] = pivots["head"].copy()

    return [Joint(name=name, parent=parent, pivot=np.asarray(pivots[name], dtype=np.float64))
            for name, parent in OWL_JOINT_PARENTS.items()]


def with_pivots(joints: list[Joint], overrides: dict[str, np.ndarray]) -> list[Joint]:
    return [Joint(j.name, j.parent, np.asarray(overrides.get(j.name, j.pivot), dtype=np.float64)) for j in joints]


REGION_PALETTE = {
    "body": (0.62, 0.62, 0.62), "chest": (0.30, 0.35, 0.90), "neck": (0.92, 0.90, 0.20), "head": (0.20, 0.80, 0.80),
    "cap": (0.12, 0.12, 0.12), "tassel": (1.00, 0.55, 0.05), "wing_left": (0.90, 0.20, 0.20), "wing_right": (0.20, 0.90, 0.25),
    "tail": (0.60, 0.33, 0.10), "leg_left": (1.00, 0.60, 0.80), "leg_right": (0.60, 1.00, 0.80),
    "foot_left": (1.00, 0.20, 0.60), "foot_right": (0.20, 1.00, 0.60),
}


def label_colors(labels: np.ndarray) -> np.ndarray:
    return np.array([REGION_PALETTE.get(l, (1.0, 1.0, 1.0)) for l in labels], dtype=np.float32)
