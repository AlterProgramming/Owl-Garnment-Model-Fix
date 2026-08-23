"""The kente wrap: cloth hung from a measured support loop and tied over
the raised wing.

Where the tunic hung from the neckline, the wrap hangs from a *diagonal*
loop: high at the raised wing's shoulder (the knot), low under the tablet
wing (the cloth passes under the arm), rising again across the back. Every
point of that loop is read off the body at build time — collar rim, wing
root ring, the fused wing's lower contour — so the garment is rebuilt,
not re-fitted, when the owl underneath changes.

  measure_support   the loop L(theta), its radius off the body, the knot
                    frame and the gather density.
  build_kente_wrap  sheet (a tube hung from the loop, gathered into the
                    knot), tail (a strip pinned behind the knot), knot
                    (meshforge.knot), baked onto the pattern chart.
  support_weights   skin weights from the support graph: what holds the
                    cloth up, never what happens to be nearest.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial import cKDTree

from meshforge import drape as drapemod
from meshforge import robe as robemod
from meshforge.drape import BodyMeasure, ClothParams, Collider, Pattern, RadialBound
from meshforge.regions import RegionMap
from meshforge.rigexport import MaterialSpec, PrimitiveSpec
from meshforge.textile import WeaveParams

TWO_PI = 2.0 * np.pi


@dataclasses.dataclass(frozen=True)
class WrapParams:
    """Fractions of the body height H unless stated. Tuned on the AI-CCORE owl."""
    knot_drop: float = 0.03           # the knot's top under the collar rim (the band turns with the head)
    gather_half_deg: float = 25.0     # half-width of the knot base arc
    gather_ratio: float = 1.2         # cloth per loop length inside the arc (1.3 threw a flap out at the shoulder)
    arm_margin: float = 0.025         # loop below the tablet wing's lower boundary (the pin at 0.012 sat in its lowest feathers)
    hem_frac: float = 0.145           # hem height (the pleated hem dips ~0.02 under its target)
    hem_ease: float = 0.03            # cloth over the measured perimeter at the hem
    hem_min_frac: float = 0.55        # the hem is never cut narrower than this share of the widest row (0.75 ruffled)
    loop_smooth_cols: float = 1.0     # gaussian sigma, scan columns
    root_band: float = 0.03           # wing vertices this close to the torso are the root ring
    sweep_band: float = 0.12          # ... and this close is the root band whose sweep the loop clears (0 = off)
    sweep_blend_deg: float = 15.0     # over how many degrees that clearance comes in outside the knot arc
    sweep_drop_max: float = 0.08      # the most the swept root may pull the loop down
    knot_size: float = 0.095          # bulge height (0.07 vanished at kiosk distance — measured on the guide turntable)
    knot_strands: int = 7
    tail_width_frac: float = 0.15     # 0.22 read as a bib panel hanging on the back
    tail_end_frac: float = 0.15       # where the tail ends
    tail_span_deg: float = 45.0       # angular width of the tail's pin arc (35 fluted the strip 2:1)
    tail_offset_deg: float = 5.0      # gap behind the root ring's back edge to the tail's pin arc
    tail_taper: float = 0.62
    tail_cut_frac: float = 0.18
    tail_curl_deg: float = 22.0
    tail_n_u: int = 16
    tail_n_v: int = 80
    tail_texture_size: tuple[int, int] = (1024, 512)
    tail_strips: int = 6
    image_format: str = "PNG"
    raised_wing: str = "wing_left"
    tablet_wing: str = "wing_right"
    material_name: str = "owl_kente_wrap"
    tail_material_name: str = "owl_kente_wrap_tail"
    taut_cut: bool = False       # cut each column to its taut path, not its drop (capped by ClothParams.taut_cap).
                                 # Measured on this owl: it buys 0.01-0.02 of stretch and costs the hem — the
                                 # surplus lands under the raised wing, where the wave then catches it (G3 16-56
                                 # depending on the cap, against 6 with the drop cut and the swept-root clearance)
    neck_bound: bool = True           # hold the cloth off where the collar band turns (head-follow)
    root_bound: bool = True           # ... and off both wings' fused roots through the clips


class Support(NamedTuple):
    theta: np.ndarray      # (n_theta,) scan angles
    y: np.ndarray          # (n_theta,) loop height, world
    r: np.ndarray          # (n_theta,) loop radius off the body, world
    gather: np.ndarray     # (n_theta,) cloth density, 1 outside the knot arc
    theta_k: float         # knot centre, radians
    y_k: float             # knot height, world
    arc: tuple[float, float]   # (theta_lo, theta_hi) of the knot base arc, radians, may straddle -pi
    frame: dict            # origin (3,), normal (3,), tangent (3,), up (3,)
    root_ring: np.ndarray  # (m, 3)
    low: np.ndarray        # (n_theta,) tablet wing lower boundary, world y; nan where there is no wing
    info: dict


def _wrap_angle(a):
    return (np.asarray(a, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def _angle_of(P: np.ndarray, axis_xz: np.ndarray) -> np.ndarray:
    return np.arctan2(P[:, 0] - axis_xz[0], P[:, 2] - axis_xz[1])


def root_mask(mesh: trimesh.Trimesh, labels: np.ndarray, wing: str, band: float) -> np.ndarray:
    """Which of the mesh's vertices are the wing's root: its own vertices
    within `band` of the torso. The ring is these; the swept contour of
    where the wing's *base* goes on a clip is these too — never the blade,
    whose reach is what turned the garment into a box when it was fed into
    the loop's stand-off."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    labels = np.asarray(labels)
    w = labels == wing
    if not w.any():
        raise ValueError(f"root_mask: no vertices labelled {wing!r}")
    torso = np.isin(labels, ["chest", "body", "neck"])
    d = cKDTree(V[torso]).query(V[w], workers=-1)[0]
    m = np.zeros(len(V), dtype=bool)
    idx = np.flatnonzero(w)
    keep = d <= band
    if keep.sum() < 8:
        keep = np.zeros(len(idx), dtype=bool)
        keep[np.argsort(d)[:max(8, int(0.1 * len(idx)))]] = True
    m[idx[keep]] = True
    return m


def root_ring(mesh: trimesh.Trimesh, labels: np.ndarray, wing: str, band: float) -> np.ndarray:
    """The wing's vertices within `band` of the torso: where it grows out
    of the body. Found by distance, not adjacency, so a wing that is a
    separate shell in the mesh (the synthetic fin) has a root too."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    labels = np.asarray(labels)
    w = labels == wing
    if not w.any():
        raise ValueError(f"root_ring: no vertices labelled {wing!r}")
    torso = np.isin(labels, ["chest", "body", "neck"])
    d = cKDTree(V[torso]).query(V[w], workers=-1)[0]
    ring = V[w][d <= band]
    if len(ring) < 8:
        ring = V[w][np.argsort(d)[:max(8, int(0.1 * w.sum()))]]
    return ring


def _lower_boundary(poses: Sequence[np.ndarray], labels: np.ndarray, wing: str, theta: np.ndarray,
                    axis_xz: np.ndarray, min_count: int = 5) -> np.ndarray:
    """Per scan angle, the 5th percentile height of the wing's vertices in
    that angular cell — the fused wing's lower contour — taken over every
    pose given and kept at its lowest: the cloth passes under where the
    wing *goes* (it dips behind the pivot when the tablet comes up), not
    only where it rests. nan where the wing is not."""
    w = np.asarray(labels) == wing if isinstance(wing, str) else np.asarray(wing, dtype=bool)
    out = np.full(len(theta), np.nan)
    if not w.any():
        return out
    edges = np.concatenate([[-np.pi], 0.5 * (theta[1:] + theta[:-1]), [np.pi]])
    for V in poses:
        V = np.asarray(V, dtype=np.float64)
        th = _angle_of(V[w], axis_xz)
        y = V[w][:, 1]
        cell = np.clip(np.searchsorted(edges, th) - 1, 0, len(theta) - 1)
        for i in range(len(theta)):
            m = cell == i
            if m.sum() >= min_count:
                out[i] = np.nanmin([out[i], np.percentile(y[m], 2)])
    return out


def _posed_radius(poses: Sequence[np.ndarray], mask: np.ndarray, theta: np.ndarray, y: np.ndarray,
                  axis_xz: np.ndarray, band: float) -> np.ndarray:
    """Per scan angle, the greatest radius any masked vertex of any pose
    reaches within `band` of the loop height there: what the pinned row
    must stand off through the clips."""
    edges = np.concatenate([[-np.pi], 0.5 * (theta[1:] + theta[:-1]), [np.pi]])
    out = np.zeros(len(theta))
    for V in poses:
        V = np.asarray(V, dtype=np.float64)[mask]
        th = _angle_of(V, axis_xz)
        cell = np.clip(np.searchsorted(edges, th) - 1, 0, len(theta) - 1)
        near = np.abs(V[:, 1] - y[cell]) <= band
        if near.any():
            r = np.hypot(V[near, 0] - axis_xz[0], V[near, 2] - axis_xz[1])
            np.maximum.at(out, cell[near], r)
    return out


def measure_support(mesh: trimesh.Trimesh, region_map: RegionMap, body: BodyMeasure,
                    params: WrapParams, sweep_vertices: Sequence[np.ndarray] | None = None,
                    held_out: np.ndarray | None = None) -> Support:
    """The loop the wrap hangs from, read off the body — and off where the
    body goes: `sweep_vertices` are posed copies of the mesh's vertices
    (the clips and the head-follow range), `held_out` marks what the cloth
    passes behind (hand, tablet) and so never stands off."""
    labels = np.asarray(region_map.labels)
    H = body.size_y
    theta = body.theta
    n = len(theta)
    poses = [np.asarray(mesh.vertices, dtype=np.float64), *(np.asarray(q, dtype=np.float64) for q in (sweep_vertices or []))]
    fused = np.ones(len(labels), dtype=bool) if held_out is None else ~np.asarray(held_out, dtype=bool)
    # the raised wing is never what the loop stands off: the cloth passes
    # *under* it, against the body, and lies on its fused root only where the
    # pins' push-out finds that root solid. Measured with the wing in: pins
    # at r 0.79 over a body at 0.30 — a shelf at the shoulder
    fused = fused & (labels != params.raised_wing) & (labels != params.raised_wing + "_tip")

    # --- the knot: on the front edge of the raised wing's root, under the rim
    ring = root_ring(mesh, labels, params.raised_wing, params.root_band * H)
    th_ring = _angle_of(ring, body.axis_xz)
    mean = float(np.angle(np.exp(1j * th_ring).mean()))
    half = float(np.percentile(np.abs(_wrap_angle(th_ring - mean)), 95))
    cands = [mean + half, mean - half]
    theta_k = float(_wrap_angle(min(cands, key=lambda a: abs(float(_wrap_angle(a))))))   # the end nearer the face
    rim = float(np.interp(theta_k, theta, body.neck_y, period=TWO_PI))
    y_k = min(rim - params.knot_drop * H, float(ring[:, 1].max()))
    gh = np.radians(params.gather_half_deg)

    # --- the tablet wing's lower contour, and where it starts and ends
    low = _lower_boundary(poses, labels, params.tablet_wing, theta, body.axis_xz)
    valid = np.isfinite(low)
    if not valid.any():
        raise ValueError(f"measure_support: no {params.tablet_wing!r} to pass the cloth under")
    d = (theta - theta_k) % TWO_PI                       # distance forward from the knot, [0, 2pi)
    i_a = int(np.flatnonzero(valid)[np.argmin(d[valid])])
    i_b = int(np.flatnonzero(valid)[np.argmax(d[valid])])
    d_a, d_b = float(d[i_a]), float(d[i_b])
    order = np.argsort(d[valid])
    low_f = np.interp(d, d[valid][order], low[valid][order])   # gaps filled along d
    y_a = float(low[i_a]) - params.arm_margin * H
    y_b = float(low[i_b]) - params.arm_margin * H

    # --- the loop
    y = np.empty(n)
    for i in range(n):
        di = d[i]
        if di <= gh or di >= TWO_PI - gh:
            y[i] = y_k
        elif di < d_a:
            y[i] = y_k + (y_a - y_k) * (di - gh) / max(d_a - gh, 1e-6)
        elif di <= d_b:
            y[i] = low_f[i] - params.arm_margin * H
        else:
            y[i] = y_b + (y_k - y_b) * (di - d_b) / max((TWO_PI - gh) - d_b, 1e-6)
    # --- and never up where the raised wing's own base *sweeps*. `wave`
    # turns wing_left through 22 deg of lift and 20 of swing, and the cloth
    # is skinned to the torso alone (the wings are forbidden supports), so
    # a loop drawn under the wing at rest is inside it at the peak. Only
    # the root band, and only outside the knot arc: the knot is tied *on*
    # that root and has to stay there.
    if params.sweep_band > 0:
        raised = root_mask(mesh, labels, params.raised_wing, params.sweep_band * H)
        swept = _lower_boundary(poses, labels, raised, theta, body.axis_xz)
        off_arc = np.abs(_wrap_angle(theta - theta_k))
        blend = np.clip((off_arc - gh) / max(np.radians(params.sweep_blend_deg), 1e-6), 0.0, 1.0)
        cap = np.where(np.isfinite(swept), swept - params.arm_margin * H, np.inf)
        cap = np.maximum(cap, y - params.sweep_drop_max * H)      # a contour cannot collapse the piece
        y = blend * np.minimum(y, cap) + (1.0 - blend) * y

    # never above the collar band's lower rim (it dips under the bib at the
    # front; a straight diagonal from the knot crossed it — measured, 56
    # vertices above the rim at theta -67..-1)
    y = np.minimum(y, body.neck_y - params.knot_drop * H)
    y = np.clip(y, body.y_hem + 0.05 * H, float(body.neck_y.max()))
    y = gaussian_filter1d(y, sigma=max(params.loop_smooth_cols, 1e-3), mode="wrap")
    y = np.minimum(y, body.neck_y - 0.5 * params.knot_drop * H)

    # --- its radius: the torso's own at that height (not `wrap_r`, which
    # merges the raised wing's base into the body), plus the solver's air
    r_body = np.array([np.interp(y[i], body.y, body.envelope[:, i]) for i in range(n)])
    finite = np.isfinite(r_body)
    r_body = np.where(finite, r_body, np.max(r_body[finite]) if finite.any() else 0.0)
    # no cloth closer to the axis than the widest thing in the cell *or its
    # neighbours*: a cell-averaged radius interpolated across a thin limb's
    # edge put pins inside the limb (measured on the synthetic fin)
    r_wide = maximum_filter1d(r_body, size=3, mode="wrap")
    # ... and than anything fused that passes the loop's height on a clip
    r_posed = _posed_radius(poses, fused, theta, y, body.axis_xz, band=0.02 * H)
    r_wide = np.maximum(r_wide, maximum_filter1d(r_posed, size=3, mode="wrap"))
    r = gaussian_filter1d(r_wide, sigma=1.0, mode="wrap") + 0.014
    r = np.maximum(r, r_wide)

    # --- gather density: the ratio inside the arc, smooth shoulders outside
    dd = np.abs(_wrap_angle(theta - theta_k))
    shoulder = np.radians(5.0)
    t = np.clip((dd - gh) / shoulder, 0.0, 1.0)
    gather = 1.0 + (params.gather_ratio - 1.0) * (1.0 - t * t * (3 - 2 * t))

    # --- the knot frame: surface point, its normal, the loop's tangent
    i_k = int(np.argmin(dd))
    origin = np.array([body.axis_xz[0] + r[i_k] * np.sin(theta_k), y_k, body.axis_xz[1] + r[i_k] * np.cos(theta_k)])
    _, _, fid = mesh.nearest.on_surface(origin[None, :])
    normal = np.asarray(mesh.face_normals[int(fid[0])], dtype=np.float64).copy()
    radial = np.array([np.sin(theta_k), 0.0, np.cos(theta_k)])
    if np.dot(normal, radial) < 0:
        normal = -normal
    normal /= max(np.linalg.norm(normal), 1e-12)
    tangent = np.array([np.cos(theta_k), 0.0, -np.sin(theta_k)])     # along the loop, +theta direction
    tangent -= normal * np.dot(tangent, normal)
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    up = np.cross(normal, tangent)
    if up[1] < 0:
        up, tangent = -up, -tangent
    frame = {"origin": origin, "normal": normal, "tangent": tangent, "up": up}

    arc = (float(_wrap_angle(theta_k - gh)), float(_wrap_angle(theta_k + gh)))
    info = {
        "theta_k_deg": round(np.degrees(theta_k), 1), "y_k_frac": round((y_k - body.y0) / H, 3),
        "rim_frac_at_knot": round((rim - body.y0) / H, 3),
        "theta_a_deg": round(float(np.degrees(theta[i_a])), 1), "theta_b_deg": round(float(np.degrees(theta[i_b])), 1),
        "y_a_frac": round((y_a - body.y0) / H, 3), "y_b_frac": round((y_b - body.y0) / H, 3),
        "loop_frac": [round((y.min() - body.y0) / H, 3), round((y.max() - body.y0) / H, 3)],
        "root_ring_vertices": int(len(ring)),
        "root_ring_theta_deg": [round(float(np.degrees(mean - half)), 1), round(float(np.degrees(mean + half)), 1)],
        "axis_xz": [float(body.axis_xz[0]), float(body.axis_xz[1])],
    }
    return Support(theta=theta, y=y, r=r, gather=gather, theta_k=theta_k, y_k=y_k, arc=arc, frame=frame,
                   root_ring=ring, low=low, info=info)


def loop_at(support: Support, theta: np.ndarray, axis_xz: np.ndarray | None = None) -> np.ndarray:
    """World points on the loop at the given angles (periodic interpolation).
    `axis_xz` defaults to the one the loop was measured about."""
    theta = np.asarray(theta, dtype=np.float64)
    ax = np.asarray(support.info.get("axis_xz", [0.0, 0.0]) if axis_xz is None else axis_xz, dtype=np.float64)
    y = np.interp(theta, support.theta, support.y, period=TWO_PI)
    r = np.interp(theta, support.theta, support.r, period=TWO_PI)
    return np.stack([ax[0] + r * np.sin(theta), y, ax[1] + r * np.cos(theta)], axis=1)


# --------------------------------------------------------------------------
# the garment
# --------------------------------------------------------------------------

class KenteWrap(NamedTuple):
    sheet: PrimitiveSpec
    tail: PrimitiveSpec
    knot: PrimitiveSpec
    sheet_grid: tuple[int, int]
    sheet_pattern: Pattern
    sheet_sets: list
    tail_grid: tuple[int, int]
    tail_pattern: Pattern
    tail_sets: list
    support: Support
    body: BodyMeasure
    pins: dict
    info: dict


def grid_faces(rows: int, cols: int) -> np.ndarray:
    """Two triangles per cell of a (rows, cols) vertex grid, row-major."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).ravel()
    b, cc, d = a + 1, a + cols, a + cols + 1
    return np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, cc], axis=1)])


def _strip_pattern(width: float, length: float, rows: int, cols: int,
                   taper: float = 1.0, cut: float = 0.0) -> Pattern:
    phi_col = np.linspace(0.0, 1.0, cols)
    phi = np.tile(phi_col[None, :], (rows, 1))
    fall = length * (1.0 - cut * phi_col)
    v = np.linspace(0.0, 1.0, rows)[:, None] * fall[None, :]
    return Pattern(phi=phi, v=v, circumference=np.array([width, width * taper]), v_profile=np.array([0.0, length]),
                   length=length, fall=fall, grid_shape=(rows, cols), theta_cols=None)


def _face_out(P: np.ndarray, faces: np.ndarray, normals: np.ndarray, body: BodyMeasure):
    radial = P - np.array([body.axis_xz[0], 0.0, body.axis_xz[1]])
    radial[:, 1] = 0.0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    if float(np.mean(np.sum(normals * radial, axis=1))) < 0:
        return faces[:, ::-1], -normals
    return faces, normals


def build_kente_wrap(mesh: trimesh.Trimesh, region_map: RegionMap, weave_params: WeaveParams,
                     cloth_params: ClothParams = ClothParams(), params: WrapParams = WrapParams(),
                     band_frame=None, sweep_vertices: Sequence[np.ndarray] | None = None,
                     n_theta: int = 144) -> KenteWrap:
    """Measure the loop, hang the sheet from it (gathered into the knot),
    hang the tail behind the knot, place the knot, bake the weave."""
    for name, strips in (("strips_around", cloth_params.strips_around), ("tail_strips", params.tail_strips)):
        if strips % weave_params.strip_cycle != 0:
            raise ValueError(f"build_kente_wrap: {name} ({strips}) must be a multiple of the weave's strip_cycle "
                             f"({weave_params.strip_cycle})")
    labels = np.asarray(region_map.labels)
    cloth = dataclasses.replace(cloth_params, hem_frac=params.hem_frac, material_name=params.material_name,
                                hem_ease=params.hem_ease, hem_min_frac=params.hem_min_frac)
    body = drapemod.measure_body(mesh, region_map, cloth, band_frame=band_frame, n_theta=n_theta)
    protruding = drapemod.held_out_vertices(mesh, labels, body.scan, body.cells, body.axis_xz, cloth.hold_tolerance)
    support = measure_support(mesh, region_map, body, params, sweep_vertices=sweep_vertices, held_out=protruding)
    H = body.size_y
    hung = body._replace(neck_y=support.y, neck_r=support.r)       # the loop is the neckline now
    # floors through the clips: the legs and tail at the hem, the neck and
    # collar band (head-follow) above, and both wings' fused roots where
    # the cloth lies on them — the raised wing's root ring on the wave, the
    # tablet wing (less what is held out) on tablet_show. Only the hem's
    # swing bound sizes the *pattern*: `draft` cuts a whole row for the
    # widest thing any bound holds anywhere in it, and a root or collar
    # bound at one angle made the entire tube a box (measured)
    swing = drapemod._swing_bounds(mesh, labels, body, cloth, sweep_vertices, with_neck=False)
    bounds = drapemod._swing_bounds(mesh, labels, body, cloth, sweep_vertices, with_neck=params.neck_bound)
    pattern_bounds = list(swing)
    if sweep_vertices and params.root_bound:
        V0 = np.asarray(mesh.vertices, dtype=np.float64)
        ring_idx = cKDTree(V0).query(support.root_ring, workers=-1)[1]
        root_mask = np.zeros(len(V0), dtype=bool)
        root_mask[ring_idx] = True
        # not the tablet wing: it lifts *away* from the cloth's edge on
        # tablet_show, and a bound built from where it goes pushed the front
        # of the sheet out to the lifted forearm at rest (stretch 1.57)
        poses = [V0, *(np.asarray(q, dtype=np.float64) for q in sweep_vertices)]
        bounds.append(RadialBound([q[root_mask] for q in poses], body.axis_xz,
                                  y_lo=float(support.y.min()) - 0.05 * H, y_hi=float(support.y.max()) + 0.12 * H,
                                  margin=cloth.collide_offset))
        # the pattern knows the root at rest (the cloth lies on it); the
        # swept root is the solve's business only
        pattern_bounds.append(RadialBound([V0[root_mask]], body.axis_xz,
                                          y_lo=float(support.y.min()) - 0.05 * H, y_hi=float(support.y.max()) + 0.12 * H,
                                          margin=cloth.collide_offset))
    collider = Collider(mesh, face_mask=~protruding[mesh.faces].any(axis=1))   # the cloth passes behind what is held out
    full = Collider(mesh)                                                       # ... but ships outside all of it

    # --- the sheet: a tube hung from the loop, gathered into the knot
    def hang(pattern, cloth_p):
        """Pin the top row on the loop and let the piece fall."""
        rows, cols = pattern.grid_shape
        n = rows * cols
        keep = np.ones(n, dtype=bool)
        pinned = np.zeros(n, dtype=bool)
        pinned[:cols] = True
        P0 = drapemod.initial_positions(pattern, hung)
        # the pins come from the scan's cell-averaged radius; a column that
        # lands on a thin limb's edge starts inside it. Pins are on the body,
        # not in it: push them out of the whole mesh before anything hangs
        drapemod.push_out(P0, pinned, full, cloth_p.collide_offset)
        v_rows = pattern.v_profile
        sets = drapemod.constraint_sets(pattern.phi.ravel(), pattern.v.ravel(), pattern.circumference, v_rows,
                                        keep, rows, cols, cloth_p)
        tet = drapemod.build_tethers(pattern.phi.ravel(), pattern.v.ravel(), pattern.circumference, v_rows,
                                     keep, pinned, rows, cols)
        grade = cloth_p.collide_offset + (cloth_p.collide_offset_hem - cloth_p.collide_offset) * np.clip(
            pattern.v.ravel() / max(pattern.length, 1e-9), 0.0, 1.0) ** 1.4
        P, d = drapemod.drape(P0, pattern, sets, [collider], pinned, keep, cloth_p,
                              offset=grade, tethers=tet, bounds=bounds)
        return P, d, sets, keep, pinned

    pattern = drapemod.draft(hung, cloth, bounds=pattern_bounds, gather=support.gather, taut=params.taut_cut)
    rows, cols = pattern.grid_shape
    n = rows * cols
    P, d1, sets, keep, pinned = hang(pattern, cloth)
    d1["pushed_out"] = drapemod.push_out(P, keep, full, cloth.collide_offset * 0.6, rounds=12)
    d1["swept_clear"] = sum(b.apply(P, free=keep) for b in bounds)
    seam_src, seam_dup = drapemod.seam_columns(rows, cols, keep)
    P[seam_dup] = P[seam_src]
    faces = grid_faces(rows, cols)
    F = np.minimum(pattern.v, pattern.fall[None, :] - pattern.v)            # distance to the top edge or the hem
    uvs = np.stack([pattern.phi.ravel(), pattern.v.ravel() / max(pattern.length, 1e-9)], axis=1)
    uvs[np.arange(n) % cols == cols - 1, 0] = 1.0
    normals = drapemod._grid_normals(P, faces, np.arange(n), cols, n)
    faces, normals = _face_out(P, faces, normals, body)
    img, normal_img = drapemod._bake(P, faces, uvs, normals, pattern, F, weave_params, cloth, body)
    sheet_mat = MaterialSpec(name=params.material_name, base_color_image=img, image_format=params.image_format,
                             roughness=cloth.roughness, metallic=0.0, double_sided=True, normal_image=normal_img)
    sheet = PrimitiveSpec(name="kente_wrap", vertices=P, faces=faces, normals=normals, uvs=uvs, material=sheet_mat)

    # --- the tail: a strip pinned behind the knot, hanging down the back
    rows_t, cols_t = params.tail_n_v, params.tail_n_u
    width = params.tail_width_frac * H
    y_end = body.y0 + params.tail_end_frac * H
    length = max(support.y_k - y_end, 0.05 * H) * (1.0 + cloth.length_slack)
    tpat = _strip_pattern(width, length, rows_t, cols_t, taper=params.tail_taper, cut=params.tail_cut_frac)
    # pinned behind the root ring's back edge, so it hangs down the back
    # clear of the wing's base (pinned at the knot's back end it hung
    # through the base and the wave swept it)
    ring_back = np.radians(support.info["root_ring_theta_deg"][0])
    th_hi = min(support.arc[0], ring_back) - np.radians(params.tail_offset_deg)
    th_cols = th_hi - np.radians(params.tail_span_deg) * (1.0 - np.linspace(0.0, 1.0, cols_t))
    pins_t = loop_at(support, th_cols, body.axis_xz)
    pins_t[:, 1] = support.y_k
    radial = np.stack([np.sin(th_cols), np.zeros(cols_t), np.cos(th_cols)], axis=1)
    pins_t += radial * (0.01 * H)
    base_direction = radial * np.sin(np.radians(15.0)) + np.array([0.0, -1.0, 0.0]) * np.cos(np.radians(15.0))
    curl = np.radians(params.tail_curl_deg * (2.0 * tpat.phi[0] - 1.0))
    curl_cos, curl_sin = np.cos(curl), np.sin(curl)
    direction = np.stack([curl_cos * base_direction[:, 0] + curl_sin * base_direction[:, 2],
                          base_direction[:, 1],
                          -curl_sin * base_direction[:, 0] + curl_cos * base_direction[:, 2]], axis=1)
    P0t = (pins_t[None, :, :] + tpat.v[:, :, None] * direction[None, :, :]).reshape(-1, 3)
    nt = rows_t * cols_t
    keep_t = np.ones(nt, dtype=bool)
    pinned_t = np.zeros(nt, dtype=bool)
    pinned_t[:cols_t] = True
    sets_t = drapemod.constraint_sets(tpat.phi.ravel(), tpat.v.ravel(), tpat.circumference, tpat.v_profile,
                                      keep_t, rows_t, cols_t, cloth, periodic=False)
    tet_t = drapemod.build_tethers(tpat.phi.ravel(), tpat.v.ravel(), tpat.circumference, tpat.v_profile,
                                   keep_t, pinned_t, rows_t, cols_t, periodic=False)
    Pt, d2 = drapemod.drape(P0t, tpat, sets_t, [collider], pinned_t, keep_t, cloth,
                            offset=cloth.collide_offset_hem, tethers=tet_t, periodic=False, bounds=bounds)
    d2["pushed_out"] = drapemod.push_out(Pt, keep_t, full, cloth.collide_offset_hem * 0.6, rounds=12)
    faces_t = grid_faces(rows_t, cols_t)
    Ft = np.minimum(np.minimum(tpat.v, tpat.fall[None, :] - tpat.v), np.minimum(tpat.phi, 1.0 - tpat.phi) * width)
    uvs_t = np.stack([tpat.phi.ravel(), tpat.v.ravel() / length], axis=1)
    normals_t = drapemod._grid_normals(Pt, faces_t, np.arange(nt), cols_t, nt, periodic=False)
    faces_t, normals_t = _face_out(Pt, faces_t, normals_t, body)
    cloth_t = dataclasses.replace(cloth, texture_size=params.tail_texture_size, strips_around=params.tail_strips,
                                  material_name=params.tail_material_name)
    img_t, normal_t = drapemod._bake(Pt, faces_t, uvs_t, normals_t, tpat, Ft, weave_params, cloth_t, body)
    tail_mat = MaterialSpec(name=params.tail_material_name, base_color_image=img_t, image_format=params.image_format,
                            roughness=cloth.roughness, metallic=0.0, double_sided=True, normal_image=normal_t)
    tail = PrimitiveSpec(name="kente_wrap_tail", vertices=Pt, faces=faces_t, normals=normals_t, uvs=uvs_t, material=tail_mat)

    # --- the knot, on the base arc
    from meshforge.knot import build_knot
    lo, hi = support.arc
    arc_th = lo + ((hi - lo) % TWO_PI) * np.linspace(0.0, 1.0, 24)
    knot = build_knot(support.frame, loop_at(support, arc_th, body.axis_xz), size=params.knot_size * H,
                      material=sheet_mat, n_strands=params.knot_strands)

    hem = P[-cols:]
    info = {
        "support": support.info,
        "sheet": {"vertices": int(n), "faces": int(len(faces)), "grid": [rows, cols],
                  "pattern": {"circumference": [round(float(pattern.circumference[0]), 3), round(float(pattern.circumference[-1]), 3)],
                              "chest": round(float(pattern.circumference.max()), 3), "wrap": pattern.wrap_perimeter,
                              "profile": {"v": [round(float(x), 4) for x in pattern.v_profile],
                                          "circ": [round(float(x), 4) for x in pattern.circumference]},
                              "length": round(float(pattern.length), 3)},
                  "hem_frac_target": params.hem_frac,
                  "hem_frac_actual": [round(float((hem[:, 1].min() - body.y0) / H), 3), round(float((hem[:, 1].max() - body.y0) / H), 3)],
                  "width_by_height": drapemod._width_profile(P, body), "drape": d1, "image": img.size},
        "tail": {"vertices": int(nt), "faces": int(len(faces_t)), "grid": [rows_t, cols_t], "width": round(width, 3),
                 "length": round(float(length), 3), "pin_theta_deg": [round(float(np.degrees(th_cols[0])), 1), round(float(np.degrees(th_cols[-1])), 1)],
                 "end_frac_actual": round(float((Pt[:, 1].min() - body.y0) / H), 3), "drape": d2, "image": img_t.size},
        "knot": knot.info,
        "colorway": weave_params.colorway.name,
    }
    return KenteWrap(sheet=sheet, tail=tail, knot=knot.primitive, sheet_grid=(rows, cols), sheet_pattern=pattern,
                     sheet_sets=sets, tail_grid=(rows_t, cols_t), tail_pattern=tpat, tail_sets=sets_t,
                     support=support, body=body, pins={"sheet": P[:cols].copy(), "tail": Pt[:cols_t].copy()}, info=info)


# --------------------------------------------------------------------------
# skinning from the support graph
# --------------------------------------------------------------------------

def support_weights(verts: np.ndarray, grid_index: np.ndarray, grid_shape: tuple[int, int], pins: np.ndarray,
                    body_V: np.ndarray, body_W: np.ndarray, joint_names: Sequence[str], y_hip: float,
                    pin_weights: np.ndarray | None = None, periodic: bool = True, sigma: float = 1.0) -> np.ndarray:
    """Skin weights for a hung piece of cloth from what holds it up.

    The pinned row takes the body's own skin at its pin point, restricted to
    the torso joints (`chest`, `body`) and renormalised — the loop moves
    with the torso that carries it, never with the wing it passes under.
    Every column blends from its pin's weights to `body` by height, fully
    `body` below the hip: the skirt rides the root translation (the hop)
    and nothing else. No joint the cloth is not supported by gets any
    weight, whatever is nearest — which is the class-F failure of nearest-
    vertex transfer (the research volume's wrong animated baseline)."""
    joint_names = list(joint_names)
    jb, jc = joint_names.index("body"), joint_names.index("chest")
    rows, cols = grid_shape
    col = np.asarray(grid_index) % cols
    if pin_weights is None:
        near = cKDTree(np.asarray(body_V)).query(np.asarray(pins), workers=-1)[1]
        Wp = np.zeros((cols, len(joint_names)))
        Wp[:, jb] = body_W[near, jb]
        Wp[:, jc] = body_W[near, jc]
        s = Wp.sum(axis=1)
        Wp[s < 1e-6, jc] = 1.0
        Wp /= Wp.sum(axis=1, keepdims=True)
    else:
        Wp = np.asarray(pin_weights, dtype=np.float64)
        if Wp.shape != (cols, len(joint_names)):
            raise ValueError(f"support_weights: pin_weights must be ({cols}, {len(joint_names)}), got {Wp.shape}")
    y_top = np.asarray(pins)[:, 1]
    s = np.clip((y_top[col] - verts[:, 1]) / np.maximum(y_top[col] - y_hip, 1e-6), 0.0, 1.0)
    W = (1.0 - s)[:, None] * Wp[col]
    W[:, jb] += s
    W = robemod.smooth_grid_weights(W, np.asarray(grid_index), grid_shape, sigma=sigma, periodic=periodic)
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)


def forbidden_mass(W: np.ndarray, joint_names: Sequence[str], allowed: Sequence[str] = ("body", "chest")) -> float:
    """Total weight on joints the garment is not supported by (gate G2)."""
    cols = [i for i, nm in enumerate(joint_names) if nm not in allowed]
    return float(np.asarray(W)[:, cols].sum()) if cols else 0.0
