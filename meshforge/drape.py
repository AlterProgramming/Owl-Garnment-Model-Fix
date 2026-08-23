"""Cloth that is cut and hung, not offset.

`meshforge.robe` builds the garment as a radial height field `r(θ, y)`:
the radius is the running maximum of the body's own radius from the top
edge down, plus a clearance. Two things follow from that parameterisation
and no choice of `RobeParams` removes either.

  * **It cannot have a shoulder.** A height field over a cylinder is
    single-valued in `r`; cloth lying roughly *horizontally* over the top
    of a shoulder is a vertical cliff in that map. The garment can only
    ever stop in a rim, so the sleeve butts against the body instead of
    growing out of it.
  * **It cannot come back in.** The running maximum inherits the widest
    thing each ray passes — on this owl the folded wing and the held-out
    tablet — and carries it to the hem. Measured on the last robe build:
    at the hem the body is 0.704 wide and the cloth 1.615, i.e. 0.456 of
    air per side on a body 1.899 tall. A lampshade.

This module builds the same garment the way a garment is actually built:

  measure_body      the torso's own radius per (angle, height) — the
                    obstacles excluded, not maximised over — plus the
                    neckline curve under the collar's lower rim.
  draft             a *flat pattern*: a closed piece whose circumference
                    is the neckline's at the top and the chest's plus ease
                    below it, with a length that includes real surplus.
                    Its (u, v) is the cloth's own grain, and stays the
                    garment's UV chart, so the weave runs along the warp
                    and bends wherever the cloth bends.
  drape             position-based dynamics: gravity, inextensible warp
                    and weft, bending resistance, and the body as a
                    collider. The pattern holds more cloth than the space
                    around the body needs, and the surplus has nowhere to
                    go but into folds — which is where drape comes from,
                    and why it cannot be faked with a displacement.
  cut_openings      where the raised wing and the held-out tablet came
                    through the cloth on the first pass, cut in *pattern*
                    space and drape again warm-started.

What it clears is measured with `robe.clearance` / `robe.posed_clearance`
exactly as before: a garment that hangs beautifully through the owl's
chest is still a bug.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree

from meshforge import robe as robemod
from meshforge.bake import bake_position_map
from meshforge.kente import bake_weave_maps
from meshforge.regions import RegionMap
from meshforge.rigexport import MaterialSpec, PrimitiveSpec
from meshforge.textile import WeaveParams, to_image

TORSO_LABELS: tuple[str, ...] = ("chest", "body", "neck", "leg_left", "leg_right", "foot_left", "foot_right")


@dataclasses.dataclass(frozen=True)
class ClothParams:
    """Lengths are in mesh units unless the name ends in `_frac` (a
    fraction of the body's bounding-box height). Tuned on the AI-CCORE owl
    (1.37 x 1.90 x 1.18) by rendering."""

    # ---- draft -----------------------------------------------------------
    hem_frac: float = 0.135        # where the hem should end up; the feet start at 0.077
    top_gap_frac: float = 0.005    # neckline, under the collar band's measured lower rim
    neck_offset: float = 0.012     # air between the neckline and the neck
    chest_ease: float = 0.08       # cloth circumference over the perimeter the cloth wraps (torso + fused limbs);
                                   # 0.20 on the old bare-torso girth was 35-75 % short, 0.20 on the real perimeter crumpled
    hem_ease: float = 0.04         # ... at the hem
    hem_min_frac: float = 0.75     # the hem is never cut narrower than this share of the widest row
    yoke_frac: float = 0.075       # fall over which the pattern opens from neck to chest girth
    length_slack: float = 0.02     # surplus in the fall: the cloth that becomes folds
    taut_cap: float = 0.15         # ... and at most this much more where the taut path is longer than the drop
    n_u: int = 152                 # columns around (a seam column is added)
    n_v: int = 84                  # rows from the neckline to the hem

    # ---- solve -----------------------------------------------------------
    iterations: int = 300
    inner_iterations: int = 4
    dt: float = 0.018
    gravity: float = -9.0
    damping: float = 0.86
    stretch: float = 0.92          # warp/weft stiffness (1 = inextensible)
    shear: float = 0.35
    bend: float = 0.25             # 0.035 let every column buckle at the grid's own wavelength (a corrugated front)
    bend_wide: float = 0.15        # the same over four cells: pleats two cells wide are still the grid's, not the cloth's
    collide_offset: float = 0.014  # air the solver keeps between cloth and body at the neckline
    collide_offset_hem: float = 0.022  # ... at the hem, where the cloth hangs free
    swing_margin: float = 0.018    # air the hem keeps from where the legs and tail *go*, over the clips
    neck_margin: float = 0.030     # ... and the neckline from where the neck turns (cursor follow, every frame)
    friction: float = 0.55         # tangential velocity kept on contact
    self_collide: float = 0.0      # radius for cloth-cloth separation; 0 = off
    settle_iterations: int = 60    # gravity-free settling pass at the end

    # ---- openings --------------------------------------------------------
    open_margin: float = 0.024     # margin around what came through the cloth
    open_reach: float = 0.045      # a limb this close to the cloth is coming through it and claims an opening
    open_smooth: float = 1.8       # rounding of an opening's edge, in pattern cells
    min_panel_width: float = 0.048   # no tongue of cloth narrower than this survives a cut (~2.5x the selvedge)
    min_panel_fraction: float = 0.04  # ... and no piece of cloth smaller than this share of the garment
    hold_tolerance: float = 0.025  # a wing surface this far beyond the fused body is held out, and gets an opening
    sleeve_covers: float = 0.12    # a held-out surface this close to a sleeve is the sleeve's job, not the opening's
    yoke_keep: float = 0.12        # cloth (in v) below the neckline that no opening may cut: the yoke is what holds
                                   # the garment up, and a cut that reaches the neckline leaves no ring to hang from
    shoulder_keep: float = 0.24    # ... and deeper over a sleeved limb's root, so the cloth lies on the sleeve's
                                   # root ring like a cap over a sleeve head instead of opening around it
    shoulder_margin: float = 0.06  # in phi, either side of the sleeve's own opening

    # ---- surface ---------------------------------------------------------
    edge_band: float = 0.018       # gold selvedge along every cut edge; 0 = none
    edge_band_color: tuple[float, float, float] = (0.80, 0.62, 0.05)
    strips_around: int = 24        # kente strips around the piece; a multiple of the weave's strip_cycle
    blocks_tall: float | None = None   # None: as many as keep the weave square
    texture_size: tuple[int, int] = (2048, 1024)
    bake_normal: bool = False
    bump_strength: float = 0.15
    roughness: float = 0.88
    material_name: str = "owl_kente_tunic"


# --------------------------------------------------------------------------
# the body: what the cloth measures itself against, and what stops it
# --------------------------------------------------------------------------

class BodyMeasure(NamedTuple):
    axis_xz: np.ndarray        # (2,) the vertical axis the girths are measured about
    theta: np.ndarray          # (n_theta,) angles, 0 = +Z (front)
    y: np.ndarray              # (n_rows,) scan heights
    envelope: np.ndarray       # (n_rows, n_theta) torso radius
    neck_y: np.ndarray         # (n_theta,) neckline height per angle
    neck_r: np.ndarray         # (n_theta,) neckline radius per angle
    wrap_r: np.ndarray         # (n_rows, n_theta) radius of everything the cloth falls over: torso + fused limbs
    y_hem: float
    size_y: float
    y0: float
    scan: object                # robe.RadialScan, kept for the held-out test
    cells: object               # robe.CellKinds


def measure_body(mesh: trimesh.Trimesh, region_map: RegionMap, params: ClothParams,
                 band_frame=None, n_theta: int = 144, n_rows: int = 80) -> BodyMeasure:
    """The torso's own envelope and the neckline it hangs from. The scan and
    the neckline are `robe`'s — they were the part of that module that was
    right: the argument is with what was built on top of them."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bmin, bmax = np.array(mesh.bounds)
    size = bmax - bmin
    size_y = float(size[1])
    labels = np.asarray(region_map.labels)
    torso_mask = np.isin(labels, ["chest", "body"])
    if torso_mask.sum() < 50:
        raise ValueError("measure_body: the region map has no torso to dress")
    axis_xz = np.array([0.5 * (bmin[0] + bmax[0]), V[torso_mask][:, 2].mean()])

    y_hem = bmin[1] + params.hem_frac * size_y
    theta = -np.pi + (np.arange(n_theta) + 0.5) * (2 * np.pi / n_theta)
    y_levels = np.linspace(y_hem - 0.03 * size_y, bmin[1] + 0.74 * size_y, n_rows)
    scan = robemod.radial_scan(mesh, labels, axis_xz, y_levels, n_theta)
    envelope = robemod.torso_envelope(scan, torso_labels=TORSO_LABELS, mirror=True)

    rp = robemod.RobeParams(top_gap_frac=params.top_gap_frac, n_theta=n_theta)
    cells = robemod.classify_cells(scan, envelope, rp, torso_labels=TORSO_LABELS)
    neck_y = robemod._neckline(scan, envelope, region_map, band_frame, bmin, size, rp)
    rows = np.clip(np.searchsorted(scan.y, neck_y) - 1, 0, len(scan.y) - 2)
    t = (neck_y - scan.y[rows]) / np.maximum(scan.y[rows + 1] - scan.y[rows], 1e-9)
    cols = np.arange(n_theta)
    neck_env = (1 - t) * envelope[rows, cols] + t * envelope[rows + 1, cols]
    # the envelope is mirrored from the -X half; where the +X half really is
    # wider (or a shoulder is fused on) the neckline must clear *that*, or
    # the ring the whole garment hangs from starts inside the body
    first = np.where(np.isfinite(cells.first_r), cells.first_r, -np.inf)
    real = np.maximum(first[rows, cols], first[np.minimum(rows + 1, len(scan.y) - 1), cols])
    neck_r = np.maximum(neck_env, real) + params.neck_offset
    neck_r = gaussian_filter1d(neck_r, sigma=max(n_theta / 36.0, 1.0), mode="wrap")
    # what the cloth actually has to go around: the torso *and* whatever is
    # fused to it — the folded wing pressed to the chest, the tail, the
    # raised wing's root. `drape_r` is the outer radius of the first solid
    # on each ray after `classify_cells` merged the fused shoulder into it;
    # held-out solids (hand, tablet) are not in it, the cloth passes behind
    # them. The torso envelope alone is mirrored from the -X half and knows
    # nothing of the folded wing: a pattern cut to it was 35-75 % short of
    # the body it wrapped (measured on the first tunic, every row).
    fused = np.where(np.isfinite(cells.drape_r), cells.drape_r, -np.inf)
    wrap_r = np.maximum(envelope, fused)
    return BodyMeasure(axis_xz=axis_xz, theta=scan.theta, y=scan.y, envelope=envelope,
                       neck_y=neck_y, neck_r=neck_r, wrap_r=wrap_r, y_hem=float(y_hem), size_y=size_y,
                       y0=float(bmin[1]), scan=scan, cells=cells)


def _perimeter(r: np.ndarray, theta: np.ndarray) -> float:
    """Length of the closed polygon at radius `r(theta)` about the axis —
    the circumference of a real cross-section, not 2*pi times a mean radius
    (which under-measures anything that is not a circle)."""
    r = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite(r)):
        r = _circ_fill(r)
    pts = np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1)
    return float(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum())


def _circ_fill(row: np.ndarray) -> np.ndarray:
    out = row.copy()
    bad = ~np.isfinite(out)
    if bad.all():
        return np.zeros_like(out)
    idx = np.arange(len(out))
    out[bad] = np.interp(idx[bad], idx[~bad], out[~bad], period=len(out))
    return out


class Collider:
    """The body as dense surface samples with their face normals. One
    nearest-sample query per solver step pushes any cloth vertex that has
    got within `offset` of the surface back out along that sample's normal.

    `poses` are posed copies of the *whole* mesh's vertices: the same
    samples are re-evaluated on each and thrown into the same tree, so the
    cloth settles clear of where a limb goes, not only where it rests. The
    hem that clears the legs at rest is otherwise kicked through on the hop
    — measured, 125 vertices."""

    def __init__(self, mesh: trimesh.Trimesh, face_mask: np.ndarray | None = None, n: int = 220_000,
                 seed: int = 0, poses: Sequence[np.ndarray] | None = None):
        V = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces)
        if face_mask is not None and not face_mask.all():
            faces = faces[face_mask]
        used = np.unique(faces)
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        sub_faces = remap[faces]
        sub = trimesh.Trimesh(vertices=V[used], faces=sub_faces, process=False)
        samples = robemod.sample_surface(sub, n=n, seed=seed)
        pts, nrm = [], []
        for W in [V, *(np.asarray(q, dtype=np.float64) for q in (poses or []))]:
            p_, n_ = samples.evaluate(W[used], sub_faces)
            pts.append(p_)
            nrm.append(n_)
        self.points = np.concatenate(pts)
        self.normals = np.concatenate(nrm)
        self.tree = cKDTree(self.points)

    def escape(self, P: np.ndarray, offset, free: np.ndarray, k: int = 8) -> tuple[int, np.ndarray]:
        """Like `resolve`, but the push direction is chosen from the `k`
        nearest samples rather than the single closest one. In a crease —
        the leg against the rump — the nearest sample's normal can point
        along the crease and the point never gets out; one of its
        neighbours faces the way out."""
        _, idx = self.tree.query(P, k=k, workers=-1)
        delta = P[:, None, :] - self.points[idx]
        sd = np.einsum("ikj,ikj->ik", delta, self.normals[idx])
        best = np.argmax(sd, axis=1)
        rows = np.arange(len(P))
        i = idx[rows, best]
        sd_best = sd[rows, best]
        off = offset if np.isscalar(offset) else np.asarray(offset)
        hit = (sd_best < off) & free
        if hit.any():
            need = (off - sd_best)[hit] if not np.isscalar(off) else (off - sd_best[hit])
            P[hit] += need[:, None] * self.normals[i[hit]]
        return int(hit.sum()), hit

    def resolve(self, P: np.ndarray, offset, free: np.ndarray) -> tuple[int, np.ndarray]:
        _, i = self.tree.query(P, workers=-1)
        delta = P - self.points[i]
        sd = np.einsum("ij,ij->i", delta, self.normals[i])
        off = offset if np.isscalar(offset) else np.asarray(offset)
        hit = (sd < off) & free
        if hit.any():
            P[hit] += ((off - sd)[hit] if not np.isscalar(off) else (off - sd[hit]))[:, None] * self.normals[i[hit]]
        return int(hit.sum()), hit


class RadialBound:
    """The floor a hem has to stand on: per (angle, height) cell, the
    greatest radius the swinging parts reach over a set of posed copies of
    the body.

    Colliding the cloth against a *union of posed meshes* does not work —
    a sample belonging to one pose sits inside another pose's shell, its
    normal points into the union, and the cloth shreds (measured: it did).
    A scalar radial bound has no such ambiguity: it says only "no cloth
    closer to the axis than this here", which is exactly what a hem needs
    to know about a leg that swings."""

    def __init__(self, points: Sequence[np.ndarray], axis_xz: np.ndarray, y_lo: float, y_hi: float,
                 n_theta: int = 72, n_rows: int = 40, smooth: float = 1.0, margin: float = 0.0):
        self.margin = float(margin)
        self.axis_xz = np.asarray(axis_xz, dtype=np.float64)
        self.theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
        self.y_edges = np.linspace(y_lo, y_hi, n_rows + 1)
        grid = np.zeros((n_rows, n_theta))
        for P in points:
            P = np.asarray(P, dtype=np.float64)
            dx, dz = P[:, 0] - self.axis_xz[0], P[:, 2] - self.axis_xz[1]
            r = np.hypot(dx, dz)
            th = np.arctan2(dx, dz)
            ri = np.clip(np.searchsorted(self.y_edges, P[:, 1]) - 1, 0, n_rows - 1)
            ci = np.clip(np.searchsorted(self.theta_edges, th) - 1, 0, n_theta - 1)
            np.maximum.at(grid, (ri, ci), r)
        if smooth > 0:
            grid = gaussian_filter(grid, sigma=smooth, mode=("nearest", "wrap"))
        self.grid = grid

    def apply(self, P: np.ndarray, margin: float | None = None, free: np.ndarray | None = None) -> int:
        margin = self.margin if margin is None else margin
        free = np.ones(len(P), dtype=bool) if free is None else free
        dx, dz = P[:, 0] - self.axis_xz[0], P[:, 2] - self.axis_xz[1]
        r = np.hypot(dx, dz)
        th = np.arctan2(dx, dz)
        n_rows, n_theta = self.grid.shape
        inside_y = (P[:, 1] >= self.y_edges[0]) & (P[:, 1] <= self.y_edges[-1])
        ri = np.clip(np.searchsorted(self.y_edges, P[:, 1]) - 1, 0, n_rows - 1)
        ci = np.clip(np.searchsorted(self.theta_edges, th) - 1, 0, n_theta - 1)
        need = self.grid[ri, ci] + margin
        hit = free & inside_y & (need > 0) & (r < need)
        if hit.any():
            scale = np.where(r[hit] > 1e-6, need[hit] / r[hit], 1.0)
            P[hit, 0] = self.axis_xz[0] + dx[hit] * scale
            P[hit, 2] = self.axis_xz[1] + dz[hit] * scale
        return int(hit.sum())


def held_out_vertices(mesh: trimesh.Trimesh, labels: np.ndarray, scan, cells, axis_xz: np.ndarray,
                      tolerance: float = 0.025) -> np.ndarray:
    """Body vertices the cloth must be cut *around* rather than fall over.

    The distinction is not distance from the torso — the folded wing's
    upper arm is 0.1 from the chest and the cloth still lies on it, because
    there is no air between them the cloth could pass through. It is
    whether the surface is *held out*: `robe.classify_cells` already walks
    every ray and reports the outermost radius of the solid fused to the
    body (`drape_r`) and the intervals of anything met after a gap
    (`held`). A vertex beyond the fused surface is held out, and the cloth
    goes behind it, through an opening cut in the pattern.

    That is why the raised wing gets an armhole, the hand and the tablet a
    slit, and the folded wing's shoulder neither."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    dx, dz = V[:, 0] - axis_xz[0], V[:, 2] - axis_xz[1]
    r = np.hypot(dx, dz)
    th = np.arctan2(dx, dz)
    row = np.clip(np.searchsorted(scan.y, V[:, 1]) - 1, 0, len(scan.y) - 1)
    dth = scan.theta[1] - scan.theta[0]
    col = np.clip(np.round((th - scan.theta[0]) / dth).astype(int) % len(scan.theta), 0, len(scan.theta) - 1)
    cover = np.where(np.isfinite(cells.first_r), cells.first_r, -np.inf)
    cover = np.maximum(cover, cells.drape_r)[row, col]
    limbs = np.isin(labels, ["wing_left", "wing_left_tip", "wing_right"])
    return limbs & (r > cover + tolerance)


# --------------------------------------------------------------------------
# the pattern
# --------------------------------------------------------------------------

class Pattern(NamedTuple):
    """A flat piece of cloth on a (rows, cols) grid, periodic in u — the
    last column repeats the first so the UV chart has a seam while the
    cloth does not. `phi` is the position around the piece in [0, 1],
    `v` the distance down it from the neckline in cloth units."""
    phi: np.ndarray            # (rows, cols) position around the piece, in [0, 1]
    v: np.ndarray              # (rows, cols) distance down the piece from the neckline, in cloth units
    circumference: np.ndarray  # (k,) cloth circumference sampled at `v_profile`
    v_profile: np.ndarray      # (k,) the v values `circumference` is sampled at
    length: float              # the longest column's fall
    fall: np.ndarray           # (cols,) each column's own cut length
    grid_shape: tuple[int, int]
    wrap_perimeter: dict = {}  # the measured perimeter the cloth wraps, per scan row, and what sized the piece
    theta_cols: np.ndarray | None = None   # (cols,) world angle each column hangs at; None = uniform, last = seam copy


def column_angles(theta: np.ndarray, gather: np.ndarray | None, cols: int,
                  seg: np.ndarray | None = None) -> np.ndarray:
    """World angle of each of `cols` pattern columns (the last is the seam
    copy of the first). Columns are spaced by *cloth*: `seg` is the length
    of the support ring in each scan cell (its 3D length — a ring that dives
    under an arm is longer there than its angle says, and a column spaced
    by angle alone would be pinned four times its rest length from its
    neighbour; measured on the wrap, ratio 3.9), and `gather` a relative
    density, 1 where the cloth hangs plain and more where it is gathered
    into a knot, so the columns crowd there and the surplus has to fold.
    This is where a wrap's folds come from: cloth, not noise. Uniform when
    neither is given."""
    if gather is None and seg is None:
        return -np.pi + 2 * np.pi * np.arange(cols) / (cols - 1)
    n = len(theta)
    g = np.ones(n) if gather is None else np.asarray(gather, dtype=np.float64)
    if g.shape != (n,) or np.any(g < 1.0 - 1e-9):
        raise ValueError("column_angles: gather must be one density >= 1 per scan angle")
    edges = np.concatenate([[-np.pi], 0.5 * (theta[1:] + theta[:-1]), [np.pi]])
    w = np.diff(edges) if seg is None else np.asarray(seg, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError("column_angles: seg must be one length per scan angle")
    cum = np.concatenate([[0.0], np.cumsum(w * g)])
    return np.interp(np.linspace(0.0, cum[-1], cols), cum, edges)


def taut_fall(y_scan: np.ndarray, r_need: np.ndarray, theta: np.ndarray, th: np.ndarray,
              top_y: np.ndarray, top_r: np.ndarray, y_hem: float, stations: int = 64):
    """Each column's cut length, and where its rows land.

    A column is not cut to the *drop* from its support to the hem but to
    the length of the taut string from the support point, over everything
    the cloth must clear at that angle, down to the hem — the path a
    hanging column actually takes. The two agree only when nothing bulges
    below the support. On the wrap they do not: the loop crosses the chest
    at 0.43 H and the belly is wider below it, so columns cut to the drop
    came out 0.555 long for a 0.70 path and the cloth hung *stretched*
    (measured: warp l/l0 1.4 under the knot, and a shell-smooth chest).

    Returns `(fall, path_y)`: the path length per column, and the height at
    `stations` equally spaced arc-lengths along it, so the caller can ask
    what a row at cloth-distance v is falling over."""
    cols = len(th)
    fall = np.empty(cols)
    path_y = np.empty((cols, stations))
    order = np.argsort(y_scan)
    ys = np.asarray(y_scan)[order]
    for c in range(cols):
        rc = np.array([np.interp(th[c], theta, r_need[i], period=2 * np.pi) for i in range(len(y_scan))])[order]
        m = (ys <= top_y[c]) & (ys >= y_hem)
        yy = np.concatenate([[top_y[c]], ys[m][::-1], [y_hem]])
        rr = np.concatenate([[top_r[c]], rc[m][::-1], [float(np.interp(y_hem, ys, rc))]])
        # hanging cloth never comes back in: below the widest thing it has
        # passed it falls straight, so the radius it keeps is the running
        # maximum from the support down (cutting to the body's own inward
        # curve below the belly made the piece long enough to pool on the
        # floor — measured, hem at -0.08 H)
        rr = np.maximum.accumulate(rr)
        # the taut string over the profile: the upper convex chain in
        # (-y, r), which is exactly the shortest path that stays outside
        pts = np.stack([-yy, rr], axis=1)
        hull = []
        for q in pts:
            while len(hull) >= 2:
                o, a = hull[-2], hull[-1]
                if (a[0] - o[0]) * (q[1] - o[1]) - (a[1] - o[1]) * (q[0] - o[0]) >= 0:
                    hull.pop()
                else:
                    break
            hull.append(q)
        hull = np.asarray(hull)
        seg = np.hypot(np.diff(hull[:, 0]), np.diff(hull[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        fall[c] = max(float(s[-1]), 1e-3)
        path_y[c] = np.interp(np.linspace(0.0, fall[c], stations), s, -hull[:, 0])
    return fall, path_y


def draft(body: BodyMeasure, params: ClothParams, bounds: Sequence["RadialBound"] = (),
          gather: np.ndarray | None = None, taut: bool = False) -> Pattern:
    """The pattern. Two shapings, both of which a real pattern has and the
    radial hull could not:

      * the piece **opens** from the neckline's own circumference to the
        chest girth plus ease over a short yoke, so the cloth is fitted
        where it is held and full where it hangs;
      * every column is cut to **its own fall** — the neckline dips at the
        front, under the bib, and a piece cut to one length there would
        reach the floor. Shaping the pattern is what makes a hem level.

    Surplus lives in the circumference, not the length: cloth hanging free
    below the widest point has nothing to absorb extra length, so a long
    cut is a long garment, while a wide cut is a folded one."""
    rows, cols = params.n_v, params.n_u + 1
    nk = np.stack([body.axis_xz[0] + body.neck_r * np.sin(body.theta),
                   body.neck_y,
                   body.axis_xz[1] + body.neck_r * np.cos(body.theta)], axis=1)
    seg = np.linalg.norm(np.diff(np.vstack([nk, nk[:1]]), axis=0), axis=1)        # (n_theta,) ring segment lengths
    g_mid = np.ones_like(seg) if gather is None else 0.5 * (np.asarray(gather) + np.roll(np.asarray(gather), -1))
    c_neck = float((seg * g_mid).sum())          # the cloth along the ring: its length, times the gather where there is one
    # the ring's length per scan *cell* (centred on the scan angle): half of
    # each neighbouring segment
    seg_cell = 0.5 * (seg + np.roll(seg, 1))

    # the girth the cloth must clear: the widest perimeter of what it falls
    # over between the neckline and the hem — the torso with the limbs fused
    # to it, measured as a real cross-section (`wrap_r`), not the widest
    # anything (the held-out tablet, which is the bug in robe.gravity_hull)
    # and not the bare torso (which left the first tunic 35-75 % short and
    # stretched over the folded wing like a skin)
    # what the cloth must go around at every height: the torso with the
    # limbs fused to it (`wrap_r`), and — where a radial bound says a leg,
    # the tail or a wing tip *swings* through on a clip — that too, since
    # the solve will hold the cloth out there regardless of the pattern
    per = np.empty(len(body.y))
    r_need = np.empty((len(body.y), len(body.theta)))
    for i, yy in enumerate(body.y):
        r = body.wrap_r[i].copy()
        for b in bounds:
            if b.y_edges[0] <= yy <= b.y_edges[-1]:
                ri = int(np.clip(np.searchsorted(b.y_edges, yy) - 1, 0, b.grid.shape[0] - 1))
                ci = np.clip(np.searchsorted(b.theta_edges, body.theta) - 1, 0, b.grid.shape[1] - 1)
                need = b.grid[ri, ci] + b.margin
                r = np.maximum(r, np.where(b.grid[ri, ci] > 0, need, -np.inf))
        r_need[i] = r
        per[i] = _perimeter(r, body.theta)

    phi = np.arange(cols) / params.n_u
    th = column_angles(body.theta, gather, cols, seg=seg_cell)
    neck_y = np.interp(th, body.theta, body.neck_y, period=2 * np.pi)
    if taut:
        neck_r = np.interp(th, body.theta, body.neck_r, period=2 * np.pi)
        path, path_y = taut_fall(body.y, r_need, body.theta, th, neck_y, neck_r, body.y_hem)
        # capped: the taut path is what the cloth *would* travel if it hugged
        # everything it passes, and where the support is high over a wide
        # body that is a fifth more than the drop. Cut to all of it and the
        # surplus hangs below the hem in exactly those columns (measured: a
        # hem at 0.02-0.06 H over anklets that start at 0.10). The cap keeps
        # the relief where it is small and honest and refuses the rest.
        drop = np.maximum(neck_y - body.y_hem, 1e-3)
        fall = np.minimum(path, drop * (1.0 + params.taut_cap)) * (1.0 + params.length_slack)
    else:
        path_y = None
        fall = np.maximum(neck_y - body.y_hem, 1e-3) * (1.0 + params.length_slack)
    t = np.linspace(0.0, 1.0, rows)[:, None]
    v = t * fall[None, :]

    v_prof = np.linspace(0.0, float(v.max()), 96)
    length = max(float(v.max()), 1e-6)
    # the piece's circumference follows the body it wraps, row by row: the
    # measured perimeter at that row's height, plus an ease that grades
    # from `chest_ease` at the top to `hem_ease` at the hem. A two-point
    # profile (chest, hem) was wrong both ways on this owl — cut to the
    # chest it hung as a pleated bell with twice the cloth the belly needs;
    # tapered to the hem it was 15-35 % short over the belly and stretched.
    # a row of the piece sits at a different height in every column (the
    # neckline dips at the front and rises at the back by 0.14 of the body
    # on this owl), so a row must be cut for the widest height any of its
    # columns reaches — mapped through the mean neckline instead, the back
    # columns sat at the widest row while the pattern was cut for the belly
    if path_y is not None:
        # with a taut cut, cloth-distance v is *along the path*, so the
        # height a row reaches is read off that path, not subtracted from
        # the support: on the wrap the two differ by up to a fifth of the fall
        sv = v_prof / (1.0 + params.length_slack)
        y_cols = np.stack([np.interp(sv, np.linspace(0.0, path[c], path_y.shape[1]), path_y[c])
                           for c in range(cols)], axis=1)                                  # (k, cols)
    else:
        y_cols = body.neck_y[None, :] - (v_prof / (1.0 + params.length_slack))[:, None]   # (k, n_theta)
    per_v = np.interp(y_cols, body.y, per).max(axis=1)
    ease = params.chest_ease + (params.hem_ease - params.chest_ease) * (v_prof / length)
    base = per_v * (1.0 + ease)
    i_max = int(np.argmax(base))
    c_max = float(base[i_max])
    # below the widest row the cloth can only come back in as fast as a side
    # seam would take it: a straight taper to the hem, never tighter than
    # the body (+ ease) there and never below `hem_min_frac` of the widest
    # row — the garment is a tunic, not a pencil skirt
    hem_c = max(float(base[-1]), params.hem_min_frac * c_max)
    taper = np.where(v_prof >= v_prof[i_max],
                     c_max + (hem_c - c_max) * (v_prof - v_prof[i_max]) / max(length - v_prof[i_max], 1e-6),
                     c_max)
    circ = np.maximum(base, np.minimum(taper, np.maximum.accumulate(np.maximum(base, c_neck))))
    circ[v_prof >= v_prof[i_max]] = np.maximum(base, taper)[v_prof >= v_prof[i_max]]
    circ = gaussian_filter1d(circ, sigma=2.0, mode="nearest")
    # the neckline row is the ring the cloth hangs from: its own circumference
    yoke = max(params.yoke_frac * body.size_y, 1e-6)
    ty = np.clip(v_prof / yoke, 0.0, 1.0)
    circ = c_neck + (circ - c_neck) * (ty * ty * (3.0 - 2.0 * ty))
    c_wrap, c_hem_wrap = float(per_v.max()), float(per_v[-1])
    return Pattern(phi=np.tile(phi[None, :], (rows, 1)), v=v, circumference=circ, v_profile=v_prof,
                   length=float(v.max()), fall=fall, grid_shape=(rows, cols), theta_cols=th,
                   wrap_perimeter={"y_frac": [round(float((yy - body.y0) / body.size_y), 3) for yy in body.y],
                                   "perimeter": [round(float(x), 3) for x in per],
                                   "chest": round(c_wrap, 3), "hem": round(c_hem_wrap, 3)})


def initial_positions(pattern: Pattern, body: BodyMeasure, flare_deg: float = 42.0) -> np.ndarray:
    """A cone hanging from the neckline, wide enough to clear everything.
    The solve starts here and falls inward; starting *inside* the body is
    the one initial condition a collider cannot recover from."""
    rows, cols = pattern.grid_shape
    th = pattern.theta_cols if pattern.theta_cols is not None else -np.pi + 2 * np.pi * pattern.phi[0]
    r0 = np.interp(th, body.theta, body.neck_r, period=2 * np.pi)
    y0 = np.interp(th, body.theta, body.neck_y, period=2 * np.pi)
    a = np.radians(flare_deg)
    radial = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1)
    down = np.array([0.0, -1.0, 0.0])
    direction = radial * np.sin(a) + down * np.cos(a)
    origin = np.stack([body.axis_xz[0] + r0 * np.sin(th), y0, body.axis_xz[1] + r0 * np.cos(th)], axis=1)
    return (origin[None, :, :] + pattern.v[:, :, None] * direction[None, :, :]).reshape(rows * cols, 3)


# --------------------------------------------------------------------------
# constraints and the solve
# --------------------------------------------------------------------------

class _Set(NamedTuple):
    i: np.ndarray
    j: np.ndarray
    rest: np.ndarray
    k: float


def _pairs(rows: int, cols: int, keep: np.ndarray, di: int, dj: int, periodic: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (r, c) -> (r + di, c + dj) that both survive `keep`,
    wrapping in the column direction across the seam: the cloth has
    `cols - 1` real columns and column `cols - 1` is the chart's copy of
    column 0, so a pair that runs off the right edge lands on column 0
    (and is kept coincident with its copy by `seam_columns`). Measured on
    the first tunic: without the wrap nothing joined the two columns and
    the "seam" gaped 0.15 at the hem — the garment was an open sheet."""
    n = cols - 1 if periodic else cols
    r, c = np.meshgrid(np.arange(rows), np.arange(n), indexing="ij")
    r2 = r + di
    c2 = (c + dj) % n if periodic else c + dj
    ok = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < n)
    a = (r * cols + c)[ok]
    b = (r2 * cols + c2)[ok]
    good = keep[a] & keep[b]
    return a[good], b[good]


def seam_columns(rows: int, cols: int, keep: np.ndarray, periodic: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(source, copy) indices of the seam: column 0 and its duplicate
    column `cols - 1`, for the rows where both survive the cut. The copy
    carries no constraints of its own; it is written from the source after
    every projection so the chart keeps its seam and the cloth has none.
    A strip (`periodic` False) has no seam."""
    if not periodic:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    src = np.arange(rows) * cols
    dup = src + cols - 1
    m = keep[src] & keep[dup]
    return src[m], dup[m]


def constraint_sets(phi: np.ndarray, v: np.ndarray, circ: np.ndarray, v_rows: np.ndarray,
                    keep: np.ndarray, rows: int, cols: int, params: ClothParams,
                    periodic: bool = True) -> list[_Set]:
    """Warp, weft, shear and bend constraints, with every rest length read
    from the two vertices' own *pattern* coordinates — so a vertex the cut
    snapped onto an opening's edge carries the right amount of cloth, and
    the piece's circumference profile is honoured without a special case."""
    def rest_of(a, b):
        dphi = _wrap(phi[b] - phi[a])
        dv = v[b] - v[a]
        c = np.interp(0.5 * (v[a] + v[b]), v_rows, circ)
        return np.hypot(dphi * c, dv)

    plan = [(0, 1, params.stretch), (1, 0, params.stretch),
            (1, 1, params.shear), (1, -1, params.shear),
            (0, 2, params.bend), (2, 0, params.bend),
            (0, 4, params.bend_wide), (4, 0, params.bend_wide)]
    sets: list[_Set] = []
    for di, dj, k in plan:
        a, b = _pairs(rows, cols, keep, di, dj, periodic)
        if len(a):
            sets.append(_Set(a, b, rest_of(a, b), k))
    return sets


class Tether(NamedTuple):
    """Long-range attachments (Kim et al. 2012): every vertex is kept within
    its own *pattern* distance of the neckline vertex it hangs from. Local
    warp constraints alone cannot hold a pinned sheet under gravity — with a
    Jacobi solve, tension propagates about one row per iteration, so an
    84-row skirt falls for twenty steps before it learns it is attached.
    Tethers are the standard fix, and they are exactly true here: no point
    of cloth can be further from its neckline than the cloth between them."""
    vertex: np.ndarray     # (m,) the hanging vertex
    anchor: np.ndarray     # (m,) the neckline vertex it hangs from
    limit: np.ndarray      # (m,) pattern distance between them


def build_tethers(phi: np.ndarray, v: np.ndarray, circ: np.ndarray, v_rows: np.ndarray,
                  keep: np.ndarray, pinned: np.ndarray, rows: int, cols: int, periodic: bool = True) -> Tether:
    idx = np.arange(rows * cols)
    anchors = np.nonzero(pinned & keep)[0]
    if not len(anchors):
        return Tether(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    # nearest anchor around the neckline, by arc length on the piece
    a_phi = phi[anchors]
    d = np.abs(_wrap(phi[:, None] - a_phi[None, :])) if periodic else np.abs(phi[:, None] - a_phi[None, :])
    near = np.argmin(d, axis=1)
    arc = d[idx, near] * circ[0]
    limit = np.hypot(arc, v) + 1e-6
    live = keep & ~pinned
    return Tether(vertex=idx[live], anchor=anchors[near[live]], limit=limit[live])


def apply_tethers(P: np.ndarray, t: Tether) -> None:
    if not len(t.vertex):
        return
    d = P[t.vertex] - P[t.anchor]
    L = np.linalg.norm(d, axis=1)
    over = L > t.limit
    if over.any():
        i = t.vertex[over]
        P[i] = P[t.anchor[over]] + d[over] * (t.limit[over] / L[over])[:, None]


def _project(P: np.ndarray, s: _Set, w: np.ndarray, acc: np.ndarray, cnt: np.ndarray) -> None:
    d = P[s.j] - P[s.i]
    L = np.linalg.norm(d, axis=1)
    good = L > 1e-9
    if not good.any():
        return
    i, j, L, d = s.i[good], s.j[good], L[good], d[good]
    rest = s.rest[good] if s.rest.ndim else s.rest
    wi, wj = w[i], w[j]
    ws = wi + wj
    ok = ws > 0
    if not ok.any():
        return
    scale = np.zeros(len(L))
    scale[ok] = s.k * (L[ok] - rest[ok]) / (L[ok] * ws[ok])
    ci = (wi * scale)[:, None] * d
    cj = -(wj * scale)[:, None] * d
    n = len(P)
    for idx, corr in ((i, ci), (j, cj)):
        for ax in range(3):
            acc[:, ax] += np.bincount(idx, weights=corr[:, ax], minlength=n)
        cnt += np.bincount(idx, minlength=n)


def drape(P: np.ndarray, pattern: Pattern, sets: Sequence[_Set], colliders: Sequence[Collider],
          pinned: np.ndarray, keep: np.ndarray, params: ClothParams, offset=None,
          tethers: Tether | None = None, iterations: int | None = None,
          bounds: Sequence["RadialBound"] = (), periodic: bool = True) -> tuple[np.ndarray, dict]:
    """Position-based dynamics. Gravity, then the constraint sets and the
    body projected in turn; velocity is read back from the corrected
    positions, which is what makes contact behave (a vertex the body just
    pushed out does not keep the velocity that drove it in)."""
    P = np.array(P, dtype=np.float64)
    n = len(P)
    w = np.where(pinned | ~keep, 0.0, 1.0)
    V = np.zeros_like(P)
    acc = np.empty_like(P)
    cnt = np.empty(n)
    iterations = params.iterations if iterations is None else iterations
    free = keep & ~pinned
    rows, cols = pattern.grid_shape
    seam_src, seam_dup = seam_columns(rows, cols, keep, periodic)
    P[seam_dup] = P[seam_src]
    hits = 0
    for step in range(iterations + params.settle_iterations):
        settling = step >= iterations
        g = 0.0 if settling else params.gravity
        V[:, 1] += g * params.dt
        V *= params.damping if not settling else 0.5
        V[~free] = 0.0
        Pn = P + V * params.dt
        Pn[~free] = P[~free]
        for _ in range(params.inner_iterations):
            acc[:] = 0.0
            cnt[:] = 0.0
            for cs in sets:
                _project(Pn, cs, w, acc, cnt)
            moved = cnt > 0
            Pn[moved] += acc[moved] / cnt[moved][:, None]
            Pn[~free] = P[~free]
            if tethers is not None:
                apply_tethers(Pn, tethers)
            Pn[seam_dup] = Pn[seam_src]
        off = params.collide_offset if offset is None else offset
        hits, hit_mask = 0, np.zeros(n, dtype=bool)
        for col in colliders:
            h, m = col.resolve(Pn, off, free)
            hits += h
            hit_mask |= m
        for b in bounds:
            b.apply(Pn, free=free)
        if params.self_collide > 0:
            _separate(Pn, free, params.self_collide)
        Pn[seam_dup] = Pn[seam_src]
        V = (Pn - P) / params.dt
        V[hit_mask] *= params.friction
        P = Pn
    return P, {"iterations": int(iterations + params.settle_iterations), "contacts": hits}


def push_out(P: np.ndarray, keep: np.ndarray, collider: Collider, offset: float,
             rounds: int = 6) -> int:
    """Move every kept vertex — the pinned neckline too — outside the body.
    The solve leaves a handful inside where the collider only ran between
    constraint passes, and a garment that ships with cloth inside the chest
    is a bug however well it hangs."""
    moved = 0
    for _ in range(rounds):
        hits, _ = collider.escape(P, offset, keep)
        moved = hits
        if not hits:
            break
    return moved


def _separate(P: np.ndarray, free: np.ndarray, radius: float) -> None:
    """One pass of cloth-cloth separation: any pair closer than `radius`
    that is not a near neighbour on the pattern is pushed apart."""
    tree = cKDTree(P)
    pairs = np.array(list(tree.query_pairs(radius)), dtype=np.int64)
    if not len(pairs):
        return
    i, j = pairs[:, 0], pairs[:, 1]
    d = P[j] - P[i]
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-9
    i, j, d, L = i[ok], j[ok], d[ok], L[ok]
    push = 0.5 * (radius - L) / L
    step = push[:, None] * d
    np.subtract.at(P, i, np.where(free[i][:, None], step, 0.0))
    np.add.at(P, j, np.where(free[j][:, None], step, 0.0))


# --------------------------------------------------------------------------
# openings, cut in pattern space
# --------------------------------------------------------------------------

def _swing_bounds(mesh: trimesh.Trimesh, labels: np.ndarray, body: BodyMeasure, params: ClothParams,
                  sweep_vertices: Sequence[np.ndarray] | None, with_neck: bool = True) -> list["RadialBound"]:
    """The floors the cloth must stand on through the clips: what sweeps
    through the height the hem hangs at (the legs and tail on the hop, the
    folded wing's lower arm when the tablet comes up), and where the neck
    goes at runtime (cursor follow) under a neckline that does not turn
    with it."""
    bounds: list[RadialBound] = []
    if not sweep_vertices:
        return bounds
    V = np.asarray(mesh.vertices, dtype=np.float64)
    poses = [V, *(np.asarray(q, dtype=np.float64) for q in sweep_vertices)]
    swing = np.isin(labels, ["leg_left", "leg_right", "foot_left", "foot_right", "tail",
                             "wing_left", "wing_right", "wing_left_tip"])
    if swing.any():
        bounds.append(RadialBound([q[swing] for q in poses], body.axis_xz,
                                  y_lo=body.y_hem - 0.10 * body.size_y,
                                  y_hi=body.y_hem + 0.14 * body.size_y,
                                  margin=params.swing_margin))
    neck = np.isin(labels, ["neck"])
    if with_neck and neck.any():
        bounds.append(RadialBound([q[neck] for q in poses], body.axis_xz,
                                  y_lo=float(body.neck_y.min()) - 0.10 * body.size_y,
                                  y_hi=float(body.neck_y.max()) + 0.02 * body.size_y,
                                  margin=params.neck_margin))
    return bounds


def opening_mask(P: np.ndarray, pattern: Pattern, points: np.ndarray, params: ClothParams,
                 du: float = 0.0, dv: float = 0.0) -> np.ndarray:
    """Cells of the pattern where a held-out limb comes *through* the
    cloth — where its surface is within `open_reach` of the first pass's
    cloth, which is where an armhole belongs. A limb further out than that
    is simply in front of the garment: an arm held across the chest wants
    the cloth behind it, not a hole the size of the arm. The margin is
    dilated in cloth units, not cells, because the cells are not square."""
    rows, cols = pattern.grid_shape
    mask = np.zeros((rows, cols), dtype=bool)
    V = np.asarray(points, dtype=np.float64)
    if not len(V):
        return mask
    d, i = cKDTree(np.asarray(P, dtype=np.float64)).query(V, workers=-1)
    crossing = i[d < params.open_reach]
    if not len(crossing):
        return mask
    mask.ravel()[crossing] = True
    mask[:, 0] |= mask[:, -1]                       # the seam column is the same cloth
    mask[:, -1] = mask[:, 0]
    ru = max(int(round(params.open_margin / max(du, 1e-9))), 1) if du else 1
    rv = max(int(round(params.open_margin / max(dv, 1e-9))), 1) if dv else 1
    yy, xx = np.mgrid[-rv:rv + 1, -ru:ru + 1]
    disk = (yy / rv) ** 2 + (xx / ru) ** 2 <= 1.0 + 1e-9
    n = cols - 1
    tiled = np.concatenate([mask[:, :n]] * 3, axis=1)
    tiled = binary_dilation(tiled, disk)
    out = tiled[:, n:2 * n]
    return np.concatenate([out, out[:, :1]], axis=1)


def trim_slivers(mask: np.ndarray, width: float, du: float, dv: float) -> np.ndarray:
    """Widen the openings until no tongue of cloth narrower than `width`
    (in cloth units) is left standing between or beside them.

    A tongue narrower than twice the selvedge is gold on both faces and
    flips out under gravity — visible on the owl as a loose sail beside
    the hand slit, and as a bridge of cloth between the hand and the
    tablet that nothing was holding. Morphological opening of the *kept*
    region is the shortest statement of "no cloth narrower than this":
    erode, dilate, and whatever does not survive was never a panel. The
    structuring element is an ellipse in cells, because the cells are not
    square."""
    ru, rv = int(round(width / max(du, 1e-9))), int(round(width / max(dv, 1e-9)))
    if ru < 1 and rv < 1:
        return mask
    ru, rv = max(ru, 1), max(rv, 1)
    yy, xx = np.mgrid[-rv:rv + 1, -ru:ru + 1]
    disk = (yy / rv) ** 2 + (xx / ru) ** 2 <= 1.0 + 1e-9
    n = mask.shape[1] - 1
    tiled = np.concatenate([(~mask)[:, :n]] * 3, axis=1)
    tiled = np.pad(tiled, ((rv, rv), (0, 0)), mode="edge")   # hem and neckline are edges, not background
    opened = binary_dilation(binary_erosion(tiled, disk), disk)[rv:-rv, n:2 * n]
    out = ~np.concatenate([opened, opened[:, :1]], axis=1)
    return out | mask                                       # only ever widen an opening


def drop_offcuts(mask: np.ndarray, min_fraction: float = 0.04) -> np.ndarray:
    """Widen the openings to swallow every scrap of cloth the cuts left
    unattached to the garment.

    A cut that isolates a patch leaves a fragment held by nothing but its
    tethers: it swings free, and being small it is inside the selvedge
    everywhere, so it renders as a gold sail hanging off the front —
    which is exactly what it looked like on the owl, over the tablet.
    Anything smaller than `min_fraction` of the cloth is offcut."""
    n = mask.shape[1] - 1
    labels, count = robemod._components_periodic(~mask[:, :n])
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    kept = np.isin(labels, np.nonzero(sizes >= min_fraction * sizes.sum())[0])
    out = ~np.concatenate([kept, kept[:, :1]], axis=1)
    return out | mask


def signed_field(mask: np.ndarray, du: float, dv: float, smooth: float = 0.0) -> np.ndarray:
    """Signed distance to the nearest opening, **in cloth units**: negative
    inside an opening, positive in the cloth.

    Two things this must get right, both learned the hard way. The grid's
    cells are not square — 16 mm around the piece against 9 mm down it on
    the owl — so the transform is given its `sampling` rather than counting
    cells, or every vertical cut edge wears a selvedge nearly twice as wide
    as every horizontal one. And the *mask* is what gets rounded, not the
    field: smoothing a distance field flattens its gradient near curvature,
    which pushed the gold binding out to four times its width."""
    rows, cols = mask.shape
    if not mask.any():
        return np.full((rows, cols), max(rows * dv, cols * du))
    n = cols - 1
    tiled = np.concatenate([mask[:, :n]] * 3, axis=1)
    if smooth > 0:
        # round the opening's edge without ever shrinking it
        tiled = (gaussian_filter(tiled.astype(np.float64), sigma=smooth, mode="nearest") > 0.5) | tiled
    f = np.where(tiled, -_cell_distance(tiled, (dv, du)), _cell_distance(~tiled, (dv, du)))
    f = f[:, n:2 * n]
    return np.concatenate([f, f[:, :1]], axis=1)


def _cell_distance(mask: np.ndarray, sampling=None) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(mask, sampling=sampling)


# --------------------------------------------------------------------------
# the garment
# --------------------------------------------------------------------------

class KenteTunic(NamedTuple):
    primitive: PrimitiveSpec
    grid_shape: tuple[int, int]
    grid_index: np.ndarray
    pattern: Pattern
    field: np.ndarray          # (rows*cols,) pattern-space signed distance to the nearest cut edge
    positions: np.ndarray      # (rows*cols,) draped positions before the cut, for warm starts
    info: dict


def build_kente_tunic(mesh: trimesh.Trimesh, region_map: RegionMap, weave_params: WeaveParams,
                      params: ClothParams = ClothParams(), band_frame=None,
                      obstacle_points: Sequence[np.ndarray] | None = None,
                      obstacle_mesh: trimesh.Trimesh | None = None,
                      sweep_vertices: Sequence[np.ndarray] | None = None,
                      n_theta: int = 144) -> KenteTunic:
    """Measure, draft, drape, cut the openings, drape again, then bake the
    weave onto the pattern's own chart."""
    if params.strips_around % weave_params.strip_cycle != 0:
        raise ValueError(f"build_kente_tunic: strips_around ({params.strips_around}) must be a multiple of the "
                         f"weave's strip_cycle ({weave_params.strip_cycle})")
    labels = np.asarray(region_map.labels)
    body = measure_body(mesh, region_map, params, band_frame=band_frame, n_theta=n_theta)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = _swing_bounds(mesh, labels, body, params, sweep_vertices)
    # the pattern is also cut for the sleeve's root, which the shoulder
    # panel lies over (the sleeve stands `clearance_root` off the wing, and
    # the cloth another offset off the sleeve): a draft-only bound, since
    # in the solve the sleeve is a collider, not a floor
    pattern_bounds = list(bounds)
    if obstacle_mesh is not None:
        sv = np.asarray(obstacle_mesh.vertices, dtype=np.float64)
        past = np.concatenate([np.asarray(o, dtype=np.float64) for o in (obstacle_points or []) if len(o)]) \
            if obstacle_points else np.zeros((0, 3))
        root = sv if not len(past) else sv[cKDTree(past).query(sv, workers=-1)[0] > 0.02]
        if len(root):
            pattern_bounds.append(RadialBound([root], body.axis_xz, y_lo=float(root[:, 1].min()) - 0.01,
                                              y_hi=float(root[:, 1].max()) + 0.01, n_theta=72, n_rows=12,
                                              smooth=0.5, margin=params.collide_offset))
    pattern = draft(body, params, bounds=pattern_bounds)
    rows, cols = pattern.grid_shape
    n = rows * cols

    protruding = held_out_vertices(mesh, labels, body.scan, body.cells, body.axis_xz, params.hold_tolerance)
    if obstacle_mesh is not None:
        # a limb already wearing a sleeve is not cut around twice: the sleeve
        # is what the cloth meets there, and it is wider than the wing
        near = cKDTree(np.asarray(obstacle_mesh.vertices)).query(
            np.asarray(mesh.vertices), distance_upper_bound=params.sleeve_covers)[0]
        protruding &= ~(near < params.sleeve_covers)
    face_prot = protruding[mesh.faces].any(axis=1)
    torso_collider = Collider(mesh, face_mask=~face_prot)
    # the legs and the tail swing; the hem must clear where they go, not only
    # where they are
    # what sweeps through the height the hem hangs at: the legs and tail on
    # the hop, and the folded wing's lower arm when the tablet comes up. The
    # cloth over the wing's *shoulder* rides the wing joint and needs no
    # bound; the cloth below the weight blend is body-driven and does
    colliders = [Collider(mesh)]
    if obstacle_mesh is not None:
        colliders.append(Collider(obstacle_mesh, n=60_000))

    keep = np.ones(n, dtype=bool)
    pinned = np.zeros(n, dtype=bool)
    pinned[:cols] = True                                   # the neckline, tucked under the collar band
    P0 = initial_positions(pattern, body)
    v_rows = pattern.v_profile
    sets = constraint_sets(pattern.phi.ravel(), pattern.v.ravel(), pattern.circumference, v_rows,
                           keep, rows, cols, params)
    tet = build_tethers(pattern.phi.ravel(), pattern.v.ravel(), pattern.circumference, v_rows,
                        keep, pinned, rows, cols)
    # the clearance the solver keeps, graded down the cloth: tight where the
    # garment is held at the neck, generous at the hem where it hangs free
    grade = lambda v: params.collide_offset + (params.collide_offset_hem - params.collide_offset) * np.clip(
        v / max(pattern.length, 1e-9), 0.0, 1.0) ** 1.4
    P1, d1 = drape(P0, pattern, sets, [torso_collider], pinned, keep, params,
                   offset=grade(pattern.v.ravel()), tethers=tet)

    dv = pattern.length / max(rows - 1, 1)
    du = float(np.mean(pattern.circumference)) / (cols - 1)
    open_points = [np.asarray(mesh.vertices, dtype=np.float64)[protruding]]
    open_points += [np.asarray(o, dtype=np.float64) for o in (obstacle_points or []) if len(o)]
    mask = opening_mask(P1, pattern, np.concatenate(open_points), params, du, dv)
    mask = trim_slivers(mask, params.min_panel_width, du, dv)
    mask = drop_offcuts(mask, params.min_panel_fraction)
    # the yoke stays whole: the second drape lays it over whatever comes
    # through there (the sleeve's root, the folded wing's shoulder) instead
    # of cutting the neckline open around it (measured on the first tunic:
    # the armscye merged into the neckline loop over 33 degrees of it)
    mask[pattern.v < params.yoke_keep] = False
    if obstacle_mesh is not None and params.shoulder_keep > params.yoke_keep:
        # where the sleeve comes through, the cloth above stays whole too:
        # the second drape lays it over the sleeve's root (a collider), the
        # way a cap sleeve sits over a sleeve head (measured on the first
        # tunic: 0 of the 196 root-ring vertices had cloth over them)
        ph = pattern.phi[0]
        sleeve_cells = opening_mask(P1, pattern, np.concatenate([np.asarray(o) for o in (obstacle_points or []) if len(o)]),
                                    params, du, dv) if obstacle_points else np.zeros_like(mask)
        cols_hit = np.nonzero(sleeve_cells[(pattern.v[:, 0] >= params.yoke_keep) & (pattern.v[:, 0] < params.shoulder_keep)].any(axis=0))[0]
        if len(cols_hit):
            centre = float(np.angle(np.exp(2j * np.pi * ph[cols_hit]).mean())) / (2 * np.pi) % 1.0
            half = 0.5 * len(cols_hit) / (cols - 1) + params.shoulder_margin
            near = np.abs(_wrap(ph - centre)) <= half
            mask[np.ix_(pattern.v[:, 0] < params.shoulder_keep, near)] = False
    field = signed_field(mask, du, dv, params.open_smooth).ravel()
    # the neckline and the hem are cut edges too, so the binding finds them
    edge = np.minimum(pattern.v, pattern.fall[None, :] - pattern.v).ravel()
    field = np.minimum(field, edge)
    F = field.reshape(rows, cols)
    faces, src_a, src_b, frac, n_snapped = robemod.cut_grid(F)
    if len(faces) < 10:
        raise ValueError("build_kente_tunic: the openings left no cloth")

    # the cut moves boundary vertices in *pattern* space; re-evaluate the
    # pattern there and warm-start the second drape from the first result
    phi = pattern.phi.ravel()
    vv = pattern.v.ravel()
    phi_cut = phi[src_a] + frac * (_wrap(phi[src_b] - phi[src_a]))
    v_cut = vv[src_a] + frac * (vv[src_b] - vv[src_a])
    P_start = P1[src_a] + frac[:, None] * (P1[src_b] - P1[src_a])
    keep2 = np.zeros(n, dtype=bool)
    keep2[np.unique(faces)] = True
    pinned2 = pinned & keep2
    pat2 = pattern._replace(phi=(phi_cut % 1.0).reshape(rows, cols), v=v_cut.reshape(rows, cols))
    sets2 = constraint_sets(phi_cut, v_cut, pattern.circumference, v_rows, keep2, rows, cols, params)
    tet2 = build_tethers(phi_cut, v_cut, pattern.circumference, v_rows, keep2, pinned2, rows, cols)
    off2 = grade(v_cut)
    P2, d2 = drape(P_start, pat2, sets2, colliders, pinned2, keep2, params, offset=off2, tethers=tet2,
                   bounds=bounds)
    d2["pushed_out"] = sum(push_out(P2, keep2, c, off2 * 0.6) for c in colliders)
    d2["swept_clear"] = sum(b.apply(P2, free=keep2) for b in bounds)
    seam_src, seam_dup = seam_columns(rows, cols, keep2)
    P2[seam_dup] = P2[seam_src]

    used = np.unique(faces)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts = P2[used]
    faces_out = remap[faces]

    # drop the triangles the cut collapsed
    tri = verts[faces_out]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces_out = faces_out[area2 > 1e-12]

    uvs = np.stack([phi_cut[used] % 1.0, v_cut[used] / max(pattern.length, 1e-9)], axis=1)
    uvs[used % cols == cols - 1, 0] = 1.0
    normals = _grid_normals(verts, faces_out, used, cols, n)
    radial = verts - np.array([body.axis_xz[0], 0.0, body.axis_xz[1]])
    radial[:, 1] = 0.0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    if float(np.mean(np.sum(normals * radial, axis=1))) < 0:
        faces_out = faces_out[:, ::-1]
        normals = -normals
    info = {
        "vertices": int(len(verts)), "faces": int(len(faces_out)),
        "pattern": {"circumference": [round(float(pattern.circumference[0]), 3), round(float(pattern.circumference[-1]), 3)],
                    "chest": round(float(pattern.circumference.max()), 3),
                    "wrap": pattern.wrap_perimeter,
                    "profile": {"v": [round(float(x), 4) for x in pattern.v_profile],
                                "circ": [round(float(x), 4) for x in pattern.circumference]},
                    "length": round(float(pattern.length), 3), "grid": [rows, cols]},
        "neckline_frac": [round(float((body.neck_y.min() - body.y0) / body.size_y), 3),
                          round(float((body.neck_y.max() - body.y0) / body.size_y), 3)],
        "hem_frac_target": params.hem_frac,
        "hem_frac_actual": [round(float((verts[:, 1].min() - body.y0) / body.size_y), 3),
                            round(float((verts[:, 1].max() - body.y0) / body.size_y), 3)],
        "width": round(float(verts[:, 0].max() - verts[:, 0].min()), 3),
        "depth": round(float(verts[:, 2].max() - verts[:, 2].min()), 3),
        "width_by_height": _width_profile(verts, body),
        "snapped": int(n_snapped), "opened_cells": int((F < 0).sum()),
        "obstacle_points": int(sum(len(np.asarray(o)) for o in (obstacle_points or []))),
        "drape": [d1, d2], "colorway": weave_params.colorway.name,
    }

    img, normal_img = _bake(verts, faces_out, uvs, normals, pat2, F, weave_params, params, body)
    mat = MaterialSpec(name=params.material_name, base_color_image=img, image_format="PNG",
                       roughness=params.roughness, metallic=0.0, double_sided=True, normal_image=normal_img)
    prim = PrimitiveSpec(name="kente_tunic", vertices=verts, faces=faces_out, normals=normals, uvs=uvs, material=mat)
    info["image"] = img.size
    return KenteTunic(primitive=prim, grid_shape=(rows, cols), grid_index=used, pattern=pat2,
                      field=field, positions=P2, info=info)


def _width_profile(verts: np.ndarray, body: BodyMeasure) -> dict:
    out = {}
    for f in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        y = body.y0 + f * body.size_y
        band = np.abs(verts[:, 1] - y) < 0.012 * body.size_y
        if band.sum() > 4:
            out[f"{f:.2f}"] = round(float(verts[band, 0].max() - verts[band, 0].min()), 3)
    return out


def _wrap(d: np.ndarray) -> np.ndarray:
    """Shortest signed difference of two phi values on the seam."""
    return (d + 0.5) % 1.0 - 0.5


def _grid_normals(verts: np.ndarray, faces: np.ndarray, used: np.ndarray, cols: int, n: int,
                  periodic: bool = True) -> np.ndarray:
    """Vertex normals taken on the cloth welded across the duplicated seam
    column, so the chart's seam shades without a crease."""
    weld = np.arange(n)
    if periodic:
        weld[cols - 1::cols] = np.arange(n)[0::cols]    # last column -> first
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    welded = remap[weld[used[faces]]]
    welded = np.where(welded >= 0, welded, faces)
    m = trimesh.Trimesh(vertices=verts, faces=welded, process=False)
    N = np.asarray(m.vertex_normals, dtype=np.float64).copy()
    bad = np.linalg.norm(N, axis=1) < 0.5
    if bad.any():
        N[bad] = np.array([0.0, 1.0, 0.0])
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)


def _bake(verts, faces, uvs, normals, pattern: Pattern, F: np.ndarray, weave_params: WeaveParams,
          params: ClothParams, body: BodyMeasure):
    """The weave, painted onto the pattern's own chart: `px` runs along the
    weft (around the piece) and `py` down the warp, both read from the
    chart's own coordinates rather than from world position — which is the
    point of drafting a pattern. A fold in the cloth carries the weave
    with it because the weave is *in* the cloth."""
    posmap = bake_position_map(verts, faces, uvs, size=tuple(params.texture_size), vertex_normals=normals)
    h, w = posmap.mask.shape
    ys, xs = np.nonzero(posmap.mask)
    u = (xs + 0.5) / w
    v = (ys + 0.5) / h
    px_per_u = weave_params.strip_px * params.strips_around
    blocks = params.blocks_tall
    if blocks is None:
        # the weave is woven, not printed: one pixel of the strip must cover
        # the same length of cloth across the piece as down it
        c_mean = float(np.mean(pattern.circumference))
        blocks = px_per_u / max(c_mean, 1e-9) * pattern.length / weave_params.block_px
    px = u * px_per_u
    py = v * weave_params.block_px * blocks
    # the direction the weave's warp runs, in world space, per texel: down
    # the cloth. Taken from the surface normal and the world up, which is
    # true wherever the cloth hangs and harmless where it does not.
    N_local = posmap.normal[posmap.mask].astype(np.float64)
    up = np.array([0.0, 1.0, 0.0])
    t_dir = np.cross(np.broadcast_to(up, N_local.shape), N_local)
    t_dir /= np.maximum(np.linalg.norm(t_dir, axis=1, keepdims=True), 1e-9)
    b_dir = np.cross(N_local, t_dir)
    img, normal_img = bake_weave_maps(posmap, px, py, t_dir, b_dir, weave_params, verts, faces, uvs,
                                      bake_normal=params.bake_normal, bump_strength=params.bump_strength, dilate=6)
    if params.edge_band > 0:
        rows, cols = pattern.grid_shape
        # pattern-space distance to the nearest cut edge, in cloth units
        gx = np.clip(u * (cols - 1), 0, cols - 1 - 1e-6)
        fall_at = np.interp(u, pattern.phi[0], pattern.fall)
        gy = np.clip(v * pattern.length / np.maximum(fall_at, 1e-9) * (rows - 1), 0, rows - 1 - 1e-6)
        i0, j0 = gy.astype(int), gx.astype(int)
        fy, fx = gy - i0, gx - j0
        edge = ((1 - fy) * ((1 - fx) * F[i0, j0] + fx * F[i0, j0 + 1])
                + fy * ((1 - fx) * F[i0 + 1, j0] + fx * F[i0 + 1, j0 + 1]))
        arr = np.asarray(img).astype(np.float64) / 255.0
        band = edge < params.edge_band
        arr[ys[band], xs[band]] = np.array([*params.edge_band_color, 1.0])
        stitch = (edge >= params.edge_band) & (edge < params.edge_band * 1.18)
        arr[ys[stitch], xs[stitch]] = np.array([*weave_params.colorway.motif, 1.0])
        img = to_image(arr)
    return img, normal_img
