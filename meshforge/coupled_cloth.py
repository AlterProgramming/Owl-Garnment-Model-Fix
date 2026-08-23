"""Animated-reference cloth coupling.

This is the core runtime-facing cloth model for the owl garment: a skinned
reference surface carries broad character motion, while a PBD residual solve
adds cloth freedom, collision response and secondary motion around that moving
reference.  Collision no longer has to pretend to be attachment.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from meshforge import drape as D
from meshforge import fk
from meshforge.cloth_exec import _contact_project, _separate_topology


@dataclasses.dataclass(frozen=True)
class CoupledClothParams:
    dt: float = 1.0 / 30.0
    inner_iterations: int = 5
    gravity: float = -9.0
    damping: float = 0.91
    drive: float = 0.72
    drive_damping: float = 0.45
    friction: float = 0.55
    collide_offset: float = 0.014
    max_deviation: float = 0.055
    self_collide: float = 0.0


def authority_from_weights(weights, joint_names, grid_index=None, grid_shape=None):
    """Return `(authority, max_deviation)` for animated-reference coupling.

    Authority is high where the cloth is structurally supported by the torso,
    medium where it locally rides a moving wing root, and low near the hem or
    loose panels.  This keeps the garment registered without turning it into a
    rigid duplicate of the body animation.
    """
    W = np.asarray(weights, dtype=np.float64)
    names = list(joint_names)
    torso = np.zeros(len(W))
    wing = np.zeros(len(W))
    for name in ("body", "chest"):
        if name in names:
            torso += W[:, names.index(name)]
    for name in ("wing_left", "wing_left_tip", "wing_right"):
        if name in names:
            wing += W[:, names.index(name)]
    other = np.clip(1.0 - torso - wing, 0.0, 1.0)

    authority = 0.78 * torso + 0.58 * wing + 0.22 * other
    max_dev = 0.022 * torso + 0.045 * wing + 0.090 * other

    if grid_index is not None and grid_shape is not None:
        rows, cols = grid_shape
        gi = np.asarray(grid_index, dtype=np.int64)
        row = gi // cols
        row_t = row / max(rows - 1, 1)
        # Top support ring should behave like cloth tied to the owl; the hem
        # should lag and swing instead of inheriting every body transform.
        authority = authority * (1.0 - 0.42 * row_t) + 0.98 * (row == 0)
        max_dev = np.where(row == 0, np.minimum(max_dev, 0.014), max_dev + 0.055 * row_t)

    return np.clip(authority, 0.0, 0.99), np.maximum(max_dev, 1e-5)


def skinned_reference(vertices, weights, joint_names, joints, rotations=None, translations=None):
    """Character-space guide positions for one animation pose."""
    world = fk.world_matrices(joints, rotations=rotations, translations=translations)
    skin = fk.skin_matrices(joints, world)
    return fk.pose_vertices(np.asarray(vertices, dtype=np.float64), np.asarray(weights, dtype=np.float64), list(joint_names), skin)


def _apply_reference_drive(P, guide, free, authority, max_deviation, strength):
    a = np.clip(np.asarray(authority, dtype=np.float64) * float(strength), 0.0, 0.99)
    live = free & (a > 0)
    if live.any():
        P[live] += a[live, None] * (guide[live] - P[live])
    d = P - guide
    L = np.linalg.norm(d, axis=1)
    md = np.asarray(max_deviation, dtype=np.float64)
    too_far = free & (L > md) & (L > 1e-12)
    if too_far.any():
        P[too_far] = guide[too_far] + d[too_far] * (md[too_far] / L[too_far])[:, None]


def step(
    P,
    V,
    guide,
    previous_guide,
    pattern,
    sets,
    colliders,
    pinned,
    keep,
    params: CoupledClothParams = CoupledClothParams(),
    authority=None,
    max_deviation=None,
    bounds: Sequence = (),
    tethers=None,
    periodic: bool = True,
):
    """Advance one animation frame.

    The solver operates on real cloth positions, but every frame has a moving
    reference produced by character skinning.  We first inherit part of the
    guide velocity, then solve constraints/contact around the guide.  The
    output is the cloth surface for this frame plus the velocity to carry into
    the next frame.
    """
    P = np.asarray(P, dtype=np.float64).copy()
    V = np.asarray(V, dtype=np.float64).copy()
    guide = np.asarray(guide, dtype=np.float64)
    previous_guide = np.asarray(previous_guide, dtype=np.float64)
    pinned = np.asarray(pinned, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    free = keep & ~pinned

    if authority is None:
        authority = np.full(len(P), 0.55, dtype=np.float64)
    else:
        authority = np.asarray(authority, dtype=np.float64)
    if max_deviation is None:
        max_deviation = np.full(len(P), params.max_deviation, dtype=np.float64)
    else:
        max_deviation = np.asarray(max_deviation, dtype=np.float64)

    rows, cols = pattern.grid_shape
    seam_src, seam_dup = D.seam_columns(rows, cols, keep, periodic)
    inv_mass = np.where(free, 1.0, 0.0)

    guide_v = (guide - previous_guide) / max(params.dt, 1e-8)
    V[free] = (1.0 - params.drive_damping) * V[free] + params.drive_damping * guide_v[free]
    V[:, 1] += params.gravity * params.dt
    V *= params.damping
    V[~free] = 0.0

    Pn = P + V * params.dt
    Pn[~free] = guide[~free]
    Pn[seam_dup] = Pn[seam_src]

    acc = np.empty_like(Pn)
    cnt = np.empty(len(Pn))
    normal_acc = np.zeros_like(Pn)
    normal_count = np.zeros(len(Pn), dtype=np.int32)

    for _ in range(params.inner_iterations):
        acc[:] = 0.0
        cnt[:] = 0.0
        for cs in sets:
            D._project(Pn, cs, inv_mass, acc, cnt)
        moved = cnt > 0
        Pn[moved] += acc[moved] / cnt[moved][:, None]
        if tethers is not None:
            D.apply_tethers(Pn, tethers)
        _apply_reference_drive(Pn, guide, free, authority, max_deviation, params.drive)
        Pn[seam_dup] = Pn[seam_src]

        for col in colliders:
            hit, nrm = _contact_project(Pn, col, params.collide_offset, free)
            if hit.any():
                normal_acc[hit] += nrm[hit]
                normal_count[hit] += 1
        for b in bounds:
            b.apply(Pn, free=free)
        Pn[~keep] = guide[~keep]
        Pn[seam_dup] = Pn[seam_src]

    if params.self_collide > 0:
        _separate_topology(Pn, free, params.self_collide, rows, cols, periodic)
        for col in colliders:
            hit, nrm = _contact_project(Pn, col, params.collide_offset, free)
            if hit.any():
                normal_acc[hit] += nrm[hit]
                normal_count[hit] += 1

    Vn = (Pn - P) / max(params.dt, 1e-8)
    hit = normal_count > 0
    if hit.any():
        N = normal_acc[hit]
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        vh = Vn[hit]
        vn = np.einsum("ij,ij->i", vh, N)
        vt = vh - vn[:, None] * N
        Vn[hit] = params.friction * vt + np.maximum(vn, 0.0)[:, None] * N
    Vn[~free] = 0.0
    return Pn, Vn


def simulate_guides(rest_vertices, guides, pattern, sets, colliders_by_frame, pinned, keep,
                    params: CoupledClothParams = CoupledClothParams(), authority=None,
                    max_deviation=None, bounds_by_frame=None, tethers=None, periodic: bool = True):
    """Bake a sequence of simulated cloth frames from skinned guide frames."""
    guides = np.asarray(guides, dtype=np.float64)
    if guides.ndim != 3:
        raise ValueError("simulate_guides: guides must be (frames, vertices, 3)")
    P = np.asarray(rest_vertices, dtype=np.float64).copy()
    V = np.zeros_like(P)
    frames = []
    bounds_by_frame = bounds_by_frame or [()] * len(guides)
    for i, guide in enumerate(guides):
        prev = guides[i - 1] if i else guide
        colliders = colliders_by_frame[i] if isinstance(colliders_by_frame, list) else colliders_by_frame
        bounds = bounds_by_frame[i] if isinstance(bounds_by_frame, list) else bounds_by_frame
        P, V = step(P, V, guide, prev, pattern, sets, colliders, pinned, keep, params,
                    authority=authority, max_deviation=max_deviation, bounds=bounds,
                    tethers=tethers, periodic=periodic)
        frames.append(P.copy())
    return np.asarray(frames)
