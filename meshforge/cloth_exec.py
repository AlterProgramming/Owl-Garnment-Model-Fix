"""Experimental cloth execution path.

This module is intentionally separate from :mod:`meshforge.drape` so the
current garment remains a reproducible baseline.  ``tools/build_cloth_exec.py``
installs these functions at runtime and then runs the normal owl pipeline.

The execution path changes two assumptions that made the shipped wrap read as
a rigid shell:

* contact is projected *inside* every constraint iteration and friction acts on
  tangential velocity, instead of projecting the body once after the solve and
  damping the whole velocity vector;
* cloth close to a moving wing root may inherit a local share of that root's
  skinning weights.  The top support row remains torso-driven, but contact
  immediately below it is allowed to move with the surface that actually
  supports it.  This replaces a static swept-volume shell with local kinematic
  support.

It also enables a conservative topology-aware self-collision pass for the
coarse wrap design mesh.  Immediate pattern neighbours are excluded, so the
self-collision radius cannot inflate the cloth's own grid.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


def _smoothstep(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _contact_project(P, collider, offset, free):
    """Project against one sampled collider and return hit normals.

    ``Collider.resolve`` is deliberately not used here because the execution
    solver needs the contact normal after projection for velocity/friction.
    """
    _, idx = collider.tree.query(P, workers=-1)
    nrm = collider.normals[idx]
    delta = P - collider.points[idx]
    sd = np.einsum("ij,ij->i", delta, nrm)
    off = np.asarray(offset, dtype=np.float64)
    if off.ndim == 0:
        hit = free & (sd < float(off))
        if hit.any():
            P[hit] += (float(off) - sd[hit])[:, None] * nrm[hit]
    else:
        hit = free & (sd < off)
        if hit.any():
            P[hit] += (off[hit] - sd[hit])[:, None] * nrm[hit]
    return hit, nrm


def _auto_self_radius(sets, stretch_k: float, n: int) -> float:
    """A conservative thickness from the design mesh's structural edges."""
    # The full-resolution tunic can have ~13k design vertices.  Automatic
    # O(n log n + pairs) self-contact is aimed at the coarse wrap; larger
    # meshes can still opt in through ClothParams.self_collide.
    if n > 6000:
        return 0.0
    rest = [np.asarray(s.rest, dtype=np.float64).ravel() for s in sets
            if abs(float(s.k) - float(stretch_k)) < 1e-9 and np.size(s.rest)]
    if not rest:
        return 0.0
    r = np.concatenate(rest)
    r = r[np.isfinite(r) & (r > 1e-6)]
    if not len(r):
        return 0.0
    # About half of the 20th-percentile structural spacing.  Adjacent pattern
    # vertices are excluded below, so this is cloth thickness, not grid ease.
    return float(0.48 * np.percentile(r, 20))


def _separate_topology(P, free, radius: float, rows: int, cols: int, periodic: bool) -> int:
    """One topology-aware cloth/cloth separation pass.

    Pairs within two pattern cells are structural neighbours and must never be
    treated as self-collision.  The old helper's docstring promised this but
    did not implement it, which is why enabling self-collision inflated the
    garment.
    """
    if radius <= 0 or len(P) < 2:
        return 0
    pairs = np.asarray(list(cKDTree(P).query_pairs(radius)), dtype=np.int64)
    if not len(pairs):
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    ri, rj = i // cols, j // cols
    ci, cj = i % cols, j % cols
    if periodic and cols > 1:
        real = cols - 1
        ci = np.where(ci == real, 0, ci)
        cj = np.where(cj == real, 0, cj)
        dc = np.abs(ci - cj)
        dc = np.minimum(dc, real - dc)
    else:
        dc = np.abs(ci - cj)
    near_pattern = (np.abs(ri - rj) <= 2) & (dc <= 2)
    live = ~near_pattern & (free[i] | free[j])
    i, j = i[live], j[live]
    if not len(i):
        return 0
    d = P[j] - P[i]
    L = np.linalg.norm(d, axis=1)
    good = (L > 1e-9) & (L < radius)
    i, j, d, L = i[good], j[good], d[good], L[good]
    if not len(i):
        return 0
    wi = free[i].astype(np.float64)
    wj = free[j].astype(np.float64)
    ws = wi + wj
    corr = ((radius - L) / np.maximum(L * ws, 1e-12))[:, None] * d
    np.subtract.at(P, i, wi[:, None] * corr)
    np.add.at(P, j, wj[:, None] * corr)
    return int(len(i))


def drape(P, pattern, sets, colliders, pinned, keep, params, offset=None,
          tethers=None, iterations: int | None = None, bounds: Sequence = (),
          periodic: bool = True):
    """Drop-in replacement for :func:`meshforge.drape.drape`.

    It keeps the repository's pattern/constraint model but changes the order
    of operations to a contact-first PBD solve: structural projection,
    tethers, body contact and swept bounds are reconciled in every inner
    iteration.  Contact velocity is then decomposed into normal/tangent parts;
    friction damps only the tangent and inward normal velocity is removed.
    """
    from meshforge import drape as D

    P = np.array(P, dtype=np.float64)
    n = len(P)
    w = np.where(pinned | ~keep, 0.0, 1.0)
    V = np.zeros_like(P)
    acc = np.empty_like(P)
    cnt = np.empty(n)
    iterations = params.iterations if iterations is None else iterations
    free = keep & ~pinned
    rows, cols = pattern.grid_shape
    seam_src, seam_dup = D.seam_columns(rows, cols, keep, periodic)
    P[seam_dup] = P[seam_src]

    self_radius = float(params.self_collide)
    if self_radius <= 0:
        self_radius = _auto_self_radius(sets, params.stretch, n)

    contacts_total = 0
    self_pairs_total = 0
    for step in range(iterations + params.settle_iterations):
        settling = step >= iterations
        g = 0.0 if settling else params.gravity
        V[:, 1] += g * params.dt
        V *= params.damping if not settling else 0.5
        V[~free] = 0.0
        Pn = P + V * params.dt
        Pn[~free] = P[~free]

        normal_acc = np.zeros_like(P)
        normal_count = np.zeros(n, dtype=np.int32)
        off = params.collide_offset if offset is None else offset

        for _ in range(params.inner_iterations):
            acc[:] = 0.0
            cnt[:] = 0.0
            for cs in sets:
                D._project(Pn, cs, w, acc, cnt)
            moved = cnt > 0
            Pn[moved] += acc[moved] / cnt[moved][:, None]
            Pn[~free] = P[~free]
            if tethers is not None:
                D.apply_tethers(Pn, tethers)
            Pn[seam_dup] = Pn[seam_src]

            # The body is not an afterthought: every structural correction is
            # immediately reconciled with contact before the next one.
            for col in colliders:
                hit, nrm = _contact_project(Pn, col, off, free)
                if hit.any():
                    contacts_total += int(hit.sum())
                    normal_acc[hit] += nrm[hit]
                    normal_count[hit] += 1
            for b in bounds:
                b.apply(Pn, free=free)
            Pn[seam_dup] = Pn[seam_src]

        # Self-contact is less frequent than body contact.  Re-project the
        # body afterwards because separating two cloth layers can push one
        # of them back into the character.
        if self_radius > 0 and (step % 4 == 0 or settling):
            self_pairs_total += _separate_topology(Pn, free, self_radius, rows, cols, periodic)
            for col in colliders:
                hit, nrm = _contact_project(Pn, col, off, free)
                if hit.any():
                    contacts_total += int(hit.sum())
                    normal_acc[hit] += nrm[hit]
                    normal_count[hit] += 1
            for b in bounds:
                b.apply(Pn, free=free)

        Pn[seam_dup] = Pn[seam_src]
        V = (Pn - P) / params.dt

        # Contact friction: preserve separating motion, kill motion into the
        # body, and damp only the tangent.  Multiplying the entire velocity by
        # `friction` makes cloth look sticky while still permitting sliding.
        hit = normal_count > 0
        if hit.any():
            N = normal_acc[hit]
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
            vh = V[hit]
            vn = np.einsum("ij,ij->i", vh, N)
            vt = vh - vn[:, None] * N
            retain = float(np.clip(params.friction, 0.0, 1.0))
            V[hit] = retain * vt + np.maximum(vn, 0.0)[:, None] * N
        V[~free] = 0.0
        P = Pn

    return P, {
        "iterations": int(iterations + params.settle_iterations),
        "contacts": int(contacts_total),
        "self_collision_radius": round(float(self_radius), 6),
        "self_pairs": int(self_pairs_total),
        "solver": "contact-driven-exec-v1",
    }


def support_weights(verts, grid_index, grid_shape, pins, body_V, body_W,
                    joint_names, y_hip, pin_weights=None, periodic: bool = True,
                    sigma: float = 1.0):
    """Support-graph skinning plus local moving-surface support.

    The support row remains body/chest driven.  Below it, vertices that are
    actually close to a wing surface may inherit that surface's local wing
    weight.  The influence fades with distance, with height below the hip,
    and over the first few rows below the pinned support.  This is a compact
    real-time approximation to body/cloth contact for the exported GLB: the
    moving root carries the cloth touching it instead of moving inside a
    pre-cleared static shell.
    """
    from meshforge import robe as R

    verts = np.asarray(verts, dtype=np.float64)
    body_V = np.asarray(body_V, dtype=np.float64)
    body_W = np.asarray(body_W, dtype=np.float64)
    pins = np.asarray(pins, dtype=np.float64)
    joint_names = list(joint_names)
    rows, cols = grid_shape
    grid_index = np.asarray(grid_index)
    col = grid_index % cols
    row = grid_index // cols
    jb, jc = joint_names.index("body"), joint_names.index("chest")

    if pin_weights is None:
        near = cKDTree(body_V).query(pins, workers=-1)[1]
        Wp = np.zeros((cols, len(joint_names)), dtype=np.float64)
        Wp[:, jb] = body_W[near, jb]
        Wp[:, jc] = body_W[near, jc]
        s = Wp.sum(axis=1)
        Wp[s < 1e-6, jc] = 1.0
        Wp /= np.maximum(Wp.sum(axis=1, keepdims=True), 1e-12)
    else:
        Wp = np.asarray(pin_weights, dtype=np.float64)
        if Wp.shape != (cols, len(joint_names)):
            raise ValueError(f"support_weights: pin_weights must be ({cols}, {len(joint_names)}), got {Wp.shape}")

    y_top = pins[:, 1]
    s = np.clip((y_top[col] - verts[:, 1]) / np.maximum(y_top[col] - y_hip, 1e-6), 0.0, 1.0)
    W = (1.0 - s)[:, None] * Wp[col]
    W[:, jb] += s

    # The tail deliberately passes pin_weights=chest_one; keep it a hanging
    # sash end rather than attaching it to whichever wing passes nearby.
    if pin_weights is None:
        k = min(8, len(body_V))
        dist, idx = cKDTree(body_V).query(verts, k=k, workers=-1)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        q = 1.0 / np.maximum(dist, 1e-5)
        q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        Wnear = np.einsum("nk,nkj->nj", q, body_W[idx])

        allowed = np.array([nm in ("body", "chest", "wing_left", "wing_left_tip", "wing_right")
                            for nm in joint_names])
        Wnear[:, ~allowed] = 0.0
        Wnear /= np.maximum(Wnear.sum(axis=1, keepdims=True), 1e-12)
        wing_cols = [i for i, nm in enumerate(joint_names)
                     if nm in ("wing_left", "wing_left_tip", "wing_right")]
        wing_mass = Wnear[:, wing_cols].sum(axis=1) if wing_cols else np.zeros(len(verts))

        H = max(float(np.ptp(body_V[:, 1])), 1e-6)
        d0 = dist[:, 0]
        proximity = 1.0 - _smoothstep((d0 - 0.008 * H) / (0.030 * H))
        above_hip = _smoothstep((verts[:, 1] - y_hip) / (0.12 * H))
        # Do not change the pinned support row; ramp local support in over the
        # next ~12% of the design rows so the tie itself remains torso-worn.
        row_ramp = _smoothstep(row / max(0.12 * rows, 1.0))
        a = np.clip(1.6 * wing_mass, 0.0, 1.0) * proximity * above_hip * row_ramp
        a *= 0.88
        W = (1.0 - a)[:, None] * W + a[:, None] * Wnear

    W = R.smooth_grid_weights(W, grid_index, grid_shape, sigma=sigma, periodic=periodic)
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
