from __future__ import annotations

from typing import Sequence

import numpy as np

from meshforge import cloth_exec as E


def _smoothstep(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _radius_profile(body, theta: float, field: str) -> np.ndarray:
    A = np.asarray(getattr(body, field), dtype=np.float64)
    out = np.asarray([
        np.interp(theta, body.theta, row, period=2.0 * np.pi)
        for row in A
    ], dtype=np.float64)
    bad = ~np.isfinite(out) | (out <= 0.0)
    if bad.any() and field != "envelope":
        fallback = _radius_profile(body, theta, "envelope")
        out[bad] = fallback[bad]
        bad = ~np.isfinite(out) | (out <= 0.0)
    if bad.any():
        good = ~bad
        if good.any():
            out[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), out[good])
        else:
            out[:] = float(np.nanmedian(np.asarray(body.neck_r, dtype=np.float64)))
    return out


def _surface_radius(body, theta: float, y: np.ndarray, top_y: float, top_r: float) -> np.ndarray:
    H = float(body.size_y)
    ygrid = np.asarray(body.y, dtype=np.float64)
    torso = np.interp(y, ygrid, _radius_profile(body, theta, "envelope"))
    wrapped = np.interp(y, ygrid, _radius_profile(body, theta, "wrap_r"))

    # Only retain the wing/root bulge close to the shoulder.  Following the
    # entire wrap envelope is what created the old box/shell silhouette.
    root_depth = np.clip((top_y - y) / max(0.22 * H, 1e-9), 0.0, 1.0)
    root_gate = 1.0 - _smoothstep(root_depth)
    bulge = np.clip(wrapped - torso, 0.0, 0.095 * H)
    r = torso + root_gate * bulge

    # A real shoulder may move outward quickly, but it should be allowed to
    # taper back toward the torso as it descends.
    inward_allow = 0.085 * H * _smoothstep(root_depth)
    r = np.maximum(r, top_r - inward_allow)
    return r


def initial_positions(pattern, body, flare_deg: float = -2.5) -> np.ndarray:
    """Draft a shoulder-supported tunic seed.

    The upper chart follows the measured neck/shoulder/root surface.  Only
    after crossing that support does it become a hanging panel, so the solver
    begins with a shoulder line instead of trying to discover one from a
    cylindrical shell by collision.
    """
    rows, cols = pattern.grid_shape
    if pattern.theta_cols is not None:
        theta = np.asarray(pattern.theta_cols, dtype=np.float64)
    else:
        theta = -np.pi + 2.0 * np.pi * np.asarray(pattern.phi[0], dtype=np.float64)

    top_y = np.interp(theta, body.theta, body.neck_y, period=2.0 * np.pi)
    top_r = np.interp(theta, body.theta, body.neck_r, period=2.0 * np.pi)
    H = float(body.size_y)
    axis = np.asarray(body.axis_xz, dtype=np.float64)
    P = np.zeros((rows, cols, 3), dtype=np.float64)

    support_target = max(0.15 * H, 0.23 * float(pattern.length))
    hang_radial_slope = np.tan(np.radians(float(flare_deg)))

    for c in range(cols):
        y1 = max(float(body.y_hem), float(top_y[c] - 0.32 * H))
        yy = np.linspace(float(top_y[c]), y1, 160)
        rr = _surface_radius(body, float(theta[c]), yy, float(top_y[c]), float(top_r[c]))

        shoulder_progress = _smoothstep((float(top_y[c]) - yy) / max(0.18 * H, 1e-9))
        rr += (0.008 * H) * shoulder_progress
        rr[0] = float(top_r[c])

        seg = np.hypot(np.diff(yy), np.diff(rr))
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        support_len = min(float(arc[-1]), support_target)
        if support_len <= 1e-8:
            support_len = min(float(pattern.length), 0.10 * H)

        y_end = float(np.interp(support_len, arc, yy))
        r_end = float(np.interp(support_len, arc, rr))
        vv = np.asarray(pattern.v[:, c], dtype=np.float64)
        on = vv <= support_len

        y = np.empty(rows, dtype=np.float64)
        r = np.empty(rows, dtype=np.float64)
        if np.any(on):
            y[on] = np.interp(vv[on], arc, yy)
            r[on] = np.interp(vv[on], arc, rr)
        if np.any(~on):
            extra = vv[~on] - support_len
            y[~on] = y_end - extra
            raw = r_end + hang_radial_slope * extra
            # The hanging panel may taper, but never seed inside the torso.
            floor = np.interp(y[~on], np.asarray(body.y, dtype=np.float64),
                              _radius_profile(body, float(theta[c]), "envelope")) + 0.022 * H
            r[~on] = np.maximum(raw, floor)

        P[:, c, 0] = axis[0] + r * np.sin(theta[c])
        P[:, c, 1] = y
        P[:, c, 2] = axis[1] + r * np.cos(theta[c])

    if cols > 1 and np.isclose(float(pattern.phi[0, -1] - pattern.phi[0, 0]), 1.0, atol=1e-5):
        P[:, -1] = P[:, 0]
    return P.reshape(rows * cols, 3)


def drape(P, pattern, sets, colliders, pinned, keep, params, offset=None,
          tethers=None, iterations: int | None = None, bounds: Sequence = (),
          periodic: bool = True):
    from meshforge import drape as D

    P = np.asarray(P, dtype=np.float64).copy()
    guide = P.copy()
    n = len(P)
    pinned = np.asarray(pinned, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    free = keep & ~pinned
    inv_mass = np.where(free, 1.0, 0.0)
    V = np.zeros_like(P)
    acc = np.empty_like(P)
    cnt = np.empty(n)
    iterations = int(params.iterations if iterations is None else iterations)
    rows, cols = pattern.grid_shape
    seam_src, seam_dup = D.seam_columns(rows, cols, keep, periodic)
    P[seam_dup] = P[seam_src]
    guide[seam_dup] = guide[seam_src]

    vnorm = np.asarray(pattern.v, dtype=np.float64).ravel() / max(float(pattern.length), 1e-9)
    yoke_t = np.clip(vnorm / 0.34, 0.0, 1.0)
    yoke_w = 1.0 - _smoothstep(yoke_t)
    yoke_gain = 0.56 * yoke_w
    scale = max(float(pattern.length), 1e-6)
    yoke_cap = scale * (0.018 + 0.095 * yoke_t)
    yoke_live = free & (yoke_w > 1e-4)

    self_radius = float(params.self_collide)
    if self_radius <= 0:
        self_radius = E._auto_self_radius(sets, params.stretch, n)

    contacts_total = 0
    self_pairs_total = 0
    for step_i in range(iterations + params.settle_iterations):
        settling = step_i >= iterations
        g = 0.0 if settling else params.gravity
        V[:, 1] += g * params.dt
        V *= params.damping if not settling else 0.5
        V[~free] = 0.0
        Pn = P + V * params.dt
        Pn[~free] = guide[~free]

        normal_acc = np.zeros_like(P)
        normal_count = np.zeros(n, dtype=np.int32)
        off = params.collide_offset if offset is None else offset

        for _ in range(params.inner_iterations):
            acc[:] = 0.0
            cnt[:] = 0.0
            for cs in sets:
                D._project(Pn, cs, inv_mass, acc, cnt)
            moved = cnt > 0
            Pn[moved] += acc[moved] / cnt[moved][:, None]
            Pn[~free] = guide[~free]
            if tethers is not None:
                D.apply_tethers(Pn, tethers)

            if np.any(yoke_live):
                Pn[yoke_live] += yoke_gain[yoke_live, None] * (guide[yoke_live] - Pn[yoke_live])
                delta = Pn - guide
                dist = np.linalg.norm(delta, axis=1)
                too = yoke_live & (dist > yoke_cap) & (dist > 1e-12)
                if np.any(too):
                    Pn[too] = guide[too] + delta[too] * (yoke_cap[too] / dist[too])[:, None]

            Pn[seam_dup] = Pn[seam_src]
            for col in colliders:
                hit, nrm = E._contact_project(Pn, col, off, free)
                if hit.any():
                    contacts_total += int(hit.sum())
                    normal_acc[hit] += nrm[hit]
                    normal_count[hit] += 1
            for b in bounds:
                b.apply(Pn, free=free)
            Pn[seam_dup] = Pn[seam_src]

        if self_radius > 0 and (step_i % 4 == 0 or settling):
            self_pairs_total += E._separate_topology(Pn, free, self_radius, rows, cols, periodic)
            for col in colliders:
                hit, nrm = E._contact_project(Pn, col, off, free)
                if hit.any():
                    contacts_total += int(hit.sum())
                    normal_acc[hit] += nrm[hit]
                    normal_count[hit] += 1
            for b in bounds:
                b.apply(Pn, free=free)

        Pn[seam_dup] = Pn[seam_src]
        V = (Pn - P) / max(params.dt, 1e-9)
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
        "solver": "shoulder-contact-exec-v3",
        "yoke_fraction": 0.34,
    }
