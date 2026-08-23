"""Acceptance gates for the wrap (spec §6): numbers with thresholds, computed
by the build and printed as a table. A failing gate does not stop the build;
it blocks promotion. G9 (the gray sheet) and G10 (guide size) are not here —
one is the user's eye, the other is `ls -l`."""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from meshforge import fk
from meshforge import robe as robemod

FORBIDDEN: tuple[str, ...] = ("neck", "head", "cap", "tassel", "eye_left", "eye_right", "lid_left", "lid_right",
                              "wing_left", "wing_left_tip", "wing_right", "leg_left", "leg_right",
                              "foot_left", "foot_right", "tail")


class Gate(NamedTuple):
    id: str
    name: str
    value: float
    threshold: str
    passed: bool
    detail: dict


def clip_skins(joints, joint_names, clips, samples: int = 5) -> list[tuple[str, float, dict]]:
    out = []
    for anim in clips:
        dur = max(float(np.max(t.times)) for t in anim.tracks) if anim.tracks else 0.0
        for tm in np.linspace(0.0, dur, samples):
            rot, tr = robemod.clip_pose_at(anim, tm)
            skin = fk.skin_matrices(joints, fk.world_matrices(joints, rotations=rot, translations=tr))
            out.append((anim.name, float(tm), skin))
    return out


def gate_worn(pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H) -> Gate:
    """G1: the pinned row moves with the torso skin it is pinned to. Anchors
    are the nearest *torso* vertices (the loop passes under the fused wing;
    the wing lifting off it is not the cloth coming off the owl)."""
    pins = np.asarray(pins, dtype=np.float64)
    body_V = np.asarray(body_V, dtype=np.float64)
    torso_idx = np.flatnonzero(np.asarray(torso_mask, dtype=bool))
    near = torso_idx[cKDTree(body_V[torso_idx]).query(pins, workers=-1)[1]]
    anchors, aW = body_V[near], np.asarray(body_W, dtype=np.float64)[near].copy()
    # the torso's own blend of chest and body is the reference; the share
    # of wing or neck a torso vertex carries from the blend rings is not
    # the support moving, it is the limb next to it moving
    keep = np.array([nm in ("body", "chest") for nm in joint_names])
    aW[:, ~keep] = 0.0
    s = aW.sum(axis=1, keepdims=True)
    pin_W = np.asarray(pin_W, dtype=np.float64)
    aW = np.where(s > 1e-9, aW / np.maximum(s, 1e-9), pin_W)     # a torso vertex with no torso share: the pin's own
    rest = pins - anchors
    worst, where = 0.0, {}
    for name, tm, skin in clip_skins(joints, joint_names, clips):
        p = fk.pose_vertices(pins, pin_W, joint_names, skin)
        a = fk.pose_vertices(anchors, aW, joint_names, skin)
        drift = np.linalg.norm((p - a) - rest, axis=1) / H
        i = int(np.argmax(drift))
        if drift[i] > worst:
            worst, where = float(drift[i]), {"clip": name, "t": round(tm, 2), "column": i}
    return Gate("G1", "worn: pinned row drift from its torso support", round(worst, 4), "<= 0.010 H", worst <= 0.010, where)


def gate_support_only(W_all, joint_names) -> Gate:
    W_all = np.asarray(W_all)
    cols = [i for i, nm in enumerate(joint_names) if nm in FORBIDDEN]
    mass = float(W_all[:, cols].sum()) if cols else 0.0
    by = {joint_names[i]: round(float(W_all[:, i].sum()), 4) for i in cols if W_all[:, i].sum() > 0}
    return Gate("G2", "support only: weight on forbidden joints", round(mass, 4), "= 0", mass == 0.0, by)


def gate_penetration(posed: dict, limit: int = 20) -> Gate:
    worst, clip = 0, None
    for name, c in (posed.get("per_clip") or {}).items():
        if c and int(c.get("inside_count", 0)) >= worst:
            worst, clip = int(c.get("inside_count", 0)), name
    return Gate("G3", "no penetration: cloth vertices inside the body, worst clip", worst, f"<= {limit}", worst <= limit, {"clip": clip})


def gate_hem(hem_points, y0, H, bead_top_y) -> Gate:
    yf = (np.asarray(hem_points)[:, 1] - y0) / H
    floor = (bead_top_y - y0) / H + 0.02
    sigma = float(yf.std())
    ok = float(yf.min()) >= floor and sigma <= 0.015
    return Gate("G4", "hem: above the beads and level", round(float(yf.min()), 4), f">= {floor:.3f} H, sigma <= 0.015 H", ok,
                {"min": round(float(yf.min()), 4), "max": round(float(yf.max()), 4), "sigma": round(sigma, 4), "floor": round(floor, 4)})


def gate_root_covered(root_ring, axis_xz, garments) -> Gate:
    """G5: rays from the torso axis out through each root-ring vertex (what
    a viewer walking round the owl sees); the ring is covered where a ray
    meets cloth beyond the vertex. Rays from the *wing* axis pointed into
    the body for the ring's inner half and could never be covered."""
    ring = np.asarray(root_ring, dtype=np.float64)
    foot = np.stack([np.full(len(ring), float(axis_xz[0])), ring[:, 1], np.full(len(ring), float(axis_xz[1]))], axis=1)
    dirs = ring - foot
    dist = np.linalg.norm(dirs, axis=1)
    good = dist > 1e-6
    dirs[good] /= dist[good][:, None]
    covered = np.zeros(len(ring), dtype=bool)
    if garments and good.any():
        mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=np.asarray(V), faces=np.asarray(F), process=False)
                                         for V, F in garments])
        loc, idx, _ = mesh.ray.intersects_location(ray_origins=foot[good], ray_directions=dirs[good], multiple_hits=True)
        if len(idx):
            t = np.einsum("ij,ij->i", loc - foot[good][idx], dirs[good][idx])
            beyond = t > dist[good][idx] + 0.002
            hit = np.zeros(int(good.sum()), dtype=bool)
            np.logical_or.at(hit, idx[beyond], True)
            covered[good] = hit
    frac = float(covered.mean()) if len(ring) else 0.0
    detail = {"covered": int(covered.sum()), "ring": int(len(ring))}
    if (~covered).any():
        bare = ring[~covered]
        detail["bare_centroid"] = [round(float(x), 3) for x in bare.mean(axis=0)]
        detail["bare_y_range"] = [round(float(bare[:, 1].min()), 3), round(float(bare[:, 1].max()), 3)]
        detail["bare_x_range"] = [round(float(bare[:, 0].min()), 3), round(float(bare[:, 0].max()), 3)]
        detail["bare_z_range"] = [round(float(bare[:, 2].min()), 3), round(float(bare[:, 2].max()), 3)]
    return Gate("G5", "root covered: ring rays meeting cloth", round(frac, 3), ">= 0.90", frac >= 0.90, detail)


def gate_tail_clear(tail_V, tail_W, body_mesh, body_W, joints, joint_names, wave_clips) -> Gate:
    posed = robemod.posed_clearance(np.asarray(tail_V), np.asarray(tail_W), body_mesh, body_W, joints, joint_names, wave_clips)
    worst = max((int(c.get("inside_count", 0)) for c in posed["per_clip"].values() if c), default=0)
    mn = min((float(c.get("min", 0.0)) for c in posed["per_clip"].values() if c), default=0.0)
    return Gate("G6", "tail clear of the waving wing", worst, "= 0 inside", worst == 0, {"min": round(mn, 4)})


def gate_stretch(P, sets, stretch_k, cols: int | None = None) -> Gate:
    P = np.asarray(P, dtype=np.float64)
    ratios, firsts = [], []
    for s in sets:
        if abs(float(s.k) - float(stretch_k)) < 1e-9:
            L = np.linalg.norm(P[s.j] - P[s.i], axis=1)
            ratios.append(L / np.maximum(s.rest, 1e-9))
            firsts.append(np.asarray(s.i))
    r = np.concatenate(ratios) if ratios else np.ones(1)
    med, p90 = float(np.median(r)), float(np.percentile(r, 90))
    ok = 0.97 <= med <= 1.05 and p90 <= 1.12
    detail = {"median": round(med, 3), "p90": round(p90, 3), "p99": round(float(np.percentile(r, 99)), 3)}
    if cols and firsts:
        row = np.concatenate(firsts) // cols
        by_row = {int(k): round(float(np.percentile(r[row == k], 90)), 3) for k in np.unique(row)}
        worst = sorted(by_row.items(), key=lambda kv: -kv[1])[:5]
        detail["worst_rows_p90"] = {str(k): v for k, v in worst}
        col = np.concatenate(firsts) % cols
        by_col = {int(k): round(float(np.percentile(r[col == k], 90)), 3) for k in np.unique(col)}
        worst_c = sorted(by_col.items(), key=lambda kv: -kv[1])[:8]
        detail["worst_cols_p90"] = {str(k): v for k, v in worst_c}
    return Gate("G7", "stretch: warp/weft l/l0", round(med, 3), "median in [0.97, 1.05], p90 <= 1.12", ok, detail)


def gate_collar_visible(cloth_V, theta_grid, rim_y, axis_xz) -> Gate:
    """G8: no cloth above the collar band's measured lower rim at its angle
    (`rim_y` per scan angle, the same curve the tunic hung from)."""
    V = np.asarray(cloth_V, dtype=np.float64)
    theta = np.arctan2(V[:, 0] - axis_xz[0], V[:, 2] - axis_xz[1])
    rim = np.interp(theta, np.asarray(theta_grid), np.asarray(rim_y), period=2 * np.pi)
    above = V[:, 1] > rim
    inside = int(above.sum())
    detail = {"max_above": round(float((V[:, 1] - rim).max()), 4)}
    if above.any():
        detail["above_theta_deg"] = [round(float(np.degrees(theta[above]).min()), 1), round(float(np.degrees(theta[above]).max()), 1)]
        detail["above_index_range"] = [int(np.flatnonzero(above).min()), int(np.flatnonzero(above).max())]
    return Gate("G8", "collar visible: cloth vertices above the band's rim", inside, "= 0", inside == 0, detail)


def run_gates(*, pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H, W_all, posed,
              hem_points, y0, bead_top_y, root_ring, axis_xz, garments, tail_V, tail_W,
              body_mesh, wave_clips, sheet_P, sheet_sets, stretch_k, sheet_cols, cloth_V, theta_grid,
              rim_y) -> list[Gate]:
    return [
        gate_worn(pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H),
        gate_support_only(W_all, joint_names),
        gate_penetration(posed),
        gate_hem(hem_points, y0, H, bead_top_y),
        gate_root_covered(root_ring, axis_xz, garments),
        gate_tail_clear(tail_V, tail_W, body_mesh, body_W, joints, joint_names, wave_clips),
        gate_stretch(sheet_P, sheet_sets, stretch_k, cols=sheet_cols),
        gate_collar_visible(cloth_V, theta_grid, rim_y, axis_xz),
    ]


def format_table(gates: Sequence[Gate]) -> str:
    rows = [("gate", "measure", "value", "threshold", "verdict")]
    for g in gates:
        rows.append((g.id, g.name, f"{g.value:g}", g.threshold, "PASS" if g.passed else "FAIL"))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    lines = [" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    failed = [g.id for g in gates if not g.passed]
    lines.append(f"{len(gates) - len(failed)}/{len(gates)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)
