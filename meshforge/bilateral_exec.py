from __future__ import annotations

import dataclasses

import numpy as np


def symmetric_opening_mask(mask: np.ndarray, theta_cols: np.ndarray | None) -> np.ndarray:
    """Mirror arm-opening evidence across the owl's sagittal plane."""
    out = np.asarray(mask, dtype=bool).copy()
    if theta_cols is None or out.ndim != 2 or out.shape[1] < 2:
        return out
    th = np.asarray(theta_cols, dtype=np.float64)
    n = out.shape[1] - 1
    if len(th) != out.shape[1] or n <= 0:
        return out
    base = th[:n]
    target = ((-base + np.pi) % (2.0 * np.pi)) - np.pi
    d = np.abs(((base[None, :] - target[:, None] + np.pi) % (2.0 * np.pi)) - np.pi)
    mirror = np.argmin(d, axis=1)
    core = out[:, :n]
    core = core | core[:, mirror]
    out[:, :n] = core
    out[:, -1] = core[:, 0]
    return out


def _centre_x(joints) -> float:
    vals = [float(j.pivot[0]) for j in joints if j.name in {"body", "chest"}]
    if vals:
        return float(np.mean(vals))
    return 0.0


def _joint_map(joints):
    return {j.name: j for j in joints}


def _joint_remap(joints) -> np.ndarray:
    names = [j.name for j in joints]
    index = {n: i for i, n in enumerate(names)}
    remap = np.arange(len(joints), dtype=np.int64)
    if "wing_left" in index and "wing_right" in index:
        remap[index["wing_left"]] = index["wing_right"]
    if "wing_left_tip" in index:
        remap[index["wing_left_tip"]] = index.get("wing_right_tip", index.get("wing_right", index["wing_left_tip"]))
    return remap


def _collapse_top4(joints: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    J = np.zeros_like(joints, dtype=np.uint8)
    W = np.zeros_like(weights, dtype=np.float32)
    for r in range(len(joints)):
        acc: dict[int, float] = {}
        for j, w in zip(joints[r], weights[r]):
            if float(w) <= 0.0:
                continue
            acc[int(j)] = acc.get(int(j), 0.0) + float(w)
        items = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:4]
        s = sum(w for _, w in items)
        if s <= 1e-12:
            continue
        for k, (j, w) in enumerate(items):
            J[r, k] = np.uint8(j)
            W[r, k] = np.float32(w / s)
    return J, W


def _mirror_point(p: np.ndarray, x0: float) -> np.ndarray:
    q = np.asarray(p, dtype=np.float64).copy()
    q[0] = 2.0 * x0 - q[0]
    return q


def _right_shoulder_registration(joints, x0: float) -> np.ndarray:
    by_name = _joint_map(joints)
    left = by_name.get("wing_left")
    right = by_name.get("wing_right")
    if left is None or right is None:
        return np.zeros(3, dtype=np.float64)
    reflected_left = _mirror_point(np.asarray(left.pivot, dtype=np.float64), x0)
    delta = np.asarray(right.pivot, dtype=np.float64) - reflected_left
    # Preserve the concept's bilateral silhouette while accepting small rig
    # asymmetries at the actual shoulder socket.  Large offsets would mean
    # the labels are not mirror counterparts and should not drag the sleeve.
    if np.linalg.norm(delta) > 0.18:
        return np.zeros(3, dtype=np.float64)
    return delta


def mirror_left_sleeve(primitives, joints):
    """Append a right sleeve, registered to the real right shoulder joint."""
    if any(p.name == "kente_sleeve_right" for p in primitives):
        return list(primitives)
    src = next((p for p in primitives if p.name == "kente_sleeve"), None)
    if src is None:
        return list(primitives)

    x0 = _centre_x(joints)
    V = np.asarray(src.vertices, dtype=np.float64).copy()
    V[:, 0] = 2.0 * x0 - V[:, 0]
    V += _right_shoulder_registration(joints, x0)[None, :]

    N = np.asarray(src.normals, dtype=np.float64).copy()
    N[:, 0] *= -1.0
    F = np.asarray(src.faces, dtype=np.int64)[:, [0, 2, 1]].copy()
    UV = None if src.uvs is None else np.asarray(src.uvs, dtype=np.float64).copy()
    C = None if src.colors is None else np.asarray(src.colors).copy()

    J = None
    W = None
    if src.joints is not None and src.weights is not None:
        mapping = _joint_remap(joints)
        raw_j = mapping[np.asarray(src.joints, dtype=np.int64)]
        J, W = _collapse_top4(raw_j, np.asarray(src.weights, dtype=np.float64))

    mirrored = dataclasses.replace(
        src,
        name="kente_sleeve_right",
        vertices=V,
        faces=F,
        normals=N,
        uvs=UV,
        colors=C,
        joints=J,
        weights=W,
    )
    return [*primitives, mirrored]
