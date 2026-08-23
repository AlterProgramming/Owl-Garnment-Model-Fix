"""Forward kinematics for the exported joint convention, shared by the
preview renderer and the validator so "what the preview shows" and "what
the exported skin does" are the same arithmetic.

Convention (see export.write_rigged_glb): every joint has identity rest
rotation; its local translation is its pivot minus its parent's pivot;
the inverse-bind matrix is translate(-pivot). An animation may add a
local rotation (quaternion, xyzw) and a local translation offset.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation


class Joint(NamedTuple):
    name: str
    parent: str | None
    pivot: np.ndarray  # (3,) world-space pivot in the bind pose


def ordered_joints(joints: list[Joint]) -> list[Joint]:
    """Parents before children (stable otherwise)."""
    by_name = {j.name: j for j in joints}
    for j in joints:
        if j.parent is not None and j.parent not in by_name:
            raise ValueError(f"ordered_joints: {j.name!r} has unknown parent {j.parent!r}")
    out: list[Joint] = []
    done: set[str] = set()
    remaining = list(joints)
    while remaining:
        progressed = False
        for j in list(remaining):
            if j.parent is None or j.parent in done:
                out.append(j)
                done.add(j.name)
                remaining.remove(j)
                progressed = True
        if not progressed:
            raise ValueError(f"ordered_joints: cycle among {[j.name for j in remaining]}")
    return out


def _rt(rotation_xyzw: np.ndarray | None, translation: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    if rotation_xyzw is not None:
        m[:3, :3] = Rotation.from_quat(rotation_xyzw).as_matrix()
    m[:3, 3] = translation
    return m


def local_rest_translation(joint: Joint, by_name: dict[str, Joint]) -> np.ndarray:
    if joint.parent is None:
        return np.asarray(joint.pivot, dtype=np.float64)
    return np.asarray(joint.pivot, dtype=np.float64) - np.asarray(by_name[joint.parent].pivot, dtype=np.float64)


def world_matrices(
    joints: list[Joint],
    rotations: dict[str, np.ndarray] | None = None,
    translations: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """World matrix per joint for a pose given as per-joint local rotation
    (xyzw) and local translation *offset* (added to the rest translation)."""
    rotations = rotations or {}
    translations = translations or {}
    by_name = {j.name: j for j in joints}
    world: dict[str, np.ndarray] = {}
    for j in ordered_joints(joints):
        t = local_rest_translation(j, by_name) + np.asarray(translations.get(j.name, np.zeros(3)), dtype=np.float64)
        local = _rt(rotations.get(j.name), t)
        world[j.name] = local if j.parent is None else world[j.parent] @ local
    return world


def inverse_bind(joint: Joint) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = -np.asarray(joint.pivot, dtype=np.float64)
    return m


def skin_matrices(joints: list[Joint], world: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {j.name: world[j.name] @ inverse_bind(j) for j in joints}


def pose_vertices(
    vertices: np.ndarray,
    weights: np.ndarray,
    joint_names: list[str],
    skin: dict[str, np.ndarray],
) -> np.ndarray:
    """Linear blend skinning: p' = sum_j w_j * (S_j p)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(vertices), len(joint_names)):
        raise ValueError(f"pose_vertices: weights must be ({len(vertices)}, {len(joint_names)}), got {weights.shape}")
    out = np.zeros_like(vertices)
    homo = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    for j, name in enumerate(joint_names):
        w = weights[:, j]
        active = w > 0
        if not active.any():
            continue
        S = skin[name]
        moved = (homo[active] @ S.T)[:, :3]
        out[active] += w[active, None] * moved
    return out


def pose_normals(normals: np.ndarray, weights: np.ndarray, joint_names: list[str],
                 skin: dict[str, np.ndarray]) -> np.ndarray:
    """Rotate normals with the same blend (rotation-only parts; our joints
    never scale, so the inverse-transpose equals the rotation)."""
    normals = np.asarray(normals, dtype=np.float64)
    out = np.zeros_like(normals)
    for j, name in enumerate(joint_names):
        w = weights[:, j]
        active = w > 0
        if not active.any():
            continue
        R = skin[name][:3, :3]
        out[active] += w[active, None] * (normals[active] @ R.T)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms
