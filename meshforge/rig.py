"""Stage-B rigid two-node hierarchy: body (static) + one articulated part
attached at a pivot, no skin weights. See the spec's "Out of scope" section
for why full-body skinning (stage A) and blended-weight boundaries (stage C)
are deliberately not implemented here.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh


class RigNode(NamedTuple):
    name: str
    mesh: trimesh.Trimesh | None
    translation: np.ndarray
    children: list["RigNode"]


def build_rig(body: trimesh.Trimesh, part: trimesh.Trimesh, pivot_point: np.ndarray) -> RigNode:
    """Build a root -> {body, wing_pivot -> part} rigid node hierarchy.

    The part mesh is re-expressed in the wing_pivot node's local space
    (vertices minus pivot_point) so that rotating the wing_pivot node
    rotates the part about the pivot, matching glTF's node-local rotation
    convention.
    """
    if part is None:
        raise ValueError("build_rig: part mesh must not be None")
    pivot_point = np.asarray(pivot_point)
    if pivot_point.shape != (3,):
        raise ValueError(
            f"build_rig: pivot_point must have shape (3,), got {pivot_point.shape}"
        )

    part_local = part.copy()
    part_local.vertices = part_local.vertices - pivot_point

    body_node = RigNode(
        name="body",
        mesh=body,
        translation=np.zeros(3),
        children=[],
    )
    wing_node = RigNode(
        name="wing_pivot",
        mesh=part_local,
        translation=np.asarray(pivot_point, dtype=np.float64),
        children=[],
    )
    return RigNode(
        name="root",
        mesh=None,
        translation=np.zeros(3),
        children=[body_node, wing_node],
    )
