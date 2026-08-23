"""Blended per-vertex skin weights for a two-bone (body/wing) skeleton.

Where segment.py's rigid PartResult treats the wing cut as a hard boundary
(patched afterwards by `_fill_body_hole`), this module computes a smooth
weight ramp across that same seam directly on the full, uncropped mesh —
the real fix for "cropping leaves an open hole once the part rotates
away" (reported directly against this pipeline's shipped output): with
blended weights the seam is never actually severed, so there is nothing
for a rotated part to expose.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def compute_skin_weights(
    mesh: trimesh.Trimesh,
    part_face_mask: np.ndarray,
    blend_rings: int = 2,
    blend_rounds: int = 6,
) -> np.ndarray:
    """Return a (len(mesh.vertices),) array of wing-bone weights in [0, 1].

    1.0 = fully driven by the wing bone, 0.0 = fully driven by the body
    bone. Vertices used only by part faces start at 1.0, vertices used
    only by body faces start at 0.0, and vertices on the seam (used by
    both) start at an even 0.5 blend. The seam plus a `blend_rings`-wide
    band of graph neighbors around it are then smoothed for
    `blend_rounds` iterations of neighbor averaging, so the transition
    ramps across a few rows of triangles instead of jumping straight from
    0 to 1 at the cut. Vertices outside that band never move off their
    starting 0/1 weight — they anchor the smoothing.
    """
    part_face_mask = np.asarray(part_face_mask, dtype=bool)
    if part_face_mask.shape != (len(mesh.faces),):
        raise ValueError(
            f"compute_skin_weights: part_face_mask must have shape "
            f"({len(mesh.faces)},), got {part_face_mask.shape}"
        )

    n = len(mesh.vertices)
    part_vertex_mask = np.zeros(n, dtype=bool)
    part_vertex_mask[mesh.faces[part_face_mask].ravel()] = True
    body_vertex_mask = np.zeros(n, dtype=bool)
    body_vertex_mask[mesh.faces[~part_face_mask].ravel()] = True
    seam_mask = part_vertex_mask & body_vertex_mask

    weight = np.where(part_vertex_mask, 1.0, 0.0)

    if not seam_mask.any():
        # part and body never share a vertex (e.g. a pre-existing separate
        # shell like the cap) — nothing to blend, a clean binary split is
        # already correct.
        return weight

    weight[seam_mask] = 0.5

    adjacency = defaultdict(set)
    for a, b in mesh.edges_unique:
        adjacency[a].add(int(b))
        adjacency[b].add(int(a))

    movable = seam_mask.copy()
    frontier = set(int(v) for v in np.where(seam_mask)[0])
    for _ in range(blend_rings):
        next_frontier = set()
        for v in frontier:
            next_frontier.update(adjacency[v])
        if next_frontier:
            movable[list(next_frontier)] = True
        frontier = next_frontier

    movable_idx = np.where(movable)[0]
    for _ in range(blend_rounds):
        new_weight = weight.copy()
        for v in movable_idx:
            neighbors = adjacency[v]
            if neighbors:
                new_weight[v] = float(np.mean([weight[nb] for nb in neighbors]))
        weight = new_weight

    return weight


def pose_vertices_lbs(
    vertices: np.ndarray,
    wing_weight: np.ndarray,
    pivot_point: np.ndarray,
    rotation_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Linear-blend-skin `vertices` with a two-joint skeleton: the body
    joint is identity, the wing joint rotates by `rotation_quat_xyzw`
    about `pivot_point`. This is the standard 2-influence glTF skinning
    formula a real viewer's skinning shader computes from
    JOINTS_0/WEIGHTS_0, so what the preview renders and what the exported
    skin actually does are the same computation.
    """
    rot = Rotation.from_quat(rotation_quat_xyzw)
    wing_pose = rot.apply(vertices - pivot_point) + pivot_point
    w = wing_weight[:, None]
    return (1.0 - w) * vertices + w * wing_pose


def compute_multi_part_skin_weights(
    mesh: trimesh.Trimesh,
    part_face_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Generalizes compute_skin_weights to N independently-blended named
    parts (e.g. both wings). Each part's blend band is computed exactly
    like the single-part case, treating "not this part" as body — this
    is only correct if parts are spatially far enough apart that their
    blend bands never touch (true for two wings on opposite sides of a
    body). Weights are defensively renormalized so they always sum to 1
    per vertex even if that assumption is ever violated: any part-weight
    overlap exceeding 1 at a vertex is scaled down proportionally across
    the involved parts, with body taking whatever is left (never
    negative) rather than going negative itself.

    Returns a dict with a "body" key plus one key per entry in
    `part_face_masks`.
    """
    part_weights = {name: compute_skin_weights(mesh, mask) for name, mask in part_face_masks.items()}

    total = np.sum(list(part_weights.values()), axis=0)
    # np.where evaluates both branches eagerly, so guard the denominator
    # (never used when total <= 1.0, but would otherwise divide by zero
    # for pure-body vertices where every part's weight is 0).
    safe_total = np.maximum(total, 1.0)
    scale = np.where(total > 1.0, 1.0 / safe_total, 1.0)
    for name in part_weights:
        part_weights[name] = part_weights[name] * scale

    total_scaled = np.sum(list(part_weights.values()), axis=0)
    body_weight = np.clip(1.0 - total_scaled, 0.0, None)

    return {"body": body_weight, **part_weights}


def pose_vertices_multi_lbs(
    vertices: np.ndarray,
    joint_weights: dict[str, np.ndarray],
    joint_pivots: dict[str, np.ndarray],
    joint_rotations: dict[str, np.ndarray],
) -> np.ndarray:
    """N-joint generalization of pose_vertices_lbs. `joint_weights` must
    include a "body" entry (the identity joint — its rotation is never
    consulted). Every other key names a part joint: its vertices pose as
    `rotate(vertices - pivot, rotation) + pivot`, weighted and summed
    with every other joint's contribution, matching the standard
    multi-influence linear-blend-skinning formula:
    p' = sum_j weight_j * (rotate_j(p - pivot_j) + pivot_j), with the
    body joint's rotation fixed at identity so its term reduces to
    `weight_body * p`.
    """
    result = joint_weights["body"][:, None] * vertices
    for name, weight in joint_weights.items():
        if name == "body":
            continue
        rot = Rotation.from_quat(joint_rotations[name])
        posed = rot.apply(vertices - joint_pivots[name]) + joint_pivots[name]
        result = result + weight[:, None] * posed
    return result
