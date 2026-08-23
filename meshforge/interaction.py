"""Deterministic interaction contracts used by the interaction lab.

These helpers intentionally keep render transforms, attachment transforms,
and simple physics state separate. They are small enough to run in tests and
are the reference equations used by the runtime adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


class AnchorContractError(ValueError):
    """Raised when a cutscene is missing a required named anchor."""


@dataclass(frozen=True)
class SphereState:
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("SphereState position and velocity must both have shape (3,)")
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("SphereState position and velocity must be finite")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


def _matrix4(value: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be finite")
    return matrix


def attach_preserve_world_transform(
    parent_world: np.ndarray,
    socket_local: np.ndarray,
    object_world: np.ndarray,
) -> np.ndarray:
    """Return the child-local transform for an object attached to a socket.

    The invariant is ``parent_world @ socket_local @ result == object_world``.
    Computing this once at attach time avoids a visible snap and leaves the
    runtime free to update the parent/socket every frame.
    """
    parent = _matrix4(parent_world, "parent_world")
    socket = _matrix4(socket_local, "socket_local")
    world = _matrix4(object_world, "object_world")
    return np.linalg.solve(parent @ socket, world)


def hinge_point(
    point: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    """Rotate one point around an authored hinge axis using Rodrigues' formula."""
    point = np.asarray(point, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if point.shape != (3,) or pivot.shape != (3,) or axis.shape != (3,):
        raise ValueError("point, pivot, and axis must all have shape (3,)")
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("hinge axis must be nonzero")
    if not np.isfinite([*point, *pivot, *axis, angle_radians]).all():
        raise ValueError("hinge inputs must be finite")

    axis = axis / length
    relative = point - pivot
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    rotated = (
        relative * cosine
        + np.cross(axis, relative) * sine
        + axis * np.dot(axis, relative) * (1.0 - cosine)
    )
    return pivot + rotated


def step_sphere(
    state: SphereState,
    *,
    dt: float,
    gravity: float,
    floor_y: float,
    radius: float,
    restitution: float,
) -> SphereState:
    """Advance a sphere against one infinite horizontal floor.

    This is deliberately a deterministic control fixture, not a replacement
    for the engine solver. It demonstrates the authority boundary: physics
    owns the sphere transform, and the render mesh follows that state.
    """
    if dt <= 0 or radius < 0 or not 0 <= restitution <= 1:
        raise ValueError("dt must be positive, radius nonnegative, restitution in [0, 1]")
    values = np.array([dt, gravity, floor_y, radius, restitution], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("sphere step inputs must be finite")

    acceleration = np.array([0.0, gravity, 0.0], dtype=np.float64)
    velocity = state.velocity + acceleration * dt
    position = state.position + state.velocity * dt + 0.5 * acceleration * dt * dt
    floor_center = floor_y + radius
    if position[1] < floor_center:
        position[1] = floor_center
        if velocity[1] < 0:
            velocity[1] = -velocity[1] * restitution
    return SphereState(position=position, velocity=velocity)


def validate_anchor_contract(
    anchors: Mapping[str, Iterable[float]],
    required: Iterable[str],
) -> dict[str, np.ndarray]:
    """Validate and normalize the named anchors required by a cutscene."""
    missing = [name for name in required if name not in anchors]
    if missing:
        raise AnchorContractError(f"missing required anchors: {', '.join(missing)}")
    normalized: dict[str, np.ndarray] = {}
    for name in required:
        value = np.asarray(anchors[name], dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise AnchorContractError(f"anchor {name!r} must be a finite 3-vector")
        normalized[name] = value
    return normalized
