"""Hand-authored keyframe clips — explicit sampled quaternion rotations
baked into the export, not a runtime procedural transform. See spec
"Problem": the whole point of this pipeline is a real animation clip in
the GLB, playable by any glTF-compliant viewer's AnimationMixer.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Keyframe(NamedTuple):
    time: float
    rotation: np.ndarray  # quaternion [x, y, z, w]


def _axis_angle_to_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
        np.cos(half),
    ])


def _eased_bump_keyframes(
    axis: np.ndarray, degrees: float, t_start: float, duration_s: float, keyframe_count: int,
) -> list[Keyframe]:
    """Shared shape behind wave_clip and the blink portion of
    blink_clip: a rest -> peak -> rest bump over [t_start, t_start +
    duration_s], eased with a sine curve so motion decelerates at both
    ends instead of moving at constant speed."""
    times = np.linspace(t_start, t_start + duration_s, keyframe_count)
    # phase runs 0 -> pi over the clip, so sin(phase) rises from 0 to a
    # peak of 1.0 at the midpoint and back to 0 — a single bump.
    phase = np.linspace(0.0, np.pi, keyframe_count)
    eased_angle_deg = degrees * np.sin(phase)
    return [
        Keyframe(time=float(t), rotation=_axis_angle_to_quaternion(axis, np.radians(a)))
        for t, a in zip(times, eased_angle_deg)
    ]


def wave_clip(
    pivot_axis: np.ndarray,
    degrees: float = 35.0,
    duration_s: float = 1.2,
    keyframe_count: int = 8,
) -> list[Keyframe]:
    """A rest -> peak -> rest wave cycle, eased with a sine curve so the
    motion decelerates at both ends instead of moving at constant speed.
    """
    if keyframe_count < 2:
        raise ValueError(f"wave_clip: keyframe_count must be >= 2, got {keyframe_count}")
    if duration_s <= 0:
        raise ValueError(f"wave_clip: duration_s must be > 0, got {duration_s}")
    if degrees <= 0:
        raise ValueError(f"wave_clip: degrees must be > 0, got {degrees}")
    if np.linalg.norm(pivot_axis) < 1e-10:
        raise ValueError("wave_clip: pivot_axis must be a nonzero vector")

    return _eased_bump_keyframes(pivot_axis, degrees, 0.0, duration_s, keyframe_count)


def blink_clip(
    pivot_axis: np.ndarray,
    close_degrees: float,
    blink_duration_s: float = 0.24,
    hold_open_s: float = 3.4,
    keyframe_count: int = 6,
) -> list[Keyframe]:
    """A quick rest -> closed -> rest blink, then a long held-open plateau
    before the clip's end — so looping this clip reads as "mostly open,
    occasional blink" rather than blinking continuously back-to-back.
    `close_degrees` should be the exact angle that rotates the eyelid's
    authored rest pose onto full pupil coverage (see
    meshforge.eyes.eyelid_blink_axis_angle) — too little and the eye
    never fully shuts, too much and it overshoots past the pupil.
    """
    if keyframe_count < 2:
        raise ValueError(f"blink_clip: keyframe_count must be >= 2, got {keyframe_count}")
    if blink_duration_s <= 0:
        raise ValueError(f"blink_clip: blink_duration_s must be > 0, got {blink_duration_s}")
    if hold_open_s <= 0:
        raise ValueError(f"blink_clip: hold_open_s must be > 0, got {hold_open_s}")
    if close_degrees <= 0:
        raise ValueError(f"blink_clip: close_degrees must be > 0, got {close_degrees}")
    if np.linalg.norm(pivot_axis) < 1e-10:
        raise ValueError("blink_clip: pivot_axis must be a nonzero vector")

    blink = _eased_bump_keyframes(pivot_axis, close_degrees, 0.0, blink_duration_s, keyframe_count)
    hold = Keyframe(
        time=blink_duration_s + hold_open_s,
        rotation=_axis_angle_to_quaternion(pivot_axis, 0.0),
    )
    return blink + [hold]
