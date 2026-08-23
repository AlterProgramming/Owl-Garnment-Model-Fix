"""Baked animation clips for the OwlV1 skeleton.

Every clip is authored as continuous functions of time and sampled at a
fixed rate into rotation/translation tracks (meshforge.rigexport.Track),
so the GLB carries real keyframes any glTF runtime can play — the idle
breathing is in the file, not in viewer JavaScript.

Clips drive disjoint joint sets (see the design spec) so a runtime can
play several at once without per-property blending logic:

    idle         chest, body(translation), tail, tassel, wing_right
    wave         wing_left
    blink        lid_left, lid_right
    nod          head
    hop          body(translation), leg_*, foot_*
    tablet_show  wing_right        (runtime stops idle's wing_right sway while it plays)

Axis conventions (mesh faces +Z, +Y up, +X is the viewer's right):
  rotation about +X by +θ pitches the front DOWN (nod),
  rotation about +Z swings a raised −X wing tip down/outward for +θ and
  up/inward for −θ, rotation about +Y yaws the front toward −X.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from meshforge.rigexport import AnimationSpec, Track

FPS = 30


def quat(axis: str, degrees: float) -> np.ndarray:
    return Rotation.from_euler(axis, degrees, degrees=True).as_quat()


def compose(*rots: Rotation) -> np.ndarray:
    r = rots[0]
    for extra in rots[1:]:
        r = r * extra
    return r.as_quat()


def smoothstep(t: float | np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bump(t: float | np.ndarray) -> np.ndarray:
    """0 -> 1 -> 0 over t in [0, 1], eased (sine)."""
    t = np.clip(t, 0.0, 1.0)
    return np.sin(np.pi * t)


def sample_rotation(joint: str, duration: float, fn: Callable[[float], np.ndarray], fps: int = FPS) -> Track:
    times = np.linspace(0.0, duration, max(2, int(round(duration * fps)) + 1))
    values = np.stack([np.asarray(fn(float(t)), dtype=np.float64) for t in times])
    # keep quaternion sign continuous for clean LINEAR/slerp interpolation
    for i in range(1, len(values)):
        if np.dot(values[i], values[i - 1]) < 0:
            values[i] = -values[i]
    return Track(joint=joint, path="rotation", times=times, values=values)


def sample_translation(joint: str, duration: float, fn: Callable[[float], np.ndarray], fps: int = FPS) -> Track:
    times = np.linspace(0.0, duration, max(2, int(round(duration * fps)) + 1))
    values = np.stack([np.asarray(fn(float(t)), dtype=np.float64) for t in times])
    return Track(joint=joint, path="translation", times=times, values=values)


# --------------------------------------------------------------------------
# clips
# --------------------------------------------------------------------------

def idle_clip(duration: float = 4.0, breath_deg: float = 1.2, bob: float = 0.004,
              tail_deg: float = 4.0, tassel_deg: float = 7.0, tablet_deg: float = 1.5) -> AnimationSpec:
    """Loopable: every term is periodic in `duration`."""
    w = 2 * np.pi / duration

    def chest(t):
        return quat("x", -breath_deg * np.sin(w * t))  # inhale lifts the chest (front up)

    def body(t):
        return np.array([0.0, bob * np.sin(w * t - 0.4), 0.0])

    def tail(t):
        return quat("z", tail_deg * np.sin(2 * w * t + 0.7))

    def tassel(t):
        # pendulum: mostly side-to-side (about Z) with a little front/back (about X)
        return compose(Rotation.from_euler("z", tassel_deg * np.sin(2.5 * w * t), degrees=True),
                       Rotation.from_euler("x", 0.45 * tassel_deg * np.sin(2.5 * w * t + 1.2), degrees=True))

    def wing_right(t):
        return quat("x", tablet_deg * np.sin(w * t + 1.0))

    return AnimationSpec("idle", [
        sample_rotation("chest", duration, chest),
        sample_translation("body", duration, body),
        sample_rotation("tail", duration, tail),
        sample_rotation("tassel", duration, tassel),
        sample_rotation("wing_right", duration, wing_right),
    ])


def wave_clip(duration: float = 1.6, lift_deg: float = 22.0, swing_deg: float = 20.0, oscillations: int = 3,
              depth_deg: float = 6.0, tip_gain: float = 1.15, tip_lag: float = 0.075,
              tip_lift_deg: float = 16.0) -> AnimationSpec:
    """Raised wing: lift away from the body, oscillate, settle back.

    **Positive** Z swings the −X wing out and up; negative Z tucks it down
    against the chest. This clip had the sign inverted from the day it was
    written — it lowered the wing for the whole oscillation and returned it
    at the end, which still moved plenty of pixels, so both harnesses
    passed it every run. Only a frame-by-frame contact sheet of the widget
    showed the wing going the wrong way.

    The wing is driven by two joints. `wing_left` carries the swing;
    `wing_left_tip` repeats it scaled by `tip_gain` and delayed by
    `tip_lag` seconds, so the outer half arrives late and overshoots as
    the base reverses — overlapping action, the thing that separates a
    wave from a windscreen wiper. One joint cannot do this no matter how
    the weights are painted: every vertex it owns takes the same
    rotation."""
    t_up = 0.28 * duration
    t_down = 0.72 * duration

    def swing_at(t):
        if t < t_up:
            return lift_deg * smoothstep(max(t, 0.0) / t_up), 0.0
        if t < t_down:
            u = (t - t_up) / (t_down - t_up)
            return lift_deg, swing_deg * np.sin(2 * np.pi * oscillations * u) * np.sin(np.pi * u) ** 0.5
        return lift_deg * (1 - smoothstep(min((t - t_down) / (duration - t_down), 1.0))), 0.0

    def wing(t):
        lift, swing = swing_at(t)
        return compose(Rotation.from_euler("z", lift + swing, degrees=True),
                       Rotation.from_euler("x", depth_deg * np.sin(np.pi * min(t / duration, 1.0)), degrees=True))

    def tip(t):
        # the tip answers the base's motion `tip_lag` later, so its own
        # rotation is the *difference* between then and now
        lift_now, swing_now = swing_at(t)
        lift_then, swing_then = swing_at(t - tip_lag)
        d_lift = (lift_then - lift_now) * tip_gain
        d_swing = (swing_then - swing_now) * tip_gain
        curl = tip_lift_deg * np.sin(np.pi * min(max(t / duration, 0.0), 1.0))
        return compose(Rotation.from_euler("z", d_lift + d_swing + curl, degrees=True),
                       Rotation.from_euler("x", 0.5 * depth_deg * np.sin(np.pi * min(t / duration, 1.0)), degrees=True))

    return AnimationSpec("wave", [sample_rotation("wing_left", duration, wing),
                                  sample_rotation("wing_left_tip", duration, tip)])


def blink_clip(lid_axes: dict[str, np.ndarray], close_deg: float, duration: float = 0.26) -> AnimationSpec:
    """One blink: rest -> closed -> rest, faster down than up (real lids
    close in ~1/3 of the blink and open in the remaining 2/3).
    `lid_axes` maps each lid joint to the rotation axis that closes it
    (meshforge.eyes2.EyeSet.blink_axes)."""
    t_closed = 0.38 * duration

    def make(axis):
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        def lid(t):
            if t < t_closed:
                a = smoothstep(t / t_closed)
            else:
                a = 1 - smoothstep((t - t_closed) / (duration - t_closed))
            return Rotation.from_rotvec(axis * np.radians(close_deg * a)).as_quat()
        return lid

    return AnimationSpec("blink", [sample_rotation(name, duration, make(axis)) for name, axis in lid_axes.items()])


def nod_clip(duration: float = 0.9, down_deg: float = 12.0, overshoot_deg: float = 4.0) -> AnimationSpec:
    def head(t):
        u = t / duration
        if u < 0.35:
            a = down_deg * smoothstep(u / 0.35)
        elif u < 0.7:
            a = down_deg - (down_deg + overshoot_deg) * smoothstep((u - 0.35) / 0.35)
        else:
            a = -overshoot_deg * (1 - smoothstep((u - 0.7) / 0.3))
        return quat("x", a)

    return AnimationSpec("nod", [sample_rotation("head", duration, head)])


def hop_clip(duration: float = 0.7, height: float = 0.07, tuck_deg: float = 14.0, anticipation: float = 0.18) -> AnimationSpec:
    """Crouch slightly, jump, tuck the legs, land."""
    def body(t):
        u = t / duration
        if u < anticipation:
            y = -0.35 * height * bump(u / anticipation)
        else:
            v = (u - anticipation) / (1 - anticipation)
            y = height * 4 * v * (1 - v)  # parabola
        return np.array([0.0, y, 0.0])

    def leg(sign):
        def fn(t):
            u = t / duration
            if u < anticipation:
                return quat("x", 0.0)
            v = (u - anticipation) / (1 - anticipation)
            return quat("x", sign * tuck_deg * bump(v))
        return fn

    def foot(t):
        u = t / duration
        if u < anticipation:
            return quat("x", 0.0)
        v = (u - anticipation) / (1 - anticipation)
        return quat("x", -0.8 * tuck_deg * bump(v))  # toes point down in the air

    return AnimationSpec("hop", [
        sample_translation("body", duration, body),
        sample_rotation("leg_left", duration, leg(+1)),
        sample_rotation("leg_right", duration, leg(+1)),
        sample_rotation("foot_left", duration, foot),
        sample_rotation("foot_right", duration, foot),
    ])


def tablet_show_clip(duration: float = 1.4, lift_deg: float = 15.0, turn_deg: float = 10.0) -> AnimationSpec:
    """Folded wing raises the tablet toward the viewer, holds, returns."""
    t_in, t_out = 0.3 * duration, 0.7 * duration

    def wing(t):
        if t < t_in:
            a = smoothstep(t / t_in)
        elif t < t_out:
            a = 1.0
        else:
            a = 1 - smoothstep((t - t_out) / (duration - t_out))
        return compose(Rotation.from_euler("y", -turn_deg * a, degrees=True),
                       Rotation.from_euler("x", -lift_deg * a, degrees=True))

    return AnimationSpec("tablet_show", [sample_rotation("wing_right", duration, wing)])


def default_clips(lid_axes: dict[str, np.ndarray], blink_close_deg: float) -> list[AnimationSpec]:
    return [
        idle_clip(),
        wave_clip(),
        blink_clip(lid_axes, blink_close_deg),
        nod_clip(),
        hop_clip(),
        tablet_show_clip(),
    ]
