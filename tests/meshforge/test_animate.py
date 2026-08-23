import numpy as np
from meshforge.animate import blink_clip, wave_clip


def test_returns_requested_keyframe_count():
    axis = np.array([0.0, 0.0, 1.0])
    keyframes = wave_clip(axis, degrees=35.0, duration_s=1.2, keyframe_count=8)
    assert len(keyframes) == 8


def test_keyframe_times_are_monotonic_and_span_duration():
    axis = np.array([0.0, 0.0, 1.0])
    keyframes = wave_clip(axis, duration_s=1.2, keyframe_count=8)
    times = [kf.time for kf in keyframes]
    assert times == sorted(times)
    assert times[0] == 0.0
    assert abs(times[-1] - 1.2) < 1e-9


def test_quaternions_are_unit_length():
    axis = np.array([0.3, 0.1, 0.9])
    axis = axis / np.linalg.norm(axis)
    keyframes = wave_clip(axis, keyframe_count=8)
    for kf in keyframes:
        assert abs(np.linalg.norm(kf.rotation) - 1.0) < 1e-6


def test_peak_rotation_matches_requested_degrees():
    axis = np.array([0.0, 0.0, 1.0])
    keyframes = wave_clip(axis, degrees=35.0, keyframe_count=9)
    # the sinusoidal ease should hit its peak angle at the midpoint keyframe
    peak = keyframes[len(keyframes) // 2]
    peak_angle_rad = 2.0 * np.arccos(np.clip(peak.rotation[3], -1.0, 1.0))
    assert abs(np.degrees(peak_angle_rad) - 35.0) < 1.0


def test_raises_on_invalid_keyframe_count():
    axis = np.array([0.0, 0.0, 1.0])
    try:
        wave_clip(axis, keyframe_count=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_raises_on_zero_degrees():
    axis = np.array([0.0, 0.0, 1.0])
    try:
        wave_clip(axis, degrees=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_raises_on_negative_degrees():
    axis = np.array([0.0, 0.0, 1.0])
    try:
        wave_clip(axis, degrees=-15.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_raises_on_zero_axis():
    axis = np.array([0.0, 0.0, 0.0])
    try:
        wave_clip(axis)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_raises_on_zero_duration():
    axis = np.array([0.0, 0.0, 1.0])
    try:
        wave_clip(axis, duration_s=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_raises_on_negative_duration():
    axis = np.array([0.0, 0.0, 1.0])
    try:
        wave_clip(axis, duration_s=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_blink_clip_ends_with_a_long_held_open_plateau():
    axis = np.array([1.0, 0.0, 0.0])
    keyframes = blink_clip(axis, close_degrees=72.0, blink_duration_s=0.24, hold_open_s=3.4, keyframe_count=6)
    # blink keyframes (6) + one trailing hold keyframe
    assert len(keyframes) == 7
    assert abs(keyframes[-1].time - (0.24 + 3.4)) < 1e-9
    # the held pose is identity (fully open again), not left mid-blink
    assert abs(keyframes[-1].rotation[3] - 1.0) < 1e-6


def test_blink_clip_peak_matches_close_degrees():
    axis = np.array([1.0, 0.0, 0.0])
    keyframes = blink_clip(axis, close_degrees=72.0, keyframe_count=7)
    blink_only = keyframes[:-1]  # exclude the trailing hold keyframe
    peak = blink_only[len(blink_only) // 2]
    peak_angle_rad = 2.0 * np.arccos(np.clip(abs(peak.rotation[3]), -1.0, 1.0))
    assert abs(np.degrees(peak_angle_rad) - 72.0) < 1.0


def test_blink_clip_times_are_monotonic():
    axis = np.array([1.0, 0.0, 0.0])
    keyframes = blink_clip(axis, close_degrees=50.0)
    times = [kf.time for kf in keyframes]
    assert times == sorted(times)
    assert times[0] == 0.0


def test_blink_clip_raises_on_nonpositive_close_degrees():
    axis = np.array([1.0, 0.0, 0.0])
    try:
        blink_clip(axis, close_degrees=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_blink_clip_raises_on_nonpositive_blink_duration():
    axis = np.array([1.0, 0.0, 0.0])
    try:
        blink_clip(axis, close_degrees=50.0, blink_duration_s=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_blink_clip_raises_on_nonpositive_hold_open():
    axis = np.array([1.0, 0.0, 0.0])
    try:
        blink_clip(axis, close_degrees=50.0, hold_open_s=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_blink_clip_raises_on_zero_axis():
    axis = np.array([0.0, 0.0, 0.0])
    try:
        blink_clip(axis, close_degrees=50.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
