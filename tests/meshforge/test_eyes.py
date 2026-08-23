import numpy as np
import trimesh

from meshforge.eyes import (
    EyeSocket,
    build_eyeball,
    build_eyelid,
    eye_pivot,
    eyelid_blink_axis_angle,
    neutralize_old_eye_paint,
)


def _forward_socket():
    """A socket whose normal is the world +Z axis, so "local +Z pole"
    and "world +Z direction" coincide — makes pupil-facing assertions
    straightforward without needing to un-rotate anything."""
    return EyeSocket(centroid=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))


def test_build_eyeball_vertex_count_matches_icosphere():
    socket = _forward_socket()
    eye = build_eyeball(socket, radius=0.1, subdivisions=3)
    reference = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    assert len(eye.vertices) == len(reference.vertices)
    assert len(eye.faces) == len(reference.faces)


def test_eyeball_colors_are_valid_rgba():
    socket = _forward_socket()
    eye = build_eyeball(socket, radius=0.1)
    colors = eye.visual.vertex_colors.astype(np.float32) / 255.0
    assert colors.shape[1] == 4
    assert np.all(colors >= 0.0) and np.all(colors <= 1.0)


def test_pupil_sits_at_forward_pole_sclera_at_back():
    socket = _forward_socket()
    radius = 0.1
    eye = build_eyeball(socket, radius=radius, bulge=0.0, pupil_deg=30, soften_deg=2)
    center = eye.vertices.mean(axis=0)
    local = eye.vertices - center
    colors = eye.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    luminance = colors @ np.array([0.299, 0.587, 0.114])

    front_pole = np.argmax(local[:, 2])  # closest to local +Z
    back_pole = np.argmin(local[:, 2])
    assert luminance[front_pole] < 0.2  # pupil: dark
    assert luminance[back_pole] > 0.7  # sclera: light


def test_soften_band_produces_no_hard_binary_cutoff():
    """A hard 0/1 color cutoff is what produced the visibly faceted
    (hexagonal) pupil edge in the first prototype pass. With soften_deg
    > 0 there must be vertices with genuinely intermediate luminance
    near the pupil/sclera boundary, not just pure black and pure white."""
    socket = _forward_socket()
    eye = build_eyeball(socket, radius=0.1, pupil_deg=29, soften_deg=4)
    colors = eye.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    luminance = colors @ np.array([0.299, 0.587, 0.114])
    mid_band = (luminance > 0.15) & (luminance < 0.85)
    assert mid_band.sum() > 0


def test_bulge_moves_sphere_along_normal():
    socket = EyeSocket(centroid=np.array([1.0, 2.0, 3.0]), normal=np.array([0.0, 0.0, 1.0]))
    radius = 0.1
    flush = build_eyeball(socket, radius=radius, bulge=0.0)
    bulging = build_eyeball(socket, radius=radius, bulge=0.03)
    # both spheres share a center offset purely along +normal (z); a
    # bigger bulge pushes the center further toward the socket (less
    # deep along -normal from the centroid).
    flush_center = flush.vertices.mean(axis=0)
    bulging_center = bulging.vertices.mean(axis=0)
    assert bulging_center[2] > flush_center[2]
    assert np.allclose(flush_center[:2], bulging_center[:2], atol=1e-9)


def test_raises_on_nonpositive_radius():
    socket = _forward_socket()
    try:
        build_eyeball(socket, radius=0.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "radius" in str(exc)


def test_raises_on_out_of_range_pupil_deg():
    socket = _forward_socket()
    try:
        build_eyeball(socket, pupil_deg=95)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pupil_deg" in str(exc)


def test_eye_pivot_matches_build_eyeball_center():
    socket = EyeSocket(centroid=np.array([0.5, 0.2, -0.3]), normal=np.array([0.0, 0.0, 1.0]))
    radius, bulge = 0.12, 0.02
    eye = build_eyeball(socket, radius=radius, bulge=bulge)
    pivot = eye_pivot(socket, radius=radius, bulge=bulge)
    assert np.allclose(eye.vertices.mean(axis=0), pivot, atol=1e-9)


def test_neutralize_blanks_only_within_radius():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    colors = np.tile(np.array([200, 30, 30, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

    socket = EyeSocket(centroid=np.array([1.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]))
    result = neutralize_old_eye_paint(mesh, [socket], radius=0.5, fill_color=np.array([0.9, 0.9, 0.9]))

    dist = np.linalg.norm(mesh.vertices - socket.centroid, axis=1)
    result_colors = result.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    inside = dist < 0.5
    outside = ~inside
    assert np.allclose(result_colors[inside], [0.9, 0.9, 0.9], atol=0.01)
    assert np.allclose(result_colors[outside], [200/255, 30/255, 30/255], atol=0.01)


def test_neutralize_does_not_mutate_input_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    colors = np.tile(np.array([10, 10, 10, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    original_colors = mesh.visual.vertex_colors.copy()

    socket = EyeSocket(centroid=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
    neutralize_old_eye_paint(mesh, [socket], radius=2.0)

    assert np.array_equal(mesh.visual.vertex_colors, original_colors)


def test_real_eye_sockets_are_distinct_and_roughly_forward_facing():
    from meshforge.eyes import LEFT_EYE_SOCKET, RIGHT_EYE_SOCKET
    assert np.linalg.norm(LEFT_EYE_SOCKET.centroid - RIGHT_EYE_SOCKET.centroid) > 0.1
    assert np.isclose(np.linalg.norm(LEFT_EYE_SOCKET.normal), 1.0)
    assert np.isclose(np.linalg.norm(RIGHT_EYE_SOCKET.normal), 1.0)
    # both sockets face predominantly +Z (toward camera at rest), not
    # sideways or backward
    assert LEFT_EYE_SOCKET.normal[2] > 0.8
    assert RIGHT_EYE_SOCKET.normal[2] > 0.8


def test_eyelid_rest_pose_does_not_cover_the_pupil():
    """Regression test for the first prototype's bug: too small a
    rest_up_deg let the lid's rest position droop over the pupil,
    reading as a permanently sleepy/closed eye."""
    socket = _forward_socket()
    pupil_deg = 29.0
    lid = build_eyelid(socket, radius=0.115, cap_deg=34.0, rest_up_deg=72.0)
    center = eye_pivot(socket, radius=0.115)
    local = lid.vertices - center
    ang = np.degrees(np.arccos(np.clip(local[:, 2] / np.linalg.norm(local, axis=1), -1, 1)))
    assert ang.min() > pupil_deg  # no lid vertex reaches as close to forward as the pupil edge


def test_eyelid_vertices_lie_on_the_expected_sphere():
    socket = EyeSocket(centroid=np.array([1.0, -0.5, 2.0]), normal=np.array([0.0, 0.0, 1.0]))
    radius, lid_scale, bulge = 0.115, 1.05, 0.022
    lid = build_eyelid(socket, radius=radius, lid_radius_scale=lid_scale, bulge=bulge)
    center = eye_pivot(socket, radius=radius, bulge=bulge)
    dists = np.linalg.norm(lid.vertices - center, axis=1)
    assert np.allclose(dists, radius * lid_scale, atol=1e-6)


def test_eyelid_blink_axis_angle_rotates_rest_onto_forward():
    from scipy.spatial.transform import Rotation
    from meshforge.eyes import _rest_lid_direction

    rest_up_deg = 72.0
    axis, angle = eyelid_blink_axis_angle(rest_up_deg)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert np.isclose(np.degrees(angle), rest_up_deg, atol=1e-6)

    rest_dir = _rest_lid_direction(rest_up_deg)
    rotated = Rotation.from_rotvec(axis * angle).apply(rest_dir)
    assert np.allclose(rotated, [0.0, 0.0, 1.0], atol=1e-6)


def test_eyelid_blink_axis_angle_raises_on_degenerate_rest_up_deg():
    try:
        eyelid_blink_axis_angle(rest_up_deg=0.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "rest_up_deg" in str(exc)


def test_eyelid_color_is_distinct_from_pupil_and_sclera():
    socket = _forward_socket()
    lid = build_eyelid(socket, radius=0.115)
    colors = lid.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    mean_color = colors.mean(axis=0)
    assert not np.allclose(mean_color, [0.04, 0.04, 0.05], atol=0.1)  # not pupil-black
    assert not np.allclose(mean_color, [0.98, 0.98, 0.98], atol=0.05)  # not pure sclera-white


def test_build_eyelid_raises_on_nonpositive_radius():
    socket = _forward_socket()
    try:
        build_eyelid(socket, radius=0.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "radius" in str(exc)


def test_build_eyelid_raises_on_out_of_range_cap_deg():
    socket = _forward_socket()
    try:
        build_eyelid(socket, cap_deg=91.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "cap_deg" in str(exc)
