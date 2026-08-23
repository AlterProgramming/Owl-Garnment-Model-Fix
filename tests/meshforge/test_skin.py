import numpy as np
import trimesh

from meshforge.segment import BBoxRegion, find_articulated_part
from meshforge.skin import (
    compute_multi_part_skin_weights,
    compute_skin_weights,
    pose_vertices_lbs,
    pose_vertices_multi_lbs,
)


def body_with_fused_wing():
    """Same fixture as test_segment.py's fused-wing case: body and wing
    triangles genuinely share two vertex indices (2, 3), standing in for
    a wing welded into one continuous surface with no pre-existing seam."""
    vertices = np.array([
        [0.0, 0.0, 0.0],   # 0: body
        [0.0, 1.0, 0.0],   # 1: body
        [1.0, 0.0, 0.0],   # 2: shared edge (body-wing attachment)
        [1.0, 1.0, 0.0],   # 3: shared edge (body-wing attachment)
        [2.0, 0.3, 0.0],   # 4: wing tip
        [2.0, 0.7, 0.0],   # 5: wing tip
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1],
        [1, 2, 3],
        [2, 4, 3],
        [3, 4, 5],
    ], dtype=np.uint32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def strip_mesh(n_cols=12):
    """A long two-row triangle strip along +x — long enough that vertices
    many columns away from a mid-strip cut fall outside the blend band and
    should stay pinned at their exact starting weight, unlike the tiny
    fused-wing fixture where every vertex is within 2 rings of the seam."""
    vertices = []
    for x in range(n_cols):
        vertices.append([float(x), 0.0, 0.0])
        vertices.append([float(x), 1.0, 0.0])
    vertices = np.array(vertices, dtype=np.float64)
    faces = []
    for x in range(n_cols - 1):
        b0, t0 = 2 * x, 2 * x + 1
        b1, t1 = 2 * (x + 1), 2 * (x + 1) + 1
        faces.append([b0, b1, t0])
        faces.append([t0, b1, t1])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.uint32), process=False)


def test_far_from_seam_vertices_stay_pinned_at_binary_weight():
    mesh = strip_mesh(n_cols=12)
    # faces are indexed by column pair (0-1, 1-2, ... 10-11); cut at the
    # midpoint so columns 0-4 are body, columns 7-11 are wing.
    n_col_pairs = 11
    cut = n_col_pairs // 2  # 5
    part_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    part_face_mask[2 * cut:] = True  # column pairs 5..10 -> wing

    weight = compute_skin_weights(mesh, part_face_mask)

    # column 0 (vertices 0,1) is 5 columns from the cut, well outside the
    # 2-ring blend band -> exactly 0 (body).
    assert weight[0] == 0.0
    assert weight[1] == 0.0
    # column 11 (last vertices) is symmetrically far on the wing side.
    assert weight[-1] == 1.0
    assert weight[-2] == 1.0
    # the seam region itself is a genuine blend, not a hard 0/1 cut.
    mid_vertex = 2 * cut
    assert 0.0 < weight[mid_vertex] < 1.0


def test_weight_ramps_monotonically_across_the_seam():
    mesh = strip_mesh(n_cols=12)
    part_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    part_face_mask[10:] = True

    weight = compute_skin_weights(mesh, part_face_mask)
    # bottom-row weights by column, left to right, should never decrease —
    # a smooth ramp from body to wing, not a jagged or reversed blend.
    bottom_row = weight[0::2]
    assert np.all(np.diff(bottom_row) >= -1e-9)


def test_seam_vertices_are_blended_not_hard_cut():
    mesh = body_with_fused_wing()
    part_face_mask = np.array([False, False, True, True])

    weight = compute_skin_weights(mesh, part_face_mask)

    # vertices 2 and 3 are shared by a body face and a wing face — the
    # exact seam a rigid crop would have severed. A real fix leaves them
    # with a genuine blend, not pinned to 0 or 1.
    assert 0.05 < weight[2] < 0.95
    assert 0.05 < weight[3] < 0.95


def test_weights_are_within_unit_range():
    mesh = body_with_fused_wing()
    part_face_mask = np.array([False, False, True, True])
    weight = compute_skin_weights(mesh, part_face_mask)
    assert np.all(weight >= 0.0)
    assert np.all(weight <= 1.0)


def test_raises_on_wrong_mask_shape():
    mesh = body_with_fused_wing()
    try:
        compute_skin_weights(mesh, np.array([True, False]))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "part_face_mask" in str(exc)


def test_no_seam_falls_back_to_clean_binary_split():
    """A part that never shares a vertex with the body (e.g. a
    pre-existing separate shell like the cap) has nothing to blend."""
    body = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    cap = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    cap.apply_translation([2.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([body, cap])

    part_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    part_face_mask[len(body.faces):] = True  # cap faces

    weight = compute_skin_weights(mesh, part_face_mask)
    body_vertex_count = len(body.vertices)
    assert np.all(weight[:body_vertex_count] == 0.0)
    assert np.all(weight[body_vertex_count:] == 1.0)


def test_compute_skin_weights_matches_real_segmentation_seam():
    """Integration with segment.py: the fused-wing region test used by
    test_segment.py should hand compute_skin_weights a real part_face_mask
    with an actual seam."""
    mesh = body_with_fused_wing()
    region = BBoxRegion(x=(0.5, 1.0), y=(-1.0, 1.0), z=(-1.0, 1.0))
    result = find_articulated_part(mesh, region, min_part_faces=1)

    weight = compute_skin_weights(mesh, result.part_face_mask)
    assert weight.shape == (len(mesh.vertices),)
    assert not np.all((weight == 0.0) | (weight == 1.0))  # some real blending happened


def test_pose_vertices_lbs_identity_rotation_is_a_noop():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    wing_weight = np.array([0.0, 0.5, 1.0])
    pivot = np.array([1.0, 0.0, 0.0])
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

    posed = pose_vertices_lbs(vertices, wing_weight, pivot, identity_quat)
    assert np.allclose(posed, vertices)


def test_pose_vertices_lbs_zero_weight_vertex_is_unmoved():
    vertices = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    wing_weight = np.array([0.0, 1.0])
    pivot = np.array([2.0, 0.0, 0.0])
    quat_90_about_z = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

    posed = pose_vertices_lbs(vertices, wing_weight, pivot, quat_90_about_z)
    assert np.allclose(posed[0], vertices[0])  # weight 0 -> untouched by rotation
    assert not np.allclose(posed[1], vertices[1])  # weight 1, not at pivot -> fully rotated


def test_pose_vertices_lbs_full_weight_matches_rigid_rotation_about_pivot():
    from scipy.spatial.transform import Rotation

    vertices = np.array([[3.0, 0.0, 0.0]])
    wing_weight = np.array([1.0])
    pivot = np.array([2.0, 0.0, 0.0])
    quat_90_about_z = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

    posed = pose_vertices_lbs(vertices, wing_weight, pivot, quat_90_about_z)
    expected = Rotation.from_quat(quat_90_about_z).apply(vertices - pivot) + pivot
    assert np.allclose(posed, expected)


def dumbbell_mesh():
    """A long strip with two independent 'wing' parts, one at each end,
    far enough apart (13 columns) that their 2-ring blend bands can never
    touch — the scenario compute_multi_part_skin_weights assumes."""
    mesh = strip_mesh(n_cols=24)
    left_mask = np.zeros(len(mesh.faces), dtype=bool)
    left_mask[0:10] = True  # column pairs 0-4 (columns 0-5)
    right_mask = np.zeros(len(mesh.faces), dtype=bool)
    right_mask[36:46] = True  # column pairs 18-22 (columns 18-23)
    return mesh, {"left": left_mask, "right": right_mask}


def test_multi_part_weights_sum_to_one_per_vertex():
    mesh, masks = dumbbell_mesh()
    weights = compute_multi_part_skin_weights(mesh, masks)

    assert set(weights.keys()) == {"body", "left", "right"}
    total = weights["body"] + weights["left"] + weights["right"]
    assert np.allclose(total, 1.0, atol=1e-9)


def test_multi_part_far_vertices_get_pinned_binary_weights():
    mesh, masks = dumbbell_mesh()
    weights = compute_multi_part_skin_weights(mesh, masks)

    # column 0 (vertices 0,1): 5 columns from the left seam (column 5),
    # outside the blend band -> fully left, untouched by right or body.
    assert weights["left"][0] == 1.0
    assert weights["left"][1] == 1.0
    assert weights["body"][0] == 0.0
    assert weights["right"][0] == 0.0

    # column 23 (last vertices): symmetric on the right end.
    assert weights["right"][-1] == 1.0
    assert weights["right"][-2] == 1.0
    assert weights["body"][-1] == 0.0
    assert weights["left"][-1] == 0.0

    # a middle column, far from both seams -> pure body.
    mid_col = 11
    mid_bottom = 2 * mid_col
    assert weights["body"][mid_bottom] == 1.0
    assert weights["left"][mid_bottom] == 0.0
    assert weights["right"][mid_bottom] == 0.0


def test_multi_part_bands_do_not_cross_contaminate():
    """The two parts sit 13 columns apart, each with only a 2-ring blend
    band around its own seam (~2 columns) -- left's band must never reach
    all the way into right's own territory (and vice versa). Note this is
    stricter than "the whole opposite half is zero": right's OWN blend
    band legitimately bleeds a couple of columns into what's nominally
    body territory near column 18 (see test_multi_part_seams_are_
    genuinely_blended) -- that's right blending with itself, not
    contamination from left."""
    mesh, masks = dumbbell_mesh()
    weights = compute_multi_part_skin_weights(mesh, masks)

    # right_part's own vertex range (columns 18-23, indices 36-47):
    # left's weight must be exactly 0 throughout, 13 columns from its seam.
    assert np.all(weights["left"][36:] == 0.0)
    # left_part's own vertex range (columns 0-5, indices 0-11):
    # right's weight must be exactly 0 throughout, 13 columns from its seam.
    assert np.all(weights["right"][:12] == 0.0)


def test_multi_part_seams_are_genuinely_blended():
    mesh, masks = dumbbell_mesh()
    weights = compute_multi_part_skin_weights(mesh, masks)

    left_seam_vertex = 10  # column 5, bottom row
    right_seam_vertex = 36  # column 18, bottom row
    assert 0.0 < weights["left"][left_seam_vertex] < 1.0
    assert 0.0 < weights["right"][right_seam_vertex] < 1.0


def test_pose_vertices_multi_lbs_identity_rotations_are_a_noop():
    mesh, masks = dumbbell_mesh()
    weights = compute_multi_part_skin_weights(mesh, masks)
    pivots = {"left": np.array([2.5, 0.5, 0.0]), "right": np.array([21.5, 0.5, 0.0])}
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    rotations = {"left": identity_quat, "right": identity_quat}

    posed = pose_vertices_multi_lbs(mesh.vertices, weights, pivots, rotations)
    assert np.allclose(posed, mesh.vertices)


def test_pose_vertices_multi_lbs_each_part_rotates_about_its_own_pivot_independently():
    mesh, masks = dumbbell_mesh()
    weights = compute_multi_part_skin_weights(mesh, masks)
    left_pivot = np.array([2.5, 0.5, 0.0])
    right_pivot = np.array([21.5, 0.5, 0.0])
    pivots = {"left": left_pivot, "right": right_pivot}

    quat_90_about_z = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

    # rotate only the left part -- the right part's fully-weighted tip
    # vertex must stay exactly put, and vice versa.
    rotations = {"left": quat_90_about_z, "right": identity_quat}
    posed = pose_vertices_multi_lbs(mesh.vertices, weights, pivots, rotations)
    assert np.allclose(posed[-1], mesh.vertices[-1])  # right tip untouched
    assert not np.allclose(posed[0], mesh.vertices[0])  # left tip rotated

    rotations = {"left": identity_quat, "right": quat_90_about_z}
    posed = pose_vertices_multi_lbs(mesh.vertices, weights, pivots, rotations)
    assert np.allclose(posed[0], mesh.vertices[0])  # left tip untouched
    assert not np.allclose(posed[-1], mesh.vertices[-1])  # right tip rotated
