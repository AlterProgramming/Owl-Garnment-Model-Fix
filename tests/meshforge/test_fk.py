"""Unit tests for meshforge.fk — the forward-kinematics arithmetic shared
by the preview renderer and the exporter (identity rest rotations, local
translation = pivot - parent pivot, inverse bind = translate(-pivot))."""
import numpy as np
import pytest

from meshforge.fk import (
    Joint,
    inverse_bind,
    local_rest_translation,
    ordered_joints,
    pose_normals,
    pose_vertices,
    skin_matrices,
    world_matrices,
)

QUAT_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
QUAT_90_Z = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
QUAT_180_Y = np.array([0.0, 1.0, 0.0, 0.0])


def chain():
    """root -> a -> b with distinct pivots."""
    return [
        Joint("root", None, np.array([0.0, 0.0, 0.0])),
        Joint("a", "root", np.array([1.0, 0.0, 0.0])),
        Joint("b", "a", np.array([1.0, 2.0, 0.0])),
    ]


# --------------------------------------------------------------------------
# ordered_joints
# --------------------------------------------------------------------------

def test_ordered_joints_puts_parents_before_children():
    joints = chain()
    shuffled = [joints[2], joints[0], joints[1]]
    ordered = ordered_joints(shuffled)
    names = [j.name for j in ordered]
    assert names.index("root") < names.index("a") < names.index("b")
    assert sorted(names) == ["a", "b", "root"]


def test_ordered_joints_is_stable_for_siblings():
    joints = [
        Joint("root", None, np.zeros(3)),
        Joint("left", "root", np.ones(3)),
        Joint("right", "root", -np.ones(3)),
    ]
    assert [j.name for j in ordered_joints(joints)] == ["root", "left", "right"]


def test_ordered_joints_raises_on_unknown_parent():
    joints = [Joint("root", None, np.zeros(3)), Joint("a", "ghost", np.ones(3))]
    with pytest.raises(ValueError, match="unknown parent"):
        ordered_joints(joints)


def test_ordered_joints_raises_on_cycle():
    joints = [Joint("a", "b", np.zeros(3)), Joint("b", "a", np.ones(3))]
    with pytest.raises(ValueError, match="cycle"):
        ordered_joints(joints)


# --------------------------------------------------------------------------
# world_matrices / local_rest_translation
# --------------------------------------------------------------------------

def test_local_rest_translation_is_pivot_minus_parent_pivot():
    joints = chain()
    by_name = {j.name: j for j in joints}
    assert np.allclose(local_rest_translation(by_name["root"], by_name), [0.0, 0.0, 0.0])
    assert np.allclose(local_rest_translation(by_name["a"], by_name), [1.0, 0.0, 0.0])
    assert np.allclose(local_rest_translation(by_name["b"], by_name), [0.0, 2.0, 0.0])


def test_world_matrices_at_rest_place_every_joint_at_its_pivot():
    joints = chain()
    world = world_matrices(joints)
    assert set(world) == {"root", "a", "b"}
    for j in joints:
        m = world[j.name]
        assert m.shape == (4, 4)
        assert np.allclose(m[:3, :3], np.eye(3))
        assert np.allclose(m[:3, 3], j.pivot)
        assert np.allclose(m[3], [0.0, 0.0, 0.0, 1.0])


def test_rotating_parent_90_about_z_swings_child_pivot():
    parent_pivot = np.array([2.0, 3.0, 5.0])
    joints = [
        Joint("parent", None, parent_pivot),
        Joint("child", "parent", parent_pivot + np.array([1.0, 0.0, 0.0])),
    ]
    world = world_matrices(joints, rotations={"parent": QUAT_90_Z})

    # the parent stays at its own pivot ...
    assert np.allclose(world["parent"][:3, 3], parent_pivot)
    # ... and the child's offset (1, 0, 0) becomes (0, 1, 0)
    assert np.allclose(world["child"][:3, 3], parent_pivot + np.array([0.0, 1.0, 0.0]), atol=1e-12)
    # the child inherits the parent's rotation
    assert np.allclose(world["child"][:3, :3], world["parent"][:3, :3])


def test_translation_offsets_add_to_the_rest_translation():
    joints = chain()
    world = world_matrices(joints, translations={"a": np.array([0.0, 0.0, 1.5])})
    assert np.allclose(world["a"][:3, 3], [1.0, 0.0, 1.5])
    # b has no offset of its own but rides along with a
    assert np.allclose(world["b"][:3, 3], [1.0, 2.0, 1.5])
    assert np.allclose(world["root"][:3, 3], [0.0, 0.0, 0.0])


def test_world_matrices_accept_joints_in_any_order():
    joints = chain()
    world = world_matrices([joints[2], joints[1], joints[0]], rotations={"root": QUAT_90_Z})
    # root rotates 90 deg about Z around the origin: a's pivot (1,0,0) -> (0,1,0), b's (1,2,0) -> (-2,1,0)
    assert np.allclose(world["a"][:3, 3], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(world["b"][:3, 3], [-2.0, 1.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------
# inverse_bind / skin_matrices / pose_vertices
# --------------------------------------------------------------------------

def test_inverse_bind_translates_by_minus_pivot():
    j = Joint("a", None, np.array([1.0, 2.0, 3.0]))
    m = inverse_bind(j)
    assert np.allclose(m[:3, :3], np.eye(3))
    assert np.allclose(m[:3, 3], [-1.0, -2.0, -3.0])


def test_skin_matrices_at_rest_are_identity():
    joints = chain()
    skin = skin_matrices(joints, world_matrices(joints))
    for name in ("root", "a", "b"):
        assert np.allclose(skin[name], np.eye(4), atol=1e-12)


def test_pose_vertices_at_rest_returns_the_input():
    joints = chain()
    names = [j.name for j in joints]
    skin = skin_matrices(joints, world_matrices(joints))
    rng = np.random.default_rng(0)
    vertices = rng.normal(size=(12, 3))
    weights = rng.random((12, 3))
    weights /= weights.sum(axis=1, keepdims=True)

    posed = pose_vertices(vertices, weights, names, skin)
    assert posed.shape == vertices.shape
    assert np.allclose(posed, vertices, atol=1e-12)


def test_pose_vertices_rejects_wrong_weight_shape():
    joints = chain()
    names = [j.name for j in joints]
    skin = skin_matrices(joints, world_matrices(joints))
    with pytest.raises(ValueError, match="weights must be"):
        pose_vertices(np.zeros((4, 3)), np.zeros((4, 2)), names, skin)


def test_fifty_fifty_blend_with_one_joint_flipped_collapses_to_origin():
    """Two root joints at the origin; rotating one of them 180 deg about Y
    sends (1,0,0) to (-1,0,0), and a 50/50 linear blend lands on (0,0,0)."""
    joints = [Joint("still", None, np.zeros(3)), Joint("flip", None, np.zeros(3))]
    names = ["still", "flip"]
    skin = skin_matrices(joints, world_matrices(joints, rotations={"flip": QUAT_180_Y}))

    vertices = np.array([[1.0, 0.0, 0.0]])
    weights = np.array([[0.5, 0.5]])
    posed = pose_vertices(vertices, weights, names, skin)
    assert np.allclose(posed, [[0.0, 0.0, 0.0]], atol=1e-12)

    # sanity: fully on the flipped joint it is the rigid rotation
    assert np.allclose(pose_vertices(vertices, np.array([[0.0, 1.0]]), names, skin), [[-1.0, 0.0, 0.0]], atol=1e-12)
    # and a zero-weight joint never contributes, even when it moves
    assert np.allclose(pose_vertices(vertices, np.array([[1.0, 0.0]]), names, skin), vertices, atol=1e-12)


def test_pose_vertices_rotates_about_the_joint_pivot_not_the_origin():
    pivot = np.array([3.0, 0.0, 0.0])
    joints = [Joint("hinge", None, pivot)]
    skin = skin_matrices(joints, world_matrices(joints, rotations={"hinge": QUAT_90_Z}))
    vertices = np.array([[4.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    posed = pose_vertices(vertices, np.ones((2, 1)), ["hinge"], skin)
    assert np.allclose(posed[0], pivot + np.array([0.0, 1.0, 0.0]), atol=1e-12)
    assert np.allclose(posed[1], pivot, atol=1e-12)  # the pivot itself does not move


# --------------------------------------------------------------------------
# pose_normals
# --------------------------------------------------------------------------

def test_pose_normals_keep_unit_length():
    joints = [Joint("still", None, np.zeros(3)), Joint("turn", None, np.array([1.0, 1.0, 1.0]))]
    names = ["still", "turn"]
    skin = skin_matrices(joints, world_matrices(joints, rotations={"turn": QUAT_90_Z}))

    normals = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.8, 0.0]])
    weights = np.array([[0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    out = pose_normals(normals, weights, names, skin)

    assert out.shape == normals.shape
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)
    # (1,0,0) blended 50/50 with its 90-deg-about-Z image (0,1,0) -> normalized diagonal
    assert np.allclose(out[0], [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-12)
    # a normal along the rotation axis is unchanged
    assert np.allclose(out[1], [0.0, 0.0, 1.0], atol=1e-12)
    # fully weighted on the turning joint: the pure rotation (translation ignored)
    assert np.allclose(out[2], [-0.8, 0.6, 0.0], atol=1e-12)


def test_pose_normals_at_rest_are_unchanged():
    joints = chain()
    names = [j.name for j in joints]
    skin = skin_matrices(joints, world_matrices(joints))
    rng = np.random.default_rng(2)
    normals = rng.normal(size=(10, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    weights = rng.random((10, 3))
    weights /= weights.sum(axis=1, keepdims=True)
    assert np.allclose(pose_normals(normals, weights, names, skin), normals, atol=1e-12)
