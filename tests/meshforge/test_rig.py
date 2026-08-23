import numpy as np
import trimesh
from meshforge.rig import build_rig


def test_root_has_body_and_part_child():
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    pivot = np.array([0.5, 0.0, 0.0])

    root = build_rig(body, part, pivot)

    assert root.name == "root"
    assert root.mesh is None
    names = {child.name for child in root.children}
    assert names == {"body", "wing_pivot"}


def test_part_vertices_are_relative_to_pivot():
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    part.apply_translation([0.75, 0.0, 0.0])  # part center at world (0.75, 0, 0)
    pivot = np.array([0.5, 0.0, 0.0])

    root = build_rig(body, part, pivot)
    wing_node = next(c for c in root.children if c.name == "wing_pivot")

    assert np.allclose(wing_node.translation, pivot)
    # part mesh vertices should now be centered near (0.25, 0, 0) in the
    # node's local space (world center 0.75 minus pivot 0.5)
    assert np.allclose(wing_node.mesh.vertices.mean(axis=0), [0.25, 0.0, 0.0], atol=1e-6)


def test_body_node_has_no_translation():
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    pivot = np.array([0.5, 0.0, 0.0])

    root = build_rig(body, part, pivot)
    body_node = next(c for c in root.children if c.name == "body")

    assert np.allclose(body_node.translation, [0.0, 0.0, 0.0])


def test_raises_on_bad_pivot_point_shape():
    body = trimesh.creation.box(extents=[1, 1, 1])
    part = trimesh.creation.box(extents=[0.5, 0.2, 0.2])
    bad_pivot = np.array([0.5, 0.0])  # wrong shape: (2,) instead of (3,)
    try:
        build_rig(body, part, bad_pivot)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pivot_point" in str(exc)


def test_raises_on_none_part():
    body = trimesh.creation.box(extents=[1, 1, 1])
    pivot = np.array([0.5, 0.0, 0.0])
    try:
        build_rig(body, None, pivot)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "part" in str(exc)
