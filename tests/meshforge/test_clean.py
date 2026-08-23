import numpy as np
import trimesh
from meshforge.clean import weld_and_clean


def test_weld_collapses_duplicate_seam(duplicate_seam_mesh):
    before_edges = np.sort(duplicate_seam_mesh.edges, axis=1)
    _, before_counts = np.unique(before_edges, axis=0, return_counts=True)
    before_boundary = int((before_counts == 1).sum())
    assert before_boundary > 0  # sanity: the fixture really has the defect

    cleaned = weld_and_clean(duplicate_seam_mesh, min_component_faces=1)

    after_edges = np.sort(cleaned.edges, axis=1)
    _, after_counts = np.unique(after_edges, axis=0, return_counts=True)
    after_boundary = int((after_counts == 1).sum())
    assert after_boundary == 0
    assert len(cleaned.vertices) == 4  # 6 verts -> 4 after welding the seam pairs


def test_debris_filter_drops_small_components(debris_mesh):
    cleaned = weld_and_clean(debris_mesh, min_component_faces=2)
    # box has 12 faces, debris shard has 1 face below threshold
    assert len(cleaned.faces) == 12


def test_denoise_preserves_uniform_color(debris_mesh):
    # a uniformly-colored mesh should stay uniform after denoise, not drift
    debris_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=debris_mesh,
        vertex_colors=np.tile([120, 40, 40, 255], (len(debris_mesh.vertices), 1)),
    )
    cleaned = weld_and_clean(debris_mesh, min_component_faces=2)
    colors = cleaned.visual.vertex_colors[:, :3].astype(np.float32)
    assert np.allclose(colors, colors[0], atol=2.0)


def test_raises_when_welding_does_not_fix_boundary():
    # a mesh that is genuinely open (a real hole, not a duplicate-vertex
    # artifact) must not be silently accepted — welding won't change its
    # boundary count, so weld_and_clean should raise rather than pretend
    # it's fine.
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = np.array([[0, 1, 2]])
    open_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    open_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=open_mesh,
        vertex_colors=np.tile([100, 100, 100, 255], (3, 1)),
    )
    try:
        weld_and_clean(open_mesh, min_component_faces=1)
        assert False, "expected ValueError for a genuinely open mesh"
    except ValueError as exc:
        assert "boundary" in str(exc).lower()


def test_raises_when_debris_filter_removes_all_components():
    # a mesh where every component is below min_component_faces threshold
    # must raise an error rather than silently return an empty mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [3.0, 2.0, 0.0],
        [2.0, 3.0, 0.0],
    ])
    # Two separate single-triangle components
    faces = np.array([
        [0, 1, 2],  # component 1: 1 face
        [3, 4, 5],  # component 2: 1 face
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile([100, 100, 100, 255], (6, 1)),
    )
    try:
        # With min_component_faces=2, both components get filtered out
        weld_and_clean(mesh, min_component_faces=2)
        assert False, "expected ValueError when all components are filtered"
    except ValueError as exc:
        assert "min_component_faces" in str(exc).lower()
        assert "removed all" in str(exc).lower()
