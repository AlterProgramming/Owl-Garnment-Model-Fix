import numpy as np
import trimesh
from meshforge.segment import BBoxRegion, find_articulated_part


def body_with_wing():
    """A body box with a thin wing box protruding from its upper-right
    corner, standing in for the owl's raised wing relative to its body."""
    body = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    wing = trimesh.creation.box(extents=[0.6, 0.15, 0.15])
    wing.apply_translation([0.7, 0.55, 0.0])  # protrudes past the body's +x face
    return trimesh.util.concatenate([body, wing])


def body_with_fused_wing():
    """A minimal hand-built mesh where 'body' and 'wing' triangles share
    two vertex indices directly — genuinely one connected component by
    construction, standing in for the real owl asset's wing being fully
    welded into one continuous surface with no pre-existing seam.

    This avoids the complexity of trying to glue box primitives via
    coordinate snapping. Vertices 2 and 3 form a shared edge between
    body triangles (0-1-2, 1-2-3) and wing triangles (2-4-3, 3-4-5)."""
    vertices = np.array([
        [0.0, 0.0, 0.0],   # 0: body
        [0.0, 1.0, 0.0],   # 1: body
        [1.0, 0.0, 0.0],   # 2: shared edge (body-wing attachment)
        [1.0, 1.0, 0.0],   # 3: shared edge (body-wing attachment)
        [2.0, 0.3, 0.0],   # 4: wing tip
        [2.0, 0.7, 0.0],   # 5: wing tip
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1],  # body triangle 1
        [1, 2, 3],  # body triangle 2 (uses shared edge 2-3)
        [2, 4, 3],  # wing triangle 1 (uses shared edge 2-3)
        [3, 4, 5],  # wing triangle 2
    ], dtype=np.uint32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def body_with_floating_part():
    """A body box with a completely disconnected floating box.
    Used to test that the pivot sanity check rejects parts that aren't
    actually attached to the body."""
    body = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    floating = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    # Move the floating part far away from the body
    floating.apply_translation([5.0, 5.0, 5.0])
    return trimesh.util.concatenate([body, floating])


def test_finds_wing_component_in_region():
    """Verify that the wing (already separate shell) is correctly isolated."""
    mesh = body_with_wing()
    # Wing bounds are approximately [0.4, 1.0] x [0.475, 0.625] x [-0.075, 0.075]
    # Mesh bounds are [-0.5, 1.0] x [-0.5, 0.625] x [-0.5, 0.5]
    # Use fractions with small margin to include boundary faces (13/15 - ε, 1.0 + ε)
    # to handle floating-point boundary issues
    region = BBoxRegion(x=(0.59, 1.01), y=(0.86, 1.01), z=(0.42, 0.58))
    result = find_articulated_part(mesh, region, min_part_faces=1)
    # the wing box has 12 triangles; the isolated part should match it,
    # not accidentally grab the whole combined mesh
    assert 10 <= len(result.part.faces) <= 14
    assert len(result.body.faces) >= 12  # the remaining body box


def test_finds_fused_wing_component_in_region():
    """Verify that a wing fused into the body (1 connected component by
    construction) is correctly isolated via the crop-then-component algorithm.

    This is the core algorithmic test: without region cropping, the whole
    mesh is one connected component. Only via crop-then-component can we
    isolate just the wing triangles (faces 2-3) from the body (faces 0-1).
    This proves the algorithm is NOT just looking for pre-existing
    disconnected components."""
    from meshforge.clean import _face_adjacency_components as _components
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    mesh = body_with_fused_wing()

    # VERIFY: mesh is genuinely fused (1 connected component by construction)
    adj = mesh.face_adjacency
    n_faces = len(mesh.faces)
    if len(adj) > 0:
        graph = sp.coo_matrix(
            (np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n_faces, n_faces)
        )
        n_comp, labels = connected_components(graph, directed=False)
        assert n_comp == 1, f"fixture should be fused: {n_comp} components found"

    # Mesh bounds: x in [0, 2], y in [0, 1], z all 0
    # Region to capture wing: x > 1.0, which is the wing triangles (verts 2,3,4,5)
    region = BBoxRegion(x=(0.5, 1.0), y=(-1.0, 1.0), z=(-1.0, 1.0))
    result = find_articulated_part(mesh, region, min_part_faces=1)
    # Wing has 2 triangles; body has 2 original triangles plus whatever
    # fan-fill triangles _fill_body_hole added to patch the cut (see
    # test_segment_fill_body_hole.py-style coverage below) — the body's
    # face count is no longer exactly 2 once hole-filling is applied, but
    # it must be at least the 2 original triangles.
    assert len(result.part.faces) == 2, f"expected 2 wing triangles, got {len(result.part.faces)}"
    assert len(result.body.faces) >= 2  # original body triangles, plus any hole-fill


def test_body_hole_is_filled_not_left_open():
    """Cropping the wing out of a fused mesh leaves an open hole where the
    cut happened, which reads as a torn-open gap once the wing is animated
    away from rest — reported directly against this pipeline's real output.
    The returned body must have that hole patched (0 boundary edges), not
    left open."""
    mesh = body_with_fused_wing()
    region = BBoxRegion(x=(0.5, 1.0), y=(-1.0, 1.0), z=(-1.0, 1.0))
    result = find_articulated_part(mesh, region, min_part_faces=1)

    edges_sorted = np.sort(result.body.edges, axis=1)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edge_count = int((counts == 1).sum())
    assert boundary_edge_count == 0, (
        f"body still has {boundary_edge_count} open boundary edges after "
        "segmentation — the wing-cut hole was not patched"
    )


def test_fill_body_hole_is_a_noop_on_an_already_closed_mesh():
    """A body that has no open boundary (e.g. the wing was already a
    separate shell, not fused) must pass through unchanged — filling
    should only ever act on a genuine hole, never alter a closed mesh."""
    mesh = body_with_wing()  # wing is a separate, unwelded shell here
    region = BBoxRegion(x=(0.59, 1.01), y=(0.86, 1.01), z=(0.42, 0.58))
    result = find_articulated_part(mesh, region, min_part_faces=1)

    # the body here is the original closed box (12 faces), untouched
    assert len(result.body.faces) == 12
    edges_sorted = np.sort(result.body.edges, axis=1)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    assert int((counts == 1).sum()) == 0


def test_pivot_is_near_attachment_not_wing_tip():
    """Verify pivot sits at attachment (boundary with body), not wing tip."""
    mesh = body_with_wing()
    region = BBoxRegion(x=(0.59, 1.01), y=(0.86, 1.01), z=(0.42, 0.58))
    result = find_articulated_part(mesh, region, min_part_faces=1)
    # pivot should sit near the wing's attachment (x ~ 0.5, the body's
    # +x face), not near the wing tip (x ~ 1.0)
    assert result.pivot_point[0] < 0.75


def test_raises_when_region_has_no_component():
    """Verify that an empty region (no faces) raises ValueError with
    'no component' in the message."""
    mesh = body_with_wing()
    # Use fractions far outside the mesh bounds
    empty_region = BBoxRegion(x=(10.0, 11.0), y=(10.0, 11.0), z=(10.0, 11.0))
    try:
        find_articulated_part(mesh, empty_region, min_part_faces=1)
        assert False, "expected ValueError for an empty region"
    except ValueError as exc:
        assert "no component" in str(exc).lower()


def test_raises_when_pivot_too_far_from_body():
    """Verify that a part that is floating/disconnected from the body
    raises ValueError for failing the pivot sanity check."""
    mesh = body_with_floating_part()
    # Create a region that contains only the floating box
    # Mesh bounds will be approximately [-0.5, 5.2] x [-0.5, 5.2] x [-0.5, 5.2]
    # Floating box is at [4.9, 5.1] x [4.9, 5.1] x [4.9, 5.1] (approx)
    # We need fractional coords to select the floating box only
    # Fractions: (x_min - mesh_min) / size
    # Let's use fractions that roughly capture just the floating part
    region = BBoxRegion(x=(0.8, 1.0), y=(0.8, 1.0), z=(0.8, 1.0))
    try:
        find_articulated_part(mesh, region, min_part_faces=1)
        assert False, "expected ValueError for disconnected part"
    except ValueError as exc:
        # Should fail on the pivot distance sanity check, not empty region
        assert "does not look attached to the body" in str(exc).lower() or \
               "exceeds 2x the part's average edge length" in str(exc).lower()
