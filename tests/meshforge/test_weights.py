"""Unit tests for meshforge.weights — region-constrained harmonic skin
weights on tiny synthetic graphs (a box, a two-row triangle strip, a
triangle-fan disk)."""
import numpy as np
import pytest
import trimesh

from meshforge.weights import (
    WeightResult,
    graph_distance_to_other_label,
    harmonic_weights,
    vertex_adjacency,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def strip(n_cols=10):
    """A 2-row x n_cols grid of vertices triangulated into a strip along +x.
    Vertex 2*x is the bottom of column x, 2*x + 1 the top (same layout as
    test_skin.strip_mesh)."""
    vertices = []
    for x in range(n_cols):
        vertices.append([float(x), 0.0, 0.0])
        vertices.append([float(x), 1.0, 0.0])
    faces = []
    for x in range(n_cols - 1):
        b0, t0 = 2 * x, 2 * x + 1
        b1, t1 = 2 * (x + 1), 2 * (x + 1) + 1
        faces.append([b0, b1, t0])
        faces.append([t0, b1, t1])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def strip_labels(n_cols=10):
    """Left half 'a', right half 'b' (per column, both rows)."""
    half = n_cols // 2
    return np.array(["a"] * (2 * half) + ["b"] * (2 * (n_cols - half)))


def column_of(vertex_index):
    return vertex_index // 2


def fan_disk(n_rings=3, n_sectors=24):
    """A triangulated disk: a center vertex, then n_rings concentric rings
    of n_sectors vertices each. Sectors are split into three equal wedges
    labelled a / b / c, so the center vertex (labelled 'a') touches all
    three labels at once."""
    vertices = [[0.0, 0.0, 0.0]]
    ring_start = [None]
    for r in range(1, n_rings + 1):
        ring_start.append(len(vertices))
        for s in range(n_sectors):
            ang = 2.0 * np.pi * s / n_sectors
            vertices.append([r * np.cos(ang), r * np.sin(ang), 0.0])
    faces = []
    for s in range(n_sectors):
        faces.append([0, ring_start[1] + s, ring_start[1] + (s + 1) % n_sectors])
    for r in range(1, n_rings):
        for s in range(n_sectors):
            a = ring_start[r] + s
            b = ring_start[r] + (s + 1) % n_sectors
            c = ring_start[r + 1] + s
            d = ring_start[r + 1] + (s + 1) % n_sectors
            faces.append([a, c, d])
            faces.append([a, d, b])
    per_label = n_sectors // 3
    labels = ["a"]
    for _ in range(1, n_rings + 1):
        for s in range(n_sectors):
            labels.append("abc"[s // per_label])
    return np.array(vertices), np.array(faces, dtype=np.int64), np.array(labels)


# --------------------------------------------------------------------------
# vertex_adjacency
# --------------------------------------------------------------------------

def test_vertex_adjacency_of_box_is_symmetric_binary_without_self_loops():
    box = trimesh.creation.box()
    adj = vertex_adjacency(len(box.vertices), box.faces)

    assert adj.shape == (8, 8)
    assert (adj != adj.T).nnz == 0
    assert adj.diagonal().sum() == 0
    assert set(np.unique(adj.data)) == {1.0}
    # a box has 12 undirected edges (18 triangle edges minus 6 shared diagonals are
    # counted once each): 12 cube edges + 6 face diagonals = 18 undirected edges
    assert adj.nnz == 2 * 18
    # every vertex is connected to at least its 3 cube neighbours
    assert np.all(np.asarray(adj.sum(axis=1)).ravel() >= 3)


def test_vertex_adjacency_collapses_duplicate_edges():
    # two triangles sharing edge (1, 2): the shared edge must still be a single 1
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    adj = vertex_adjacency(4, faces)
    assert adj[1, 2] == 1.0 and adj[2, 1] == 1.0
    assert adj[0, 3] == 0.0


# --------------------------------------------------------------------------
# graph_distance_to_other_label
# --------------------------------------------------------------------------

@pytest.mark.parametrize("max_rings", [2, 3, 6])
def test_graph_distance_grows_one_per_column_from_the_seam_and_is_capped(max_rings):
    n_cols = 10
    _, faces = strip(n_cols)
    labels = strip_labels(n_cols)
    adj = vertex_adjacency(2 * n_cols, faces)

    dist = graph_distance_to_other_label(adj, labels, max_rings)

    assert dist.shape == (2 * n_cols,)
    for v in range(2 * n_cols):
        col = column_of(v)
        hops_to_seam = 4 - col if col <= 4 else col - 5
        expected = min(hops_to_seam, max_rings + 1)
        assert dist[v] == expected, (v, col, dist[v], expected)
    # the two seam columns are exactly 0, nothing else is
    assert set(np.nonzero(dist == 0)[0]) == {8, 9, 10, 11}


def test_graph_distance_is_all_capped_when_there_is_a_single_label():
    _, faces = strip(6)
    adj = vertex_adjacency(12, faces)
    dist = graph_distance_to_other_label(adj, np.array(["a"] * 12), max_rings=3)
    assert np.all(dist == 4)


# --------------------------------------------------------------------------
# harmonic_weights on the strip
# --------------------------------------------------------------------------

def _strip_result(rings):
    n_cols = 10
    _, faces = strip(n_cols)
    return harmonic_weights(faces, strip_labels(n_cols), ["a", "b"], rings=rings)


def test_harmonic_weights_rows_are_convex_combinations():
    res = _strip_result(rings=2)
    assert isinstance(res, WeightResult)
    assert res.joint_names == ["a", "b"]
    assert res.weights.shape == (20, 2)
    assert res.weights.dtype == np.float32
    assert np.all(res.weights >= 0.0)
    assert np.all(res.weights <= 1.0)
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)


def test_harmonic_weights_far_vertices_are_exactly_one_hot():
    res = _strip_result(rings=2)
    W = res.weights
    # columns 0-1 are >= 3 hops from the seam, columns 8-9 likewise: pinned
    for v in (0, 1, 2, 3):
        assert W[v, 0] == 1.0 and W[v, 1] == 0.0
        assert not res.free_mask[v]
    for v in (16, 17, 18, 19):
        assert W[v, 1] == 1.0 and W[v, 0] == 0.0
        assert not res.free_mask[v]


def test_harmonic_weights_band_vertices_blend_both_joints():
    res = _strip_result(rings=2)
    W = res.weights
    # columns 3..6 are within 1 hop of the seam and must be genuinely blended
    for col in (3, 4, 5, 6):
        for v in (2 * col, 2 * col + 1):
            assert res.free_mask[v]
            assert 0.02 < W[v, 0] < 0.98, (v, W[v])
            assert 0.02 < W[v, 1] < 0.98, (v, W[v])
    # the seam itself is roughly an even split
    assert 0.3 < W[8, 0] < 0.7
    assert 0.3 < W[10, 0] < 0.7


def test_harmonic_weights_decrease_monotonically_from_a_side_to_b_side():
    res = _strip_result(rings=2)
    W = res.weights
    bottom_a = W[0::2, 0]
    top_a = W[1::2, 0]
    assert np.all(np.diff(bottom_a) <= 1e-7)
    assert np.all(np.diff(top_a) <= 1e-7)
    # and it is a real ramp, not a step: strictly decreasing inside the band
    assert np.all(np.diff(bottom_a[3:7]) < -1e-3)
    assert bottom_a[0] == 1.0 and bottom_a[-1] == 0.0


"""Unit tests for meshforge.weights — region-constrained harmonic skin
weights on tiny synthetic graphs (a box, a two-row triangle strip, a
triangle-fan disk)."""
import numpy as np
import pytest
import trimesh

from meshforge.weights import (
    WeightResult,
    graph_distance_to_other_label,
    harmonic_weights,
    vertex_adjacency,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def strip(n_cols=10):
    """A 2-row x n_cols grid of vertices triangulated into a strip along +x.
    Vertex 2*x is the bottom of column x, 2*x + 1 the top (same layout as
    test_skin.strip_mesh)."""
    vertices = []
    for x in range(n_cols):
        vertices.append([float(x), 0.0, 0.0])
        vertices.append([float(x), 1.0, 0.0])
    faces = []
    for x in range(n_cols - 1):
        b0, t0 = 2 * x, 2 * x + 1
        b1, t1 = 2 * (x + 1), 2 * (x + 1) + 1
        faces.append([b0, b1, t0])
        faces.append([t0, b1, t1])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def strip_labels(n_cols=10):
    """Left half 'a', right half 'b' (per column, both rows)."""
    half = n_cols // 2
    return np.array(["a"] * (2 * half) + ["b"] * (2 * (n_cols - half)))


def column_of(vertex_index):
    return vertex_index // 2


def fan_disk(n_rings=3, n_sectors=24):
    """A triangulated disk: a center vertex, then n_rings concentric rings
    of n_sectors vertices each. Sectors are split into three equal wedges
    labelled a / b / c, so the center vertex (labelled 'a') touches all
    three labels at once."""
    vertices = [[0.0, 0.0, 0.0]]
    ring_start = [None]
    for r in range(1, n_rings + 1):
        ring_start.append(len(vertices))
        for s in range(n_sectors):
            ang = 2.0 * np.pi * s / n_sectors
            vertices.append([r * np.cos(ang), r * np.sin(ang), 0.0])
    faces = []
    for s in range(n_sectors):
        faces.append([0, ring_start[1] + s, ring_start[1] + (s + 1) % n_sectors])
    for r in range(1, n_rings):
        for s in range(n_sectors):
            a = ring_start[r] + s
            b = ring_start[r] + (s + 1) % n_sectors
            c = ring_start[r + 1] + s
            d = ring_start[r + 1] + (s + 1) % n_sectors
            faces.append([a, c, d])
            faces.append([a, d, b])
    per_label = n_sectors // 3
    labels = ["a"]
    for _ in range(1, n_rings + 1):
        for s in range(n_sectors):
            labels.append("abc"[s // per_label])
    return np.array(vertices), np.array(faces, dtype=np.int64), np.array(labels)


# --------------------------------------------------------------------------
# vertex_adjacency
# --------------------------------------------------------------------------

def test_vertex_adjacency_of_box_is_symmetric_binary_without_self_loops():
    box = trimesh.creation.box()
    adj = vertex_adjacency(len(box.vertices), box.faces)

    assert adj.shape == (8, 8)
    assert (adj != adj.T).nnz == 0
    assert adj.diagonal().sum() == 0
    assert set(np.unique(adj.data)) == {1.0}
    # a box has 12 undirected edges (18 triangle edges minus 6 shared diagonals are
    # counted once each): 12 cube edges + 6 face diagonals = 18 undirected edges
    assert adj.nnz == 2 * 18
    # every vertex is connected to at least its 3 cube neighbours
    assert np.all(np.asarray(adj.sum(axis=1)).ravel() >= 3)


def test_vertex_adjacency_collapses_duplicate_edges():
    # two triangles sharing edge (1, 2): the shared edge must still be a single 1
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    adj = vertex_adjacency(4, faces)
    assert adj[1, 2] == 1.0 and adj[2, 1] == 1.0
    assert adj[0, 3] == 0.0


# --------------------------------------------------------------------------
# graph_distance_to_other_label
# --------------------------------------------------------------------------

@pytest.mark.parametrize("max_rings", [2, 3, 6])
def test_graph_distance_grows_one_per_column_from_the_seam_and_is_capped(max_rings):
    n_cols = 10
    _, faces = strip(n_cols)
    labels = strip_labels(n_cols)
    adj = vertex_adjacency(2 * n_cols, faces)

    dist = graph_distance_to_other_label(adj, labels, max_rings)

    assert dist.shape == (2 * n_cols,)
    for v in range(2 * n_cols):
        col = column_of(v)
        hops_to_seam = 4 - col if col <= 4 else col - 5
        expected = min(hops_to_seam, max_rings + 1)
        assert dist[v] == expected, (v, col, dist[v], expected)
    # the two seam columns are exactly 0, nothing else is
    assert set(np.nonzero(dist == 0)[0]) == {8, 9, 10, 11}


def test_graph_distance_is_all_capped_when_there_is_a_single_label():
    _, faces = strip(6)
    adj = vertex_adjacency(12, faces)
    dist = graph_distance_to_other_label(adj, np.array(["a"] * 12), max_rings=3)
    assert np.all(dist == 4)


# --------------------------------------------------------------------------
# harmonic_weights on the strip
# --------------------------------------------------------------------------

def _strip_result(rings):
    n_cols = 10
    _, faces = strip(n_cols)
    return harmonic_weights(faces, strip_labels(n_cols), ["a", "b"], rings=rings)


def test_harmonic_weights_rows_are_convex_combinations():
    res = _strip_result(rings=2)
    assert isinstance(res, WeightResult)
    assert res.joint_names == ["a", "b"]
    assert res.weights.shape == (20, 2)
    assert res.weights.dtype == np.float32
    assert np.all(res.weights >= 0.0)
    assert np.all(res.weights <= 1.0)
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)


def test_harmonic_weights_far_vertices_are_exactly_one_hot():
    res = _strip_result(rings=2)
    W = res.weights
    # columns 0-1 are >= 3 hops from the seam, columns 8-9 likewise: pinned
    for v in (0, 1, 2, 3):
        assert W[v, 0] == 1.0 and W[v, 1] == 0.0
        assert not res.free_mask[v]
    for v in (16, 17, 18, 19):
        assert W[v, 1] == 1.0 and W[v, 0] == 0.0
        assert not res.free_mask[v]


def test_harmonic_weights_band_vertices_blend_both_joints():
    res = _strip_result(rings=2)
    W = res.weights
    # columns 3..6 are within 1 hop of the seam and must be genuinely blended
    for col in (3, 4, 5, 6):
        for v in (2 * col, 2 * col + 1):
            assert res.free_mask[v]
            assert 0.02 < W[v, 0] < 0.98, (v, W[v])
            assert 0.02 < W[v, 1] < 0.98, (v, W[v])
    # the seam itself is roughly an even split
    assert 0.3 < W[8, 0] < 0.7
    assert 0.3 < W[10, 0] < 0.7


def test_harmonic_weights_decrease_monotonically_from_a_side_to_b_side():
    res = _strip_result(rings=2)
    W = res.weights
    bottom_a = W[0::2, 0]
    top_a = W[1::2, 0]
    assert np.all(np.diff(bottom_a) <= 1e-7)
    assert np.all(np.diff(top_a) <= 1e-7)
    # and it is a real ramp, not a step: strictly decreasing inside the band
    assert np.all(np.diff(bottom_a[3:7]) < -1e-3)
    assert bottom_a[0] == 1.0 and bottom_a[-1] == 0.0


def test_harmonic_weights_rings_zero_is_fully_rigid():
    res = _strip_result(rings=0)
    W = res.weights
    assert res.free_mask.sum() == 0
    assert np.all((W == 0.0) | (W == 1.0))
    assert np.array_equal(W.argmax(axis=1), (np.arange(20) >= 10).astype(int))


def test_harmonic_weights_rings_zero_keeps_everything_outside_the_seam_rigid():
    """Convention-independent part of the rings=0 contract: whatever happens
    to the two seam columns, no other vertex may blend."""
    res = _strip_result(rings=0)
    W = res.weights
    non_seam = [v for v in range(20) if column_of(v) not in (4, 5)]
    assert np.all((W[non_seam] == 0.0) | (W[non_seam] == 1.0))
    assert np.array_equal(W[non_seam].argmax(axis=1), (np.array(non_seam) >= 10).astype(int))
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)


def test_harmonic_weights_per_label_rings_dict_limits_each_side_separately():
    n_cols = 10
    _, faces = strip(n_cols)
    labels = strip_labels(n_cols)
    res = harmonic_weights(faces, labels, ["a", "b"], rings={"a": 3, "b": 1})
    free_cols = sorted({column_of(v) for v in np.nonzero(res.free_mask)[0]})
    # the 'a' side is allowed a wider band than the 'b' side
    a_band = [c for c in free_cols if c <= 4]
    b_band = [c for c in free_cols if c >= 5]
    assert len(a_band) > len(b_band)
    assert 9 not in free_cols and 0 not in free_cols
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)


def test_harmonic_weights_rejects_labels_outside_joint_names():
    _, faces = strip(10)
    labels = np.array(["a"] * 10 + ["mystery"] * 10)
    with pytest.raises(ValueError, match="mystery"):
        harmonic_weights(faces, labels, ["a", "b"])


def test_harmonic_weights_diagnostics_report_per_joint_vertex_counts():
    _, faces = strip(10)
    labels = np.array(["a"] * 6 + ["b"] * 14)
    res = harmonic_weights(faces, labels, ["a", "b", "unused"], rings=1)

    d = res.diagnostics
    assert d["vertices"] == 20
    assert d["per_joint_vertices"] == {"a": 6, "b": 14, "unused": 0}
    assert d["free_vertices"] == int(res.free_mask.sum())
    assert d["max_influences_seen"] == int((res.weights > 0).sum(axis=1).max())
    assert sum(d["influence_histogram"].values()) == 20
    # weight mass is conserved: one unit per vertex, none of it on the unused joint
    assert np.isclose(sum(d["per_joint_weight_mass"].values()), 20.0, atol=1e-4)
    assert d["per_joint_weight_mass"]["unused"] == 0.0
    assert res.weights[:, 2].sum() == 0.0


# --------------------------------------------------------------------------
# max_influences pruning on a three-label fan
# --------------------------------------------------------------------------

def test_fan_center_collects_three_influences_without_pruning():
    _, faces, labels = fan_disk()
    res = harmonic_weights(faces, labels, ["a", "b", "c"], rings=2, max_influences=4)
    influences = (res.weights > 0).sum(axis=1)
    # the center vertex touches all three wedges and blends all three joints
    assert influences[0] == 3
    assert np.all(res.weights[0] > 0.2)
    assert (influences == 3).sum() >= 3
    # vertices deep inside a wedge on the outer ring stay pinned
    assert (influences == 1).sum() >= 3
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)


def test_max_influences_two_never_leaves_more_than_two_nonzero_weights():
    _, faces, labels = fan_disk()
    res = harmonic_weights(faces, labels, ["a", "b", "c"], rings=2, max_influences=2)

    influences = (res.weights > 0).sum(axis=1)
    assert influences.max() <= 2
    assert res.diagnostics["max_influences_seen"] <= 2
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(res.weights >= 0.0) and np.all(res.weights <= 1.0)
    # the center keeps its two strongest joints and drops the weakest
    assert influences[0] == 2
    assert set(res.diagnostics["influence_histogram"].keys()) <= {1, 2}
