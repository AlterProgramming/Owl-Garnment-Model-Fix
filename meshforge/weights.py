"""Region-constrained automatic skin weights.

Implements the rigging guide's Phase 3 ("generate weights with regional
constraints"): every vertex carries a semantic region label (see
meshforge.regions); the interior of each region is a rigid *core* bound
100 % to that region's joint, and only a narrow band of vertices around
each region boundary is free. On the free band the weights are solved as
harmonic functions of the mesh graph (Laplacian = 0 with the cores as
Dirichlet boundary), which is the classic bone-heat / harmonic-coordinates
answer to "blend smoothly across the seam and nowhere else".

Why not plain inverse-distance everywhere: a distance field cannot tell a
folded wing from the flank it rests on, or a tablet from the belly behind
it — the guide calls this out as the reason region maps exist. Harmonic
weights on an explicit band keep every blend inside a declared seam.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class WeightResult(NamedTuple):
    weights: np.ndarray          # (n_verts, n_joints) float32, rows sum to 1
    joint_names: list[str]
    free_mask: np.ndarray        # (n_verts,) bool — vertices whose weights were solved, not pinned
    diagnostics: dict


def vertex_adjacency(n_verts: int, faces: np.ndarray) -> sp.csr_matrix:
    faces = np.asarray(faces, dtype=np.int64)
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    row = np.concatenate([a, b])
    col = np.concatenate([b, a])
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n_verts, n_verts)).tocsr()
    adj.data[:] = 1.0  # collapse duplicate edges
    return adj


def graph_distance_to_other_label(adjacency: sp.csr_matrix, labels: np.ndarray, max_rings: int) -> np.ndarray:
    """Ring index of every vertex relative to its region boundary: 0 for a
    vertex that shares an edge with a differently-labelled vertex, 1 for
    their neighbours, and so on, capped at max_rings + 1 ("farther")."""
    labels = np.asarray(labels)
    n = len(labels)
    dist = np.full(n, max_rings + 1, dtype=np.int32)
    # ring 0: vertices that have a neighbour with a different label
    indptr, indices = adjacency.indptr, adjacency.indices
    neighbor_labels_differ = np.zeros(n, dtype=bool)
    deg = np.diff(indptr)
    rows = np.repeat(np.arange(n), deg)
    differ = labels[rows] != labels[indices]
    neighbor_labels_differ[rows[differ]] = True
    frontier = neighbor_labels_differ
    dist[frontier] = 0
    for ring in range(1, max_rings + 1):
        reached = np.asarray((adjacency @ frontier.astype(np.float64)) > 0).ravel()
        newly = reached & (dist > ring)
        if not newly.any():
            break
        dist[newly] = ring
        frontier = newly
    return dist


def harmonic_weights(
    faces: np.ndarray,
    labels: np.ndarray,
    joint_names: list[str],
    rings: int | dict[str, int] = 3,
    max_influences: int = 4,
    min_weight: float = 0.01,
) -> WeightResult:
    """Solve region-constrained harmonic skin weights.

    `labels[i]` is the joint name owning vertex i's region. `rings` is how
    many vertex rings, counted from the region boundary, are allowed to
    blend — either a single int or a per-label dict: rings=0 pins the
    whole region rigid, rings=1 frees only the boundary ring (vertices
    sharing an edge with another label), rings=2 that ring and its
    neighbours, etc. Vertices outside every band are pinned to 1.0 on
    their own joint.
    """
    labels = np.asarray(labels)
    faces = np.asarray(faces, dtype=np.int64)
    n = int(faces.max()) + 1 if len(faces) else len(labels)
    n = max(n, len(labels))
    names = list(joint_names)
    unknown = sorted(set(np.unique(labels)) - set(names))
    if unknown:
        raise ValueError(f"harmonic_weights: labels reference joints not in joint_names: {unknown}")
    j_index = {name: i for i, name in enumerate(names)}
    label_idx = np.array([j_index[l] for l in labels], dtype=np.int64)

    adjacency = vertex_adjacency(n, faces)
    if isinstance(rings, dict):
        max_rings = max(rings.values()) if rings else 0
        per_vertex_rings = np.array([rings.get(l, 0) for l in labels], dtype=np.int32)
    else:
        max_rings = int(rings)
        per_vertex_rings = np.full(n, int(rings), dtype=np.int32)
    dist = graph_distance_to_other_label(adjacency, labels, max_rings)
    free = dist < per_vertex_rings
    # a joint whose region is entirely "free" would have no Dirichlet
    # anchor; keep at least its farthest vertices pinned.
    for j in range(len(names)):
        members = label_idx == j
        if members.any() and free[members].all():
            far = np.where(members)[0][np.argsort(-dist[members])[: max(1, members.sum() // 10)]]
            free[far] = False

    W = np.zeros((n, len(names)), dtype=np.float64)
    W[np.arange(n), label_idx] = 1.0  # pinned values (overwritten on free rows)

    if free.any():
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        L = sp.diags(degree) - adjacency
        L = L.tocsr()
        f_idx = np.where(free)[0]
        c_idx = np.where(~free)[0]
        L_ff = L[f_idx][:, f_idx].tocsc()
        L_fc = L[f_idx][:, c_idx].tocsr()
        rhs = -(L_fc @ W[c_idx])
        # free vertices with no path to any pinned vertex would make L_ff
        # singular; a tiny diagonal shift keeps the solve well-posed and
        # only perturbs such pathological islands.
        L_ff = L_ff + sp.identity(len(f_idx), format="csc") * 1e-9
        solve = spla.splu(L_ff)
        W[f_idx] = solve.solve(rhs)

    W = np.clip(W, 0.0, 1.0)
    # prune + renormalize
    W[W < min_weight] = 0.0
    if W.shape[1] > max_influences:
        order = np.argsort(-W, axis=1)[:, max_influences:]
        np.put_along_axis(W, order, 0.0, axis=1)
    row_sums = W.sum(axis=1)
    empty = row_sums <= 0
    if empty.any():
        W[empty, label_idx[empty]] = 1.0
        row_sums = W.sum(axis=1)
    W /= row_sums[:, None]

    influences = (W > 0).sum(axis=1)
    diagnostics = {
        "vertices": int(n),
        "free_vertices": int(free.sum()),
        "max_influences_seen": int(influences.max()) if n else 0,
        "influence_histogram": {int(k): int((influences == k).sum()) for k in np.unique(influences)},
        "per_joint_vertices": {name: int((label_idx == i).sum()) for i, name in enumerate(names)},
        "per_joint_weight_mass": {name: float(W[:, i].sum()) for i, name in enumerate(names)},
    }
    return WeightResult(weights=W.astype(np.float32), joint_names=names, free_mask=free, diagnostics=diagnostics)
