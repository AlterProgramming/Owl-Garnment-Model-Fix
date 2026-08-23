"""Focused regressions for the experimental contact-driven cloth path."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from meshforge import cloth_exec as E


def test_self_contact_filters_pattern_neighbours_but_separates_remote_layers():
    rows, cols = 6, 6
    n = rows * cols
    P = np.zeros((n, 3), dtype=np.float64)
    P[:, 0] = 100.0 + 10.0 * np.arange(n)

    # A structural neighbour is deliberately closer than the collision
    # thickness.  It is cloth connected to itself, not a self-collision.
    P[0] = [0.00, 0.0, 0.0]
    P[1] = [0.03, 0.0, 0.0]

    # Two vertices far apart in pattern space represent two folded layers.
    a, b = 1 * cols + 1, 4 * cols + 4
    P[a] = [2.00, 0.0, 0.0]
    P[b] = [2.03, 0.0, 0.0]

    before_neighbour = P[[0, 1]].copy()
    moved = E._separate_topology(P, np.ones(n, dtype=bool), 0.10, rows, cols, periodic=False)

    assert moved == 1
    assert np.allclose(P[[0, 1]], before_neighbour), "the cloth grid must not repel its own structural neighbours"
    assert np.linalg.norm(P[b] - P[a]) == pytest.approx(0.10, abs=1e-8)


def test_contact_projection_moves_penetrating_point_to_collision_offset():
    class PlaneCollider:
        pass

    col = PlaneCollider()
    col.points = np.array([[0.0, 0.0, 0.0]])
    col.normals = np.array([[0.0, 1.0, 0.0]])
    col.tree = cKDTree(col.points)

    P = np.array([[0.0, -0.08, 0.0], [0.0, 0.20, 0.0]])
    hit, normals = E._contact_project(P, col, 0.02, np.ones(2, dtype=bool))

    assert hit.tolist() == [True, False]
    assert P[0, 1] == pytest.approx(0.02)
    assert P[1, 1] == pytest.approx(0.20)
    assert np.allclose(normals[0], [0.0, 1.0, 0.0])


def _weight_fixture():
    joint_names = ["body", "chest", "wing_left", "wing_left_tip", "wing_right"]
    body_V = np.array([
        [0.00, 2.00, 0.00],
        [0.20, 2.00, 0.00],
        [0.40, 2.00, 0.00],
        [0.20, 1.50, 0.00],  # moving wing surface
        [5.00, 0.00, 0.00],
        [-5.0, 0.00, 0.00],
    ])
    body_W = np.zeros((len(body_V), len(joint_names)))
    body_W[0, 0] = 1.0
    body_W[1, 1] = 1.0
    body_W[2, 0] = 1.0
    body_W[3, 2] = 1.0
    body_W[4, 0] = 1.0
    body_W[5, 1] = 1.0

    rows, cols = 5, 3
    pins = np.array([[0.00, 2.00, 0.00], [0.20, 2.00, 0.00], [0.40, 2.00, 0.00]])
    V = np.empty((rows * cols, 3), dtype=np.float64)
    V[:cols] = pins
    for r in range(1, rows):
        y = 2.0 - 0.5 * r
        V[r * cols:(r + 1) * cols] = np.array([[3.0, y, 0.0], [3.2, y, 0.0], [3.4, y, 0.0]])
    V[cols + 1] = [0.205, 1.50, 0.0]  # row 1, directly on the wing support
    return joint_names, body_V, body_W, rows, cols, pins, V


def test_local_wing_support_begins_below_the_pinned_row():
    joint_names, body_V, body_W, rows, cols, pins, V = _weight_fixture()
    W = E.support_weights(V, np.arange(rows * cols), (rows, cols), pins,
                          body_V, body_W, joint_names, y_hip=0.5,
                          periodic=False, sigma=0.01)

    wing = joint_names.index("wing_left")
    assert np.allclose(W[:cols, wing], 0.0), "the tied support row remains torso driven"
    assert W[cols + 1, wing] > 0.25, "cloth actually touching the moving root inherits local support"
    assert W[cols + 2, wing] < W[cols + 1, wing] * 0.1, "wing support is local, not a garment-wide skinning shortcut"
    assert np.allclose(W.sum(axis=1), 1.0)


def test_explicit_tail_pin_weights_never_pick_up_wing_support():
    joint_names, body_V, body_W, rows, cols, pins, V = _weight_fixture()
    chest = joint_names.index("chest")
    pin_W = np.zeros((cols, len(joint_names)))
    pin_W[:, chest] = 1.0

    W = E.support_weights(V, np.arange(rows * cols), (rows, cols), pins,
                          body_V, body_W, joint_names, y_hip=0.5,
                          pin_weights=pin_W, periodic=False, sigma=0.01)

    wing_cols = [joint_names.index(n) for n in ("wing_left", "wing_left_tip", "wing_right")]
    assert np.allclose(W[:, wing_cols], 0.0)
    assert np.allclose(W.sum(axis=1), 1.0)
