"""Tests for meshforge.beads — the anklet ring.

Synthetic geometry, same reasoning as the other decal tests: a cylinder
standing in for a leg, with a known radius, so the measured ring geometry
can be checked against a closed-form answer rather than "looks right".
"""
import numpy as np
import pytest
import trimesh

from meshforge.beads import DEFAULT_BEAD_COLORS, build_bead_ring, build_symmetric_bead_rings
from meshforge.regions import RegionMap


def make_leg(radius: float = 0.15, y_lo: float = -1.0, y_hi: float = -0.4, n_theta: int = 40, n_y: int = 12,
            label: str = "leg_left"):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ys = np.linspace(y_lo, y_hi, n_y)
    T, Y = np.meshgrid(th, ys, indexing="ij")
    V = np.stack([radius * np.cos(T), Y, radius * np.sin(T)], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(n_theta):
        j = (i + 1) % n_theta
        for k in range(n_y - 1):
            a, b = i * n_y + k, j * n_y + k
            faces.append([a, b, a + 1])
            faces.append([b, b + 1, a + 1])
    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(faces), process=False)
    labels = np.full(len(V), label, dtype=object)
    rm = RegionMap(labels=labels, joints=[], fractions=None, collar={}, stats={})
    return mesh, rm


def make_two_legs(radius_left: float = 0.10, radius_right: float = 0.10, y_lo: float = -1.0, y_hi: float = -0.4):
    """Two cylinders (different radii allowed) sharing one mesh, offset
    apart in X, labelled leg_left / leg_right — a stand-in for the real
    owl's asymmetric-radius, asymmetric-label geometry."""
    left, _ = make_leg(radius=radius_left, y_lo=y_lo, y_hi=y_hi)
    right, _ = make_leg(radius=radius_right, y_lo=y_lo, y_hi=y_hi)
    right.vertices = right.vertices + np.array([1.0, 0.0, 0.0])
    n_left = len(left.vertices)
    V = np.concatenate([left.vertices, right.vertices], axis=0)
    F = np.concatenate([left.faces, right.faces + n_left], axis=0)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    labels = np.array(["leg_left"] * n_left + ["leg_right"] * len(right.vertices), dtype=object)
    rm = RegionMap(labels=labels, joints=[], fractions=None, collar={}, stats={})
    return mesh, rm


def test_ring_radius_matches_the_measured_leg_radius():
    radius = 0.15
    mesh, rm = make_leg(radius=radius)
    ring = build_bead_ring(mesh, rm, "leg_left", "leg_left", bead_count=10, ring_radius_scale=1.0)
    # the cylinder has a constant radius everywhere, so the measured ring
    # radius must land on it (well within tolerance for the sampling band)
    assert ring.info["ring_radius"] == pytest.approx(radius, rel=0.05)


def test_bead_count_matches_request():
    mesh, rm = make_leg()
    ring = build_bead_ring(mesh, rm, "leg_left", "leg_left", bead_count=9)
    assert ring.info["beads"] == 9


def test_every_bead_is_one_of_the_declared_colors():
    mesh, rm = make_leg()
    ring = build_bead_ring(mesh, rm, "leg_left", "leg_left", bead_count=12)
    colors = ring.primitive.colors
    declared = np.array(DEFAULT_BEAD_COLORS, dtype=np.float64)
    close = np.all(np.isclose(colors[:, None, :], declared[None, :, :], atol=1e-9), axis=-1)
    assert np.all(close.any(axis=-1))


def test_beads_sit_at_the_requested_height_band():
    mesh, rm = make_leg(y_lo=-1.0, y_hi=-0.4)
    ring = build_bead_ring(mesh, rm, "leg_left", "leg_left", bead_count=10, height_percentile=10.0)
    # height_percentile=10 on a uniform y distribution should land near
    # y_lo + 0.1*(y_hi-y_lo) = -0.94
    assert ring.info["world_y"] == pytest.approx(-0.94, abs=0.05)


def test_raises_when_leg_label_has_too_few_vertices():
    mesh, rm = make_leg()
    with pytest.raises(ValueError):
        build_bead_ring(mesh, rm, "leg_right", "leg_right")  # no leg_right vertices in this fixture


def test_faces_are_valid_indices_into_vertices():
    mesh, rm = make_leg()
    ring = build_bead_ring(mesh, rm, "leg_left", "leg_left", bead_count=6)
    n = len(ring.primitive.vertices)
    assert ring.primitive.faces.min() >= 0
    assert ring.primitive.faces.max() < n


def test_radius_override_replaces_the_measured_radius():
    mesh, rm = make_leg(radius=0.15)
    ring = build_bead_ring(mesh, rm, "leg_left", "leg_left", radius_override=0.5, ring_radius_scale=1.0)
    assert ring.info["ring_radius"] == pytest.approx(0.5, rel=1e-6)
    assert ring.info["measured_radius"] == pytest.approx(0.15, rel=0.05)
    assert ring.info["radius_overridden"] is True


def test_symmetric_rings_use_one_shared_radius_despite_asymmetric_geometry():
    # reproduces the real bug: leg_right measured 2-3x wider than leg_left
    # on the actual owl (a labelling asymmetry, not a real body-shape
    # difference) — the fix must use ONE shared radius for both anklets,
    # not each leg's own (unequally contaminated) measurement.
    mesh, rm = make_two_legs(radius_left=0.06, radius_right=0.20)
    left, right = build_symmetric_bead_rings(mesh, rm, bead_count=8)
    assert left.info["ring_radius"] == pytest.approx(right.info["ring_radius"], rel=1e-6)
    # the shared radius must be the SMALLER (less contaminated) reading
    assert left.info["ring_radius"] < 0.10
    # both are explicitly given the shared radius (build_symmetric_bead_rings
    # always passes radius_override), but only the right one's *measured*
    # value actually differs from what it ended up using
    assert left.info["measured_radius"] == pytest.approx(left.info["ring_radius"], rel=0.2)
    assert right.info["measured_radius"] > right.info["ring_radius"] * 1.5


def test_symmetric_rings_keep_each_legs_own_position():
    mesh, rm = make_two_legs(radius_left=0.08, radius_right=0.08)
    left, right = build_symmetric_bead_rings(mesh, rm, bead_count=8)
    # the two legs are offset by 1.0 in x (see make_two_legs) — positions
    # must stay distinct even though the radius is now shared
    assert right.info["centre_xz"][0] - left.info["centre_xz"][0] == pytest.approx(1.0, abs=0.05)


def test_beads_are_oval_not_spherical():
    # measure the FIRST bead in isolation, relative to its own centroid —
    # comparing to the world origin (as an earlier version of this test
    # did) is dominated by the ring's own position and barely moves,
    # since elongation is along the ring's local *tangent* direction,
    # roughly perpendicular to the radial vector from the origin.
    mesh, rm = make_leg()
    n_per_bead = len(trimesh.creation.icosphere(subdivisions=2, radius=1.0).vertices)

    round_beads = build_bead_ring(mesh, rm, "leg_left", "leg_left", bead_count=4, bead_elongation=1.0, seed=1)
    oval_beads = build_bead_ring(mesh, rm, "leg_left", "leg_left", bead_count=4, bead_elongation=1.8, seed=1)

    round0 = round_beads.primitive.vertices[:n_per_bead]
    oval0 = oval_beads.primitive.vertices[:n_per_bead]
    round_span = round0.max(axis=0) - round0.min(axis=0)
    oval_span = oval0.max(axis=0) - oval0.min(axis=0)

    # a round bead is roughly equal in all three axes; an elongated one
    # must be measurably longer along its longest axis than the round
    # baseline was along its corresponding axis
    assert oval_span.max() > round_span.max() * 1.3
    # and it should NOT have simply grown uniformly (that would be a
    # bigger sphere, not an oval) — the shortest axis stays close to the
    # round baseline's radius
    assert oval_span.min() == pytest.approx(round_span.min(), rel=0.15)
