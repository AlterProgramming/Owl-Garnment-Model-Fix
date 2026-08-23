"""The designed knot: a cinched bunch, waist band, short ends, and folds."""
import numpy as np

from meshforge import knot as K
from meshforge.rigexport import MaterialSpec

FRAME = {"origin": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0]),
         "tangent": np.array([1.0, 0.0, 0.0]), "up": np.array([0.0, 1.0, 0.0])}


def _arc(n=24, half=0.1):
    return np.stack([np.linspace(-half, half, n), np.zeros(n), np.zeros(n)], axis=1)


def test_knot_is_a_valid_primitive_on_its_frame():
    """The bunch construction stays compact and preserves the mesh contract."""
    knot = K.build_knot(FRAME, _arc(), size=0.1, material=MaterialSpec(name="m"))
    p = knot.primitive
    assert p.faces.min() >= 0 and p.faces.max() < len(p.vertices)
    assert np.allclose(np.linalg.norm(p.normals, axis=1), 1.0, atol=1e-6)
    assert p.uvs.shape == (len(p.vertices), 2)
    assert p.uvs.min() >= 0.0
    assert p.uvs.max() <= 1.0
    compact_extent = np.ptp(p.vertices, axis=0) / 0.1
    assert np.all(compact_extent <= np.array([1.6, 1.4, 0.9]) + 1e-9)
    assert knot.base_points.shape == (7, 3)
    assert np.allclose(knot.base_points[:, 1:], 0.0, atol=1e-9)
    assert 800 < len(p.faces) < 3600
    assert p.material.name == "m"


def test_knot_scales_with_size_and_strands():
    small = K.build_knot(FRAME, _arc(), size=0.05, material=MaterialSpec(name="m"), n_strands=4, stations=10)
    big = K.build_knot(FRAME, _arc(), size=0.10, material=MaterialSpec(name="m"))
    assert len(small.primitive.faces) < len(big.primitive.faces)
    assert small.primitive.vertices[:, 2].max() < big.primitive.vertices[:, 2].max()
    assert small.base_points.shape == (4, 3)


def test_tube_sweeps_a_ring_along_a_path():
    path = np.stack([np.linspace(0, 1, 5), np.zeros(5), np.zeros(5)], axis=1)
    V, F, UV = K._tube(path, np.full(5, 0.1), 6, along=(0.0, 1.0), around=(0.0, 1.0))
    assert V.shape == (30, 3) and F.shape == (48, 3) and UV.shape == (30, 2)
    assert np.allclose(np.linalg.norm(V[:6, 1:], axis=1), 0.1)      # the first ring is a circle of radius 0.1 about the path
    assert np.allclose(V[:6, 0], 0.0)
