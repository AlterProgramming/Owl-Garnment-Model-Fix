"""Gates are numbers with thresholds. Each test builds the smallest input
that makes one gate pass and one that makes it fail."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from meshforge import gates as G
from meshforge.drape import _Set
from meshforge.fk import Joint
from meshforge.rigexport import AnimationSpec, Track

NAMES = ["body", "chest", "wing_left"]
JOINTS = [Joint("body", None, np.array([0.0, 0.0, 0.0])), Joint("chest", "body", np.array([0.0, 0.5, 0.0])),
          Joint("wing_left", "chest", np.array([-0.3, 0.8, 0.0]))]


def _clip(joint, deg, axis="z"):
    times = np.array([0.0, 0.5, 1.0])
    q = Rotation.from_euler(axis, np.array([0.0, deg, 0.0])[:, None], degrees=True).as_quat()
    return AnimationSpec("probe", [Track(joint, "rotation", times, q)])


def _onehot(names, which):
    W = np.zeros((len(which), len(names)))
    for i, w in enumerate(which):
        W[i, names.index(w)] = 1.0
    return W


def test_g1_worn_is_zero_when_the_pins_move_with_the_torso_and_not_when_the_wing_carries_them():
    pins = np.array([[0.0, 0.9, 0.3], [0.1, 0.9, 0.3]])
    body_V = np.array([[0.0, 0.9, 0.28], [0.1, 0.9, 0.28], [-0.3, 0.9, 0.0]])
    body_W = _onehot(NAMES, ["chest", "chest", "wing_left"])
    torso = np.array([True, True, False])
    ok = G.gate_worn(pins, _onehot(NAMES, ["chest", "chest"]), body_V, body_W, torso, JOINTS, NAMES, [_clip("chest", 30)], H=2.0)
    assert ok.id == "G1" and ok.passed and ok.value < 1e-9
    bad = G.gate_worn(pins, _onehot(NAMES, ["wing_left", "wing_left"]), body_V, body_W, torso, JOINTS, NAMES, [_clip("wing_left", 30)], H=2.0)
    assert not bad.passed and bad.value > 0.01


def test_g2_support_only():
    assert G.gate_support_only(np.array([[0.5, 0.5, 0.0]]), NAMES).passed
    g = G.gate_support_only(np.array([[0.5, 0.4, 0.1]]), NAMES)
    assert not g.passed and g.value == 0.1


def test_g3_penetration_reads_the_worst_clip():
    posed = {"per_clip": {"idle": {"inside_count": 3, "min": 0.01}, "hop": {"inside_count": 25, "min": -0.02}}, "worst": {}}
    g = G.gate_penetration(posed)
    assert not g.passed and g.value == 25 and g.detail["clip"] == "hop"
    assert G.gate_penetration({"per_clip": {"idle": {"inside_count": 3, "min": 0.01}}, "worst": {}}).passed


def test_g4_hem_is_above_the_beads_and_level():
    hem = np.stack([np.zeros(10), np.full(10, 0.13) + 0.004 * np.sin(np.arange(10)), np.zeros(10)], axis=1)
    assert G.gate_hem(hem, y0=0.0, H=1.0, bead_top_y=0.10).passed
    assert not G.gate_hem(hem - [0, 0.02, 0], y0=0.0, H=1.0, bead_top_y=0.10).passed
    wavy = hem.copy()
    wavy[:, 1] += 0.05 * (np.arange(10) % 2)
    assert not G.gate_hem(wavy, y0=0.0, H=1.0, bead_top_y=0.10).passed


def test_g5_root_covered_by_a_cylinder_around_it():
    ph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([0.2 * np.cos(ph), np.zeros(24), 0.2 * np.sin(ph)], axis=1)
    cyl = trimesh.creation.cylinder(radius=0.3, height=1.0, sections=32)     # axis along z by default
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))   # now along y
    covered = G.gate_root_covered(ring, np.array([0.0, 0.0]), [(cyl.vertices, cyl.faces)])
    assert covered.passed and covered.value == 1.0
    bare = G.gate_root_covered(ring, np.array([0.0, 0.0]), [])
    assert not bare.passed and bare.value == 0.0


def test_g7_stretch():
    P = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    sets = [_Set(np.array([0, 1]), np.array([1, 2]), np.array([1.0, 1.0]), 0.92)]
    assert G.gate_stretch(P, sets, 0.92).passed
    assert not G.gate_stretch(P * 1.2, sets, 0.92).passed


def test_g8_collar_visible():
    theta = -np.pi + (np.arange(8) + 0.5) * (2 * np.pi / 8)
    rim = np.full(8, 0.80)
    axis = np.array([0.0, 0.0])
    assert G.gate_collar_visible(np.array([[0.0, 0.5, 0.4]]), theta, rim, axis).passed
    g = G.gate_collar_visible(np.array([[0.0, 0.85, 0.4]]), theta, rim, axis)
    assert not g.passed and g.value == 1


def test_table_lists_every_gate_with_its_verdict():
    gates = [G.Gate("G2", "support only", 0.0, "= 0", True, {}), G.Gate("G4", "hem", 0.11, ">= 0.122", False, {"min": 0.11})]
    text = G.format_table(gates)
    assert "G2" in text and "PASS" in text and "G4" in text and "FAIL" in text
