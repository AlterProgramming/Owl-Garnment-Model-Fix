"""Regression tests for the animation-coupled wrap baseline.

These tests deliberately avoid beauty-render assertions.  They encode the
mechanical contract the previous wrap violated: upper cloth next to a moving
wing root must inherit some of that root's animation, unrelated joints must
never leak in, and the hem must still release back to body transport.
"""
from __future__ import annotations

import numpy as np
import pytest

from meshforge import coupling as C


NAMES = ["body", "chest", "wing_left", "wing_left_tip", "wing_right", "head", "leg_left"]
ROWS, COLS = 9, 5   # final column duplicates the seam, like the real wrap


def _fixture():
    pins = np.array([
        [-0.35, 1.00, 0.00],
        [-0.15, 1.00, 0.00],
        [ 0.15, 1.00, 0.00],
        [ 0.35, 1.00, 0.00],
        [-0.35, 1.00, 0.00],  # periodic seam copy of column zero
    ], dtype=float)
    verts = np.vstack([
        pins + np.array([0.0, -0.90 * r / (ROWS - 1), 0.030 * r / (ROWS - 1)])
        for r in range(ROWS)
    ])

    body = np.array([
        [-0.35, 1.00, 0.00],
        [-0.15, 1.00, 0.00],
        [ 0.15, 1.00, 0.00],
        [ 0.35, 1.00, 0.00],
        [-0.35, 0.55, 0.00],
        [-0.15, 0.55, 0.00],
        [ 0.15, 0.55, 0.00],
        [ 0.35, 0.55, 0.00],
        [-0.35, 0.10, 0.00],
        [-0.15, 0.10, 0.00],
        [ 0.15, 0.10, 0.00],
        [ 0.35, 0.10, 0.00],
        [ 0.00, 1.05, 0.01],   # deliberately nearby head-only sample
    ], dtype=float)
    W = np.zeros((len(body), len(NAMES)), dtype=float)
    # left upper/root skin follows the moving wing plus chest
    W[0:2, NAMES.index("wing_left")] = 0.72
    W[0:2, NAMES.index("chest")] = 0.28
    # right upper/root skin follows tablet wing plus chest
    W[2:4, NAMES.index("wing_right")] = 0.65
    W[2:4, NAMES.index("chest")] = 0.35
    # middle body is chest/body blended
    W[4:8, NAMES.index("chest")] = 0.55
    W[4:8, NAMES.index("body")] = 0.45
    # lower body rides the root/body
    W[8:12, NAMES.index("body")] = 1.0
    W[12, NAMES.index("head")] = 1.0
    return verts, pins, body, W


def _weights(verts, pins, body, body_W):
    return C.guide_weights(
        verts, np.arange(len(verts)), (ROWS, COLS), pins,
        body, body_W, NAMES, y_hip=0.30,
        local_k=1, sigma=0.01,
    )


def test_upper_contact_inherits_wing_root_motion():
    verts, pins, body, body_W = _fixture()
    W = _weights(verts, pins, body, body_W)
    left = NAMES.index("wing_left")
    right = NAMES.index("wing_right")
    # This is the regression the old support_weights could never pass: it
    # forced both quantities to exactly zero.  Ignore the duplicated seam
    # column when checking the right side.
    assert W[:COLS, left].mean() > 0.25
    assert W[2:4, right].mean() > 0.30


def test_hem_releases_back_to_body_transport():
    verts, pins, body, body_W = _fixture()
    W = _weights(verts, pins, body, body_W)
    body_i = NAMES.index("body")
    wing_i = [NAMES.index("wing_left"), NAMES.index("wing_left_tip"), NAMES.index("wing_right")]
    assert np.all(W[-COLS:, body_i] > 0.98)
    assert np.all(W[-COLS:, wing_i].sum(axis=1) < 0.02)


def test_unrelated_nearby_joint_cannot_leak_into_garment():
    verts, pins, body, body_W = _fixture()
    # Put a top cloth point almost on the head-only decoy.  Filtering must
    # fall back to the support baseline rather than importing head motion.
    verts = verts.copy()
    verts[1] = body[-1]
    W = _weights(verts, pins, body, body_W)
    assert W[:, NAMES.index("head")].sum() == pytest.approx(0.0)
    assert W[:, NAMES.index("leg_left")].sum() == pytest.approx(0.0)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_diagnostics_separate_upper_coupling_from_free_hem():
    verts, pins, body, body_W = _fixture()
    W = _weights(verts, pins, body, body_W)
    d = C.guide_diagnostics(W, NAMES, np.arange(len(verts)), (ROWS, COLS))
    assert d["upper_wing_mass_max"] > 0.25
    assert d["lower_wing_mass_mean"] < 0.02
    assert d["unrelated_mass"] == pytest.approx(0.0)


def test_install_is_idempotent_and_changes_only_experiment_symbols():
    from meshforge import drape, gates, wrap

    C.install()
    first = wrap.support_weights
    C.install()
    assert wrap.support_weights is first is C.guide_weights
    assert wrap.WrapParams().gather_ratio < wrap.LegacyWrapParams().gather_ratio
    assert drape.ClothParams().length_slack < drape.LegacyClothParams().length_slack
    # run_gates resolves these global functions at call time, so replacing
    # them is enough; the production implementation remains available in
    # main for the A/B build.
    assert gates.gate_worn is C._coupled_gate_worn
    assert gates.gate_support_only is C._coupled_gate_support_only
