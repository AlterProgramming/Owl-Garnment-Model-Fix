"""Animation-to-garment coupling for the owl wrap.

The shipped GLB has no runtime cloth solver.  The PBD solve in ``drape.py``
creates a *rest shape* which is then exported as a skinned mesh.  That makes
skinning the garment's kinematic baseline, not a bookkeeping step.

The old wrap deliberately removed all wing influence and blended every column
from ``chest/body`` at the top to ``body`` at the hem.  It therefore satisfied
collision/clearance gates while the animated wings could move underneath an
already-wrinkled carrier.  In motion this reads as a character swimming inside
a clothing shell.

This module implements the hybrid baseline used by character-cloth systems:

    animated guide + controlled physical-looking residual

There is still no runtime simulation here.  Instead, the upper/contact part of
the baked cloth is allowed to inherit the *local* skin motion of the body and
wing roots it actually lies on, while the lower garment progressively returns
to the support-graph/body transport used by the old wrap.  The baked drape is
therefore a residual shape carried by a moving garment guide instead of an
independent world-like shell.

``install`` is intentionally opt-in.  The experimental entry point
``python -m meshforge.owl_pipeline_coupled`` installs it before delegating to
the production pipeline, so main remains a clean A/B control.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

GUIDE_JOINTS: tuple[str, ...] = (
    "body", "chest", "wing_left", "wing_left_tip", "wing_right",
)
UNRELATED_JOINTS: tuple[str, ...] = (
    "neck", "head", "cap", "tassel", "eye_left", "eye_right",
    "lid_left", "lid_right", "leg_left", "leg_right", "foot_left",
    "foot_right", "tail",
)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _restricted_local_skin(
    garment_vertices: np.ndarray,
    body_vertices: np.ndarray,
    body_weights: np.ndarray,
    joint_names: Sequence[str],
    *,
    k: int = 12,
    allowed: Sequence[str] = GUIDE_JOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the animated body skin under each garment vertex.

    ``transfer_body_weights`` historically pooled only pre-labelled torso
    vertices.  For coupling we intentionally query the complete character
    surface, then discard influences that cannot physically guide the wrap.
    This lets cloth near a fused wing root inherit that root's motion without
    giving the garment head/leg/eye weights merely because they are nearby.

    Returns ``(weights, nearest_distance)``.  Rows whose neighbours contain no
    allowed influence are left at zero; the caller falls back to the support
    graph for them.
    """
    G = np.asarray(garment_vertices, dtype=np.float64)
    B = np.asarray(body_vertices, dtype=np.float64)
    BW = np.asarray(body_weights, dtype=np.float64)
    if G.ndim != 2 or G.shape[1] != 3 or B.ndim != 2 or B.shape[1] != 3:
        raise ValueError("garment_vertices and body_vertices must be (n, 3)")
    if BW.shape != (len(B), len(joint_names)):
        raise ValueError("body_weights shape does not match body/joint counts")

    k = max(1, min(int(k), len(B)))
    dist, idx = cKDTree(B).query(G, k=k, workers=-1)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    inv = 1.0 / np.maximum(dist, 1e-5)
    inv /= np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
    local = np.einsum("nk,nkj->nj", inv, BW[idx])

    allow = np.array([name in set(allowed) for name in joint_names], dtype=bool)
    local[:, ~allow] = 0.0
    total = local.sum(axis=1, keepdims=True)
    valid = total[:, 0] > 1e-10
    local[valid] /= total[valid]
    local[~valid] = 0.0
    return local, dist[:, 0]


def _legacy_support_baseline(
    verts: np.ndarray,
    grid_index: np.ndarray,
    grid_shape: tuple[int, int],
    pins: np.ndarray,
    body_V: np.ndarray,
    body_W: np.ndarray,
    joint_names: Sequence[str],
    y_hip: float,
    *,
    pin_weights: np.ndarray | None = None,
    periodic: bool = True,
    sigma: float = 1.0,
) -> np.ndarray:
    """The old support-graph transport, kept as the free-cloth baseline."""
    names = list(joint_names)
    jb, jc = names.index("body"), names.index("chest")
    rows, cols = grid_shape
    idx = np.asarray(grid_index, dtype=np.int64)
    col = idx % cols
    if pin_weights is None:
        near = cKDTree(np.asarray(body_V, dtype=np.float64)).query(np.asarray(pins, dtype=np.float64), workers=-1)[1]
        Wp = np.zeros((cols, len(names)), dtype=np.float64)
        Wp[:, jb] = np.asarray(body_W)[near, jb]
        Wp[:, jc] = np.asarray(body_W)[near, jc]
        s = Wp.sum(axis=1)
        Wp[s < 1e-8, jc] = 1.0
        Wp /= np.maximum(Wp.sum(axis=1, keepdims=True), 1e-12)
    else:
        Wp = np.asarray(pin_weights, dtype=np.float64).copy()
        if Wp.shape != (cols, len(names)):
            raise ValueError(f"pin_weights must be ({cols}, {len(names)}), got {Wp.shape}")
        # A supplied pin map is a *hint*, not permission to couple to
        # unrelated joints.  Keep only body/chest for this baseline.
        keep = np.array([n in ("body", "chest") for n in names])
        Wp[:, ~keep] = 0.0
        empty = Wp.sum(axis=1) < 1e-8
        Wp[empty, jc] = 1.0
        Wp /= np.maximum(Wp.sum(axis=1, keepdims=True), 1e-12)

    y_top = np.asarray(pins, dtype=np.float64)[:, 1]
    s = np.clip(
        (y_top[col] - np.asarray(verts)[:, 1]) /
        np.maximum(y_top[col] - float(y_hip), 1e-6),
        0.0, 1.0,
    )
    W = (1.0 - s)[:, None] * Wp[col]
    W[:, jb] += s
    return _smooth_grid(W, idx, grid_shape, sigma=sigma, periodic=periodic)


def _smooth_grid(
    weights: np.ndarray,
    grid_index: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    sigma: float,
    periodic: bool,
) -> np.ndarray:
    """Smooth only through existing garment cells and renormalize."""
    W = np.asarray(weights, dtype=np.float64)
    rows, cols = grid_shape
    idx = np.asarray(grid_index, dtype=np.int64)
    G = np.zeros((rows * cols, W.shape[1]), dtype=np.float64)
    M = np.zeros(rows * cols, dtype=np.float64)
    G[idx], M[idx] = W, 1.0
    G, M = G.reshape(rows, cols, -1), M.reshape(rows, cols)
    if periodic:
        G, M = G[:, :-1], M[:, :-1]
    mode = "wrap" if periodic else "nearest"
    GM = gaussian_filter1d(
        gaussian_filter1d(G * M[..., None], sigma, axis=0, mode="nearest"),
        sigma, axis=1, mode=mode,
    )
    MM = gaussian_filter1d(
        gaussian_filter1d(M, sigma, axis=0, mode="nearest"),
        sigma, axis=1, mode=mode,
    )
    out = np.where(MM[..., None] > 1e-9, GM / np.maximum(MM[..., None], 1e-9), G)
    if periodic:
        out = np.concatenate([out, out[:, :1]], axis=1)
    out = out.reshape(rows * cols, -1)[idx]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def guide_weights(
    verts: np.ndarray,
    grid_index: np.ndarray,
    grid_shape: tuple[int, int],
    pins: np.ndarray,
    body_V: np.ndarray,
    body_W: np.ndarray,
    joint_names: Sequence[str],
    y_hip: float,
    pin_weights: np.ndarray | None = None,
    periodic: bool = True,
    sigma: float = 1.0,
    *,
    local_k: int = 12,
    contact_near_frac: float = 0.012,
    contact_far_frac: float = 0.095,
    release_start: float = 0.10,
    release_end: float = 0.66,
    upper_guide_floor: float = 0.34,
    contact_gain: float = 0.56,
) -> np.ndarray:
    """Hybrid skin weights for a baked cloth carrier.

    The old support graph remains the free-cloth baseline.  A local animated
    guide is then blended in according to two independent facts:

    * **vertical freedom** — upper cloth is worn/guided; freedom increases
      toward the hem;
    * **contact likelihood** — cloth already close to the character should
      inherit more of that local surface's animation than cloth hanging in
      free space.

    This is the static-GLB analogue of a Max-Distance / Anim-Drive field.  It
    deliberately does *not* turn the garment into a body decal: the blend goes
    to zero before the hem, and unrelated joints are excluded entirely.
    """
    V = np.asarray(verts, dtype=np.float64)
    idx = np.asarray(grid_index, dtype=np.int64)
    rows, cols = grid_shape
    if len(V) != len(idx):
        raise ValueError("grid_index must have one entry per garment vertex")

    base = _legacy_support_baseline(
        V, idx, grid_shape, pins, body_V, body_W, joint_names, y_hip,
        pin_weights=pin_weights, periodic=periodic, sigma=max(0.5, sigma),
    )
    local, distance = _restricted_local_skin(V, body_V, body_W, joint_names, k=local_k)

    H = float(np.ptp(np.asarray(body_V, dtype=np.float64)[:, 1]))
    H = max(H, 1e-6)
    row = (idx // cols).astype(np.float64) / max(rows - 1, 1)
    release = 1.0 - _smoothstep01((row - release_start) / max(release_end - release_start, 1e-6))
    contact = 1.0 - _smoothstep01(
        (distance / H - contact_near_frac) / max(contact_far_frac - contact_near_frac, 1e-6)
    )
    local_valid = local.sum(axis=1) > 1e-8

    # Even without direct contact, the upper carrier receives a modest local
    # guide so chest/torso pose is transported before the baked folds move.
    # Direct contact raises this toward ~0.9.  The lower garment releases to
    # the support/body baseline, preserving a visibly free hem.
    alpha = release * (upper_guide_floor + contact_gain * contact)
    alpha = np.where(local_valid, np.clip(alpha, 0.0, 0.92), 0.0)

    W = (1.0 - alpha)[:, None] * base + alpha[:, None] * local
    W = _smooth_grid(W, idx, grid_shape, sigma=max(0.7, sigma), periodic=periodic)

    # Hard safety invariant: no accidental face/head/leg/tail influence can
    # be introduced by smoothing or by a malformed input map.
    allowed = np.array([name in GUIDE_JOINTS for name in joint_names], dtype=bool)
    W[:, ~allowed] = 0.0
    empty = W.sum(axis=1) < 1e-10
    if empty.any():
        jb = list(joint_names).index("body")
        W[empty, jb] = 1.0
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)


def guide_diagnostics(W: np.ndarray, joint_names: Sequence[str], grid_index: np.ndarray,
                      grid_shape: tuple[int, int]) -> dict:
    """Small report used by tests/build logs and future visual gates."""
    W = np.asarray(W, dtype=np.float64)
    names = list(joint_names)
    rows, cols = grid_shape
    row = np.asarray(grid_index, dtype=np.int64) // cols
    upper = row <= max(1, int(round(0.35 * (rows - 1))))
    lower = row >= max(0, int(round(0.80 * (rows - 1))))
    wing_cols = [names.index(n) for n in ("wing_left", "wing_left_tip", "wing_right") if n in names]
    unrelated_cols = [names.index(n) for n in UNRELATED_JOINTS if n in names]
    return {
        "upper_wing_mass_mean": float(W[upper][:, wing_cols].sum(axis=1).mean()) if upper.any() and wing_cols else 0.0,
        "upper_wing_mass_max": float(W[upper][:, wing_cols].sum(axis=1).max()) if upper.any() and wing_cols else 0.0,
        "lower_wing_mass_mean": float(W[lower][:, wing_cols].sum(axis=1).mean()) if lower.any() and wing_cols else 0.0,
        "unrelated_mass": float(W[:, unrelated_cols].sum()) if unrelated_cols else 0.0,
    }


def _coupled_gate_worn(pins, pin_W, body_V, body_W, torso_mask, joints, joint_names, clips, H):
    """Replacement G1: compare a pin to its *local animated guide*.

    The legacy gate compared every pin only to chest/body and therefore
    penalized the very wing-root coupling required for a worn garment.
    """
    from meshforge import fk
    from meshforge import gates as gatesmod

    pins = np.asarray(pins, dtype=np.float64)
    body_V = np.asarray(body_V, dtype=np.float64)
    body_W = np.asarray(body_W, dtype=np.float64)
    near = cKDTree(body_V).query(pins, workers=-1)[1]
    anchors = body_V[near]
    aW = body_W[near].copy()
    allowed = np.array([n in GUIDE_JOINTS for n in joint_names])
    aW[:, ~allowed] = 0.0
    s = aW.sum(axis=1, keepdims=True)
    pin_W = np.asarray(pin_W, dtype=np.float64)
    aW = np.where(s > 1e-9, aW / np.maximum(s, 1e-9), pin_W)
    rest = pins - anchors
    worst, where = 0.0, {}
    for name, tm, skin in gatesmod.clip_skins(joints, joint_names, clips):
        p = fk.pose_vertices(pins, pin_W, joint_names, skin)
        a = fk.pose_vertices(anchors, aW, joint_names, skin)
        drift = np.linalg.norm((p - a) - rest, axis=1) / max(float(H), 1e-9)
        i = int(np.argmax(drift))
        if float(drift[i]) > worst:
            worst = float(drift[i])
            where = {"clip": name, "t": round(float(tm), 2), "column": i}
    return gatesmod.Gate("G1", "worn: guide-relative support drift", round(worst, 4), "<= 0.020 H", worst <= 0.020, where)


def _coupled_gate_support_only(W_all, joint_names):
    """Replacement G2: reject unrelated joints, not legitimate wing roots."""
    from meshforge import gates as gatesmod

    W = np.asarray(W_all, dtype=np.float64)
    cols = [i for i, n in enumerate(joint_names) if n in UNRELATED_JOINTS]
    mass = float(W[:, cols].sum()) if cols else 0.0
    by = {joint_names[i]: round(float(W[:, i].sum()), 5) for i in cols if W[:, i].sum() > 1e-8}
    return gatesmod.Gate("G2", "guide only: weight on unrelated joints", round(mass, 6), "= 0", mass <= 1e-8, by)


_INSTALLED = False


def install() -> None:
    """Install the coupled-wrap experiment into the existing pipeline.

    This is deliberately a narrow runtime patch rather than a rewrite of the
    mature mesh/drape code.  It changes only the hypotheses implicated by the
    motion failure: kinematic coupling and the amount of baked-in surplus.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from meshforge import drape, gates, wrap

    # Preserve the original symbols for explicit A/B/debug access.
    if not hasattr(wrap, "legacy_support_weights"):
        wrap.legacy_support_weights = wrap.support_weights
    if not hasattr(wrap, "LegacyWrapParams"):
        wrap.LegacyWrapParams = wrap.WrapParams
    if not hasattr(drape, "LegacyClothParams"):
        drape.LegacyClothParams = drape.ClothParams

    @dataclasses.dataclass(frozen=True)
    class CoupledWrapParams(wrap.LegacyWrapParams):
        # Broaden the visual support region while *reducing* pre-authored
        # bunching.  The motion should sell attachment, not static wrinkles.
        knot_drop: float = 0.012
        gather_half_deg: float = 34.0
        gather_ratio: float = 1.065
        arm_margin: float = 0.015
        hem_ease: float = 0.012
        hem_min_frac: float = 0.50
        sweep_band: float = 0.045
        sweep_blend_deg: float = 10.0
        sweep_drop_max: float = 0.025
        root_bound: bool = False
        knot_size: float = 0.080
        tail_width_frac: float = 0.125
        tail_curl_deg: float = 14.0

    @dataclasses.dataclass(frozen=True)
    class CoupledClothParams(drape.LegacyClothParams):
        # The exported GLB cannot generate new folds at runtime.  Keep the
        # rest drape calm enough that body-driven deformation remains legible.
        chest_ease: float = 0.045
        length_slack: float = 0.006
        bend: float = 0.32
        bend_wide: float = 0.22
        collide_offset: float = 0.011
        collide_offset_hem: float = 0.017

    wrap.support_weights = guide_weights
    wrap.WrapParams = CoupledWrapParams
    drape.ClothParams = CoupledClothParams
    gates.gate_worn = _coupled_gate_worn
    gates.gate_support_only = _coupled_gate_support_only
    _INSTALLED = True
