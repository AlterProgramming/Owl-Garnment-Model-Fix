"""Measure whether the shipped wrap moves *with* the owl or around it.

This is deliberately different from a clearance/cloth-quality diagnostic.  It
reads the final GLB and compares the displacement of each wrap-sheet vertex to
the displacement of the animated body surface directly underneath it.

Run it on the existing control build and on the coupled experiment::

    PYTHONPATH=. python tools/wrap_coupling_diag.py build/owl-kente-wrap.glb \
        --out probes/coupling-control.json
    PYTHONPATH=. python tools/wrap_coupling_diag.py build/owl-kente-coupled.glb \
        --out probes/coupling-coupled.json

The most useful fields are ``carrier_static_fraction`` and
``cosine_to_body``.  A broad upper garment should not remain almost stationary
while the nearby animated body/wing root moves.  The hem is intentionally freer
and is reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tools.garment_diag import Scene, body_frame, stats


BANDS = (
    ("upper", 0.32, 0.55),
    ("middle", 0.18, 0.32),
    ("hem", 0.08, 0.18),
)


def _dense_weights(sc: Scene, prim: dict) -> np.ndarray:
    W = np.zeros((len(prim["V"]), len(sc.joint_names)), dtype=np.float64)
    for k in range(prim["J"].shape[1]):
        np.add.at(W, (np.arange(len(W)), prim["J"][:, k].astype(int)), prim["W"][:, k])
    return W


def _sheet(sc: Scene) -> dict:
    matches = [p for p in sc.prims if p["material"] == "owl_kente_wrap"]
    if not matches:
        raise ValueError("GLB has no material named 'owl_kente_wrap'")
    # knot and sheet share the material.  The sheet is overwhelmingly the
    # largest primitive, so choose by vertex count instead of file order.
    return max(matches, key=lambda p: len(p["V"]))


def _band_masks(V: np.ndarray, frame: dict) -> dict[str, np.ndarray]:
    yfrac = (V[:, 1] - frame["y0"]) / frame["H"]
    out = {name: (yfrac >= lo) & (yfrac < hi) for name, lo, hi in BANDS}
    out["all"] = np.ones(len(V), dtype=bool)
    return out


def _safe_stat(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return stats(x) if len(x) else {"n": 0}


def analyze(path: Path, samples_per_clip: int = 13) -> dict:
    sc = Scene(path)
    body = sc.prim("owl_body")
    sheet = _sheet(sc)
    frame = body_frame(body["V"])
    H = frame["H"]
    bands = _band_masks(sheet["V"], frame)

    body_tree = cKDTree(body["V"])
    distance, nearest = body_tree.query(sheet["V"], workers=-1)
    Wg = _dense_weights(sc, sheet)
    Wb_all = _dense_weights(sc, body)
    Wb = Wb_all[nearest]
    names = np.array(sc.joint_names)
    wing_cols = np.flatnonzero(np.isin(names, ["wing_left", "wing_left_tip", "wing_right"]))
    unrelated_cols = np.flatnonzero(np.isin(names, [
        "head", "neck", "cap", "tassel", "eye_left", "eye_right", "lid_left", "lid_right",
        "leg_left", "leg_right", "foot_left", "foot_right", "tail",
    ]))

    mismatch = 0.5 * np.abs(Wg - Wb).sum(axis=1)
    wing_mass = Wg[:, wing_cols].sum(axis=1) if len(wing_cols) else np.zeros(len(Wg))
    unrelated_mass = Wg[:, unrelated_cols].sum(axis=1) if len(unrelated_cols) else np.zeros(len(Wg))

    result = {
        "file": str(path),
        "frame": frame,
        "sheet_vertices": int(len(sheet["V"])),
        "rest_nearest_body_distance": {k: _safe_stat(distance[m]) for k, m in bands.items()},
        "weight_mismatch_vs_nearest_body": {k: _safe_stat(mismatch[m]) for k, m in bands.items()},
        "wing_weight_mass": {k: _safe_stat(wing_mass[m]) for k, m in bands.items()},
        "unrelated_weight_mass_total": float(unrelated_mass.sum()),
        "clips": {},
    }

    V0, B0 = sheet["V"], body["V"]
    for clip in sc.anims:
        duration = sc.clip_duration(clip)
        frames = []
        for tm in np.linspace(0.0, duration, samples_per_clip):
            over = sc.clip_overrides(clip, float(tm))
            Vp = sc.pose(sheet, over)
            Bp = sc.pose(body, over)
            dG = Vp - V0
            dB = (Bp - B0)[nearest]
            mg = np.linalg.norm(dG, axis=1)
            mb = np.linalg.norm(dB, axis=1)
            moving = mb > 0.0015 * H
            cosine = np.full(len(V0), np.nan)
            ratio = np.full(len(V0), np.nan)
            cosine[moving] = np.einsum("ij,ij->i", dG[moving], dB[moving]) / np.maximum(mg[moving] * mb[moving], 1e-12)
            ratio[moving] = mg[moving] / np.maximum(mb[moving], 1e-12)
            # Direct perceptual signature of "owl moves under shell": nearby
            # body has meaningful motion while the garment carrier has less
            # than a quarter of that displacement.
            carrier_static = moving & (mg < 0.25 * mb)
            rec = {"time": round(float(tm), 4), "moving_fraction": float(moving.mean())}
            for name, mask in bands.items():
                active = mask & moving
                rec[name] = {
                    "moving_vertices": int(active.sum()),
                    "cosine_to_body": _safe_stat(cosine[active]),
                    "magnitude_ratio": _safe_stat(ratio[active]),
                    "carrier_static_fraction": float((carrier_static & mask).sum() / max(active.sum(), 1)),
                }
            frames.append(rec)
        # Keep the full series.  A single best/worst frame hides phase errors.
        result["clips"][clip] = {"duration": float(duration), "frames": frames}

    # A compact ranking for CI/review.  Ignore time zero and frames where
    # almost nothing under the garment moves.
    candidates = []
    for clip, data in result["clips"].items():
        for rec in data["frames"]:
            u = rec["upper"]
            if rec["time"] > 0 and u["moving_vertices"] >= 8:
                candidates.append((u["carrier_static_fraction"], clip, rec["time"], u))
    if candidates:
        worst = max(candidates, key=lambda x: x[0])
        result["worst_upper_shell_frame"] = {
            "clip": worst[1], "time": worst[2], **worst[3],
        }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=13, help="samples per animation clip")
    args = ap.parse_args(argv)
    result = analyze(args.glb, samples_per_clip=max(3, args.samples))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "file": result["file"],
        "unrelated_weight_mass_total": result["unrelated_weight_mass_total"],
        "upper_wing_weight": result["wing_weight_mass"]["upper"],
        "worst_upper_shell_frame": result.get("worst_upper_shell_frame"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
