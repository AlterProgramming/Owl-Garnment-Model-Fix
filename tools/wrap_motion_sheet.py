"""Render gray A/B motion sheets from two *shipped* owl GLBs.

The purpose is to expose carrier attachment without kente texture or authored
surface detail hiding it.  Each row shows the same animation times from the
same camera for the control and candidate builds.

Example::

    PYTHONPATH=. python tools/wrap_motion_sheet.py \
      --control build/owl-guide-kente-wrap.glb \
      --candidate build/owl-guide-kente-coupled.glb \
      --out renders/coupling-ab

The script writes front/three-quarter and side sheets for ``idle``, ``wave``,
``tablet_show`` and ``hop`` when those clips exist in both assets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from meshforge.fastpreview import contact_sheet, render_flat
from tools.garment_diag import Scene


CLIPS = ("idle", "wave", "tablet_show", "hop")
YAWS = (25.0, 90.0)
TIMES = (0.0, 0.25, 0.50, 0.75, 1.0)


def _garments(sc: Scene) -> list[dict]:
    return [p for p in sc.prims if str(p["material"]).startswith("owl_kente_wrap")]


def _posed_scene(sc: Scene, clip: str, frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body = sc.prim("owl_body")
    over = sc.clip_overrides(clip, frac * sc.clip_duration(clip))
    chunks_v = [sc.pose(body, over)]
    chunks_f = [np.asarray(body["F"], dtype=np.int64)]
    chunks_c = [np.tile(np.array([[0.30, 0.30, 0.30]]), (len(body["V"]), 1))]
    offset = len(chunks_v[0])
    for p in _garments(sc):
        pv = sc.pose(p, over)
        chunks_v.append(pv)
        chunks_f.append(np.asarray(p["F"], dtype=np.int64) + offset)
        chunks_c.append(np.tile(np.array([[0.86, 0.86, 0.84]]), (len(pv), 1)))
        offset += len(pv)
    return np.vstack(chunks_v), np.vstack(chunks_f), np.vstack(chunks_c)


def _bounds(control: Scene, candidate: Scene) -> np.ndarray:
    pts = []
    for sc in (control, candidate):
        pts.append(sc.prim("owl_body")["V"])
        pts.extend(p["V"] for p in _garments(sc))
    P = np.vstack(pts)
    lo, hi = P.min(axis=0), P.max(axis=0)
    centre = 0.5 * (lo + hi)
    extent = 0.58 * (hi - lo)   # a small framing margin
    return np.stack([centre - extent, centre + extent])


def _frame(sc: Scene, clip: str, frac: float, yaw: float, bounds: np.ndarray,
           label: str, resolution: int = 420):
    V, F, C = _posed_scene(sc, clip, frac)
    im = render_flat(V, F, C, yaw_deg=yaw, pitch_deg=-2.0, resolution=resolution,
                     bounds=bounds, background=(24, 24, 24), fill_fraction=0.89)
    draw = ImageDraw.Draw(im)
    text = f"{label}  {clip}  {frac:0.2f}"
    draw.rectangle((8, 8, 8 + 9 * len(text), 34), fill=(10, 10, 10))
    draw.text((14, 13), text, fill=(242, 242, 242))
    return im


def render(control_path: Path, candidate_path: Path, out: Path) -> list[Path]:
    control, candidate = Scene(control_path), Scene(candidate_path)
    out.mkdir(parents=True, exist_ok=True)
    common = [c for c in CLIPS if c in control.anims and c in candidate.anims]
    if not common:
        raise ValueError("control and candidate have no common diagnostic clips")
    bounds = _bounds(control, candidate)
    paths = []
    overview = []
    for clip in common:
        for yaw in YAWS:
            frames = []
            for sc, label in ((control, "CONTROL"), (candidate, "COUPLED")):
                frames.extend(_frame(sc, clip, f, yaw, bounds, label) for f in TIMES)
            sheet = contact_sheet(frames, cols=len(TIMES), pad=5, background=(15, 15, 15))
            p = out / f"{clip}_yaw{int(yaw)}.png"
            sheet.save(p)
            paths.append(p)
            # Keep one representative three-quarter sheet per clip for a
            # single-page overview.
            if yaw == YAWS[0]:
                overview.append(sheet.resize((sheet.width // 2, sheet.height // 2)))
    if overview:
        p = out / "overview.png"
        contact_sheet(overview, cols=1, pad=8, background=(12, 12, 12)).save(p)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    for p in render(args.control, args.candidate, args.out):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
