"""Solidify the identity numbers before anything is concluded from them.

Every comparison this project has drawn from SFace cosine assumes the metric is
stable to the precision of the gaps being compared. That assumption has never
been tested. A 0.027 difference between two builds means nothing if the same
build measured from a slightly different camera swings by 0.05.

This runs the controls that decide whether the numbers carry weight:

* **Upper bound** — the photo against itself. Must be 1.000. If it is not, the
  embedding path is broken and nothing downstream is interpretable.
* **Re-encode stability** — the photo against a JPEG round-trip of itself. Shows
  how much the metric moves under a change that preserves identity completely.
* **Determinism** — the same build rendered and measured twice. Any difference
  here is pure noise in the pipeline.
* **View sensitivity** — each build measured from several camera angles. This is
  the number that decides whether build-to-build gaps are real, because it is
  the noise floor those gaps must clear.

Writes a JSON record and prints a table.

    python3 tools/benchmark.py in/face.jpg out/base.glb out/proj.glb out/inv.glb
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import render  # noqa: E402

from avatarforge import photo_fit  # noqa: E402

THRESHOLD = 0.363
VIEWS = ("portrait", "face", "front")
RENDER_SIZE = 640


def measure(reference_embedding: np.ndarray, image_path: Path) -> float | None:
    try:
        det = photo_fit.detect(image_path)
    except (ValueError, FileNotFoundError):
        return None
    return photo_fit.identity_similarity(reference_embedding, det.embedding)


def run(photo: Path, builds: list[Path]) -> dict:
    from PIL import Image

    reference = photo_fit.detect(photo)
    scratch = Path("out/.bench")
    scratch.mkdir(parents=True, exist_ok=True)

    controls: dict[str, float | None] = {}

    # Upper bound: the reference against itself.
    controls["photo_vs_self"] = photo_fit.identity_similarity(
        reference.embedding, reference.embedding
    )

    # A JPEG round-trip changes pixels but not identity. Whatever this costs is
    # the metric's sensitivity to pure encoding noise.
    round_trip = scratch / "reencoded.jpg"
    Image.open(photo).convert("RGB").save(round_trip, quality=85)
    controls["photo_vs_reencoded"] = measure(reference.embedding, round_trip)

    # Horizontal flip preserves identity for a human and often does not for a
    # recognition network. Worth knowing which case we are in.
    flipped = scratch / "flipped.jpg"
    Image.open(photo).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT).save(flipped, quality=95)
    controls["photo_vs_mirrored"] = measure(reference.embedding, flipped)

    results = []
    for glb in builds:
        per_view: dict[str, float | None] = {}
        for view in VIEWS:
            shot = scratch / f"{glb.stem}_{view}.png"
            render(glb, view=view, size=RENDER_SIZE).save(shot)
            per_view[view] = measure(reference.embedding, shot)

        # Determinism: render the same view again and remeasure.
        repeat = scratch / f"{glb.stem}_portrait_repeat.png"
        render(glb, view="portrait", size=RENDER_SIZE).save(repeat)
        repeat_score = measure(reference.embedding, repeat)

        scored = [v for v in per_view.values() if v is not None]
        results.append({
            "build": glb.name,
            "per_view": per_view,
            "mean": float(np.mean(scored)) if scored else None,
            "spread": float(max(scored) - min(scored)) if len(scored) > 1 else None,
            "determinism_delta": (
                None if repeat_score is None or per_view.get("portrait") is None
                else abs(repeat_score - per_view["portrait"])
            ),
            "views_detected": len(scored),
        })

    spreads = [r["spread"] for r in results if r["spread"] is not None]
    noise_floor = float(np.mean(spreads)) if spreads else None

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": str(photo),
        "render_size": RENDER_SIZE,
        "views": list(VIEWS),
        "threshold": THRESHOLD,
        "controls": controls,
        "builds": results,
        "noise_floor": noise_floor,
    }


def report(data: dict) -> None:
    c = data["controls"]
    print(f"reference: {data['reference']}   renders {data['render_size']}px\n")
    print("controls")
    for k, v in c.items():
        print(f"  {k:22} {'n/a' if v is None else f'{v:+.4f}'}")

    print(f"\n{'build':16}" + "".join(f"{v:>10}" for v in data["views"])
          + f"{'mean':>10}{'spread':>9}{'repeat':>9}")
    for r in data["builds"]:
        row = f"  {r['build']:14}"
        for v in data["views"]:
            s = r["per_view"].get(v)
            row += f"{'  n/a' if s is None else f'{s:+.4f}':>10}"
        row += f"{r['mean']:+10.4f}" if r["mean"] is not None else f"{'n/a':>10}"
        row += f"{r['spread']:9.4f}" if r["spread"] is not None else f"{'n/a':>9}"
        row += f"{r['determinism_delta']:9.4f}" if r["determinism_delta"] is not None else f"{'n/a':>9}"
        print(row)

    floor = data["noise_floor"]
    if floor is not None:
        print(f"\nnoise floor (mean spread across views): {floor:.4f}")
        means = [(r["build"], r["mean"]) for r in data["builds"] if r["mean"] is not None]
        means.sort(key=lambda kv: -kv[1])
        if len(means) > 1:
            gap = means[0][1] - means[1][1]
            verdict = "EXCEEDS the noise floor" if gap > floor else "is INSIDE the noise floor"
            print(f"top two: {means[0][0]} vs {means[1][0]}, gap {gap:.4f} — {verdict}")
            if gap <= floor:
                print("  -> that ranking is not supported by this measurement")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    data = run(Path(argv[0]), [Path(p) for p in argv[1:]])
    out = Path("out/benchmark.json")
    out.write_text(json.dumps(data, indent=2))
    report(data)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
