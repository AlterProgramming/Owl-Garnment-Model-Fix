"""Measure whether a rendered avatar actually resembles the source photo.

Renders a straight-on portrait, runs the same face detector and recognition
network over it as over the photograph, and reports the cosine similarity of
the two embeddings.

This exists because "does it look like him?" is not a question I can answer
about my own output. Every judgement in this pipeline that relied on looking at
a picture has been wrong at least once. A number that moves when the pipeline
changes is worth more than a confident opinion.

The threshold is SFace's own published same-person cutoff for photographs
(0.363). A render is not a photograph, so treat it as a relative yardstick
between builds, not a claim of recognition.

    python3 tools/identity.py in/face.jpg out/base.glb out/proj.glb
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import render  # noqa: E402

from avatarforge import photo_fit  # noqa: E402

THRESHOLD = 0.363


def score(photo: Path, glb: Path, view: str = "portrait", size: int = 900) -> dict:
    reference = photo_fit.detect(photo)

    shot = Path("out") / f".identity_{glb.stem}.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    render(glb, view=view, size=size).save(shot)

    try:
        rendered = photo_fit.detect(shot)
    except ValueError:
        return {"glb": glb.name, "detected": False, "cosine": None, "render": str(shot)}

    return {
        "glb": glb.name,
        "detected": True,
        "detector_score": rendered.score,
        "cosine": photo_fit.identity_similarity(reference.embedding, rendered.embedding),
        "render": str(shot),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    photo = Path(argv[0])
    results = [score(photo, Path(g)) for g in argv[1:]]

    print(f"reference: {photo}")
    print(f"{'build':22} {'face found':>10} {'cosine':>9}   verdict")
    for r in results:
        if not r["detected"]:
            print(f"{r['glb']:22} {'no':>10} {'-':>9}   detector found no face in the render")
            continue
        verdict = "above same-person threshold" if r["cosine"] >= THRESHOLD else "below threshold"
        print(f"{r['glb']:22} {r['detector_score']:10.3f} {r['cosine']:+9.4f}   {verdict}")

    scored = [r for r in results if r["detected"]]
    if len(scored) > 1:
        best = max(scored, key=lambda r: r["cosine"])
        print(f"\nbest: {best['glb']} at {best['cosine']:+.4f}")
    print(f"(SFace same-person threshold for photographs: {THRESHOLD})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
