"""Select the most useful frames from a capture clip.

A seventeen-second clip is about five hundred frames, of which perhaps eight
are worth baking. Picking them by hand means picking whichever moments you
happened to pause on; picking them by measurement means covering the poses the
single frontal photograph could not reach.

Three filters, applied in order:

* **Sharpness** — variance of the Laplacian. Turning your head is exactly when
  motion blur happens, so the frames with the most new information are also the
  ones most likely to be soft.
* **Detection confidence** — a frame the detector is unsure about will produce
  a bad pose solve, and a bad pose smears the texture rather than filling it.
* **Pose spread** — frames are then chosen greedily to maximise distance in
  (yaw, pitch) from everything already selected. Ten frames clustered near
  frontal are worth barely more than one.

    python3 tools/frames.py "~/Documents/Movie on 7-27-26 at 4.52 PM.mov"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = Path(__file__).resolve().parent.parent / "models"
# Sharpness is judged relative to the clip, not against a fixed number. Video
# frames are softer than stills by construction — this clip runs 20-26 variance
# where a still JPEG runs far higher — so an absolute floor calibrated on
# photographs rejects every frame of every video.
SHARPNESS_PERCENTILE = 45.0
MIN_SCORE = 0.80


def head_angles(lm: np.ndarray) -> tuple[float, float]:
    """Rough yaw and pitch from landmark geometry — enough to spread a selection."""
    eye_r = lm[36:42].mean(axis=0)
    eye_l = lm[42:48].mean(axis=0)
    nose = lm[30]
    span = np.linalg.norm(eye_l - eye_r) + 1e-6
    yaw = float(((nose[0] - (eye_l[0] + eye_r[0]) / 2) / span) * 90.0)
    brow = lm[17:27, 1].mean()
    chin = lm[8][1]
    pitch = float(((nose[1] - brow) / max(chin - brow, 1e-6) - 0.52) * 160.0)
    return yaw, pitch


def main(argv: list[str] | None = None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=str)
    ap.add_argument("--out", type=Path, default=Path("in/capture"))
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--clear", action="store_true", default=True)
    args = ap.parse_args(argv)

    path = Path(args.video).expanduser()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"could not open {path}", file=sys.stderr)
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{path.name}: {total} frames, {fps:.1f} fps, {w}x{h}  ({total / fps:.1f}s)")

    detector = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (w, h), 0.6, 0.3, 5000)
    marker = cv2.face.createFacemarkLBF()
    marker.loadModel(str(MODELS / "lbfmodel.yaml"))

    # First pass: measure this clip's own sharpness distribution.
    sample = []
    probe = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if probe % args.stride == 0:
            sample.append(float(cv2.Laplacian(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
        probe += 1
    floor = float(np.percentile(sample, SHARPNESS_PERCENTILE)) if sample else 0.0
    print(f"  sharpness: median {np.median(sample):.1f}, keeping frames above "
          f"{floor:.1f} (p{SHARPNESS_PERCENTILE:.0f})")
    cap.release()
    cap = cv2.VideoCapture(str(path))

    candidates = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.stride == 0:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
            if sharp >= floor:
                _, faces = detector.detect(frame)
                if faces is not None and len(faces):
                    face = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
                    if face[-1] >= MIN_SCORE:
                        got, marks = marker.fit(frame, np.array([face[:4].astype(np.int32)]))
                        if got:
                            yaw, pitch = head_angles(marks[0][0].astype(np.float64))
                            candidates.append({
                                "index": index, "frame": frame.copy(),
                                "sharp": sharp, "score": float(face[-1]),
                                "yaw": yaw, "pitch": pitch,
                            })
        index += 1
    cap.release()

    if not candidates:
        print("no usable frames — every frame was soft or undetected", file=sys.stderr)
        return 1

    yaws = np.array([c["yaw"] for c in candidates])
    pitches = np.array([c["pitch"] for c in candidates])
    print(f"usable frames: {len(candidates)} of {total // args.stride} sampled")
    print(f"  yaw range   {yaws.min():+.1f} to {yaws.max():+.1f} deg")
    print(f"  pitch range {pitches.min():+.1f} to {pitches.max():+.1f} deg")

    # Greedy farthest-point selection in pose space, seeded with the sharpest
    # near-frontal frame so the set always contains a good reference view.
    pose = np.stack([yaws, pitches], axis=1)
    frontal = np.argmin(np.abs(yaws) + np.abs(pitches) - 0.02 * np.array([c["sharp"] for c in candidates]))
    chosen = [int(frontal)]
    while len(chosen) < min(args.count, len(candidates)):
        d = np.min(np.linalg.norm(pose[:, None, :] - pose[chosen][None, :, :], axis=2), axis=1)
        # Break ties toward sharper frames rather than arbitrarily.
        d = d + 0.01 * np.array([c["sharp"] for c in candidates]) / max(1.0, np.max([c["sharp"] for c in candidates]))
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))

    args.out.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for old in args.out.glob("*"):
            if old.is_file():
                old.unlink()

    print(f"\nselected {len(chosen)}:")
    for n, i in enumerate(sorted(chosen, key=lambda k: candidates[k]["index"]), 1):
        c = candidates[i]
        dst = args.out / f"view{n:02d}.jpg"
        cv2.imwrite(str(dst), c["frame"], [cv2.IMWRITE_JPEG_QUALITY, 96])
        print(f"  {dst.name}  frame {c['index']:4d}  yaw {c['yaw']:+6.1f}  "
              f"pitch {c['pitch']:+6.1f}  sharp {c['sharp']:6.0f}  score {c['score']:.2f}")
    print(f"\nwrote {len(chosen)} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
