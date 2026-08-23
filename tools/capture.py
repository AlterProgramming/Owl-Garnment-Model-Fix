"""Multi-view capture for photo fitting.

A single frontal photograph sees roughly half the head; everything at the sides,
under the chin and behind the ears keeps the base texture. Turning the head
under a fixed camera is the cheapest way to fill that in.

This opens a preview window and captures automatically. It watches head yaw and
pitch, and banks a frame the first time each target pose is held steadily and
sharply. Nothing is saved on a blurred or badly-detected frame, so there is no
manual timing to get right — just move slowly and watch the checklist fill in.

    python3 tools/capture.py            # saves into in/capture/
    python3 tools/capture.py --out in/capture --views front,left,right,up,down

Press q to stop early; whatever was banked is kept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avatarforge import photo_fit  # noqa: E402

# Target head poses, in degrees of yaw and pitch relative to the frontal solve.
# Tolerances are generous: the point is coverage, not precision.
VIEWS = {
    "front": {"yaw": 0.0, "pitch": 0.0, "tol": 8.0},
    "left":  {"yaw": -28.0, "pitch": 0.0, "tol": 12.0},
    "right": {"yaw": 28.0, "pitch": 0.0, "tol": 12.0},
    "up":    {"yaw": 0.0, "pitch": -18.0, "tol": 10.0},
    "down":  {"yaw": 0.0, "pitch": 18.0, "tol": 10.0},
}

MIN_SHARPNESS = 55.0     # variance of Laplacian; below this the frame is soft
MIN_SCORE = 0.85         # detector confidence
HOLD_FRAMES = 4          # consecutive good frames before banking


def sharpness(gray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main(argv: list[str] | None = None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("in/capture"))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args(argv)

    wanted = [v.strip() for v in args.views.split(",") if v.strip() in VIEWS]
    args.out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("could not open the camera — grant camera permission and retry", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, frame = cap.read()
    if not ok:
        print("camera opened but returned no frame", file=sys.stderr)
        return 2
    h, w = frame.shape[:2]

    models = Path(__file__).resolve().parent.parent / "models"
    detector = cv2.FaceDetectorYN.create(str(models / "yunet.onnx"), "", (w, h), 0.7, 0.3, 5000)
    marker = cv2.face.createFacemarkLBF()
    marker.loadModel(str(models / "lbfmodel.yaml"))

    banked: dict[str, Path] = {}
    hold = {v: 0 for v in wanted}
    print("move slowly; each view banks itself when held steady. q to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = sharpness(gray)

        _, faces = detector.detect(frame)
        status, yaw, pitch = "no face", None, None

        if faces is not None and len(faces):
            face = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
            if face[-1] >= MIN_SCORE:
                got, marks = marker.fit(frame, np.array([face[:4].astype(np.int32)]))
                if got:
                    lm = marks[0][0].astype(np.float64)
                    # Yaw from the asymmetry of the eye-to-nose distances; pitch
                    # from where the nose sits between the brow and chin. Crude
                    # next to solvePnP, but it only has to gate a capture.
                    eye_r = lm[36:42].mean(axis=0)
                    eye_l = lm[42:48].mean(axis=0)
                    nose = lm[30]
                    span = np.linalg.norm(eye_l - eye_r) + 1e-6
                    yaw = float(((nose[0] - (eye_l[0] + eye_r[0]) / 2) / span) * 90.0)
                    brow = lm[17:27, 1].mean()
                    chin = lm[8][1]
                    ratio = (nose[1] - brow) / max(chin - brow, 1e-6)
                    pitch = float((ratio - 0.52) * 160.0)

                    for view in wanted:
                        if view in banked:
                            continue
                        spec = VIEWS[view]
                        near = (abs(yaw - spec["yaw"]) <= spec["tol"]
                                and abs(pitch - spec["pitch"]) <= spec["tol"])
                        if near and sharp >= MIN_SHARPNESS:
                            hold[view] += 1
                            if hold[view] >= HOLD_FRAMES:
                                path = args.out / f"{view}.jpg"
                                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
                                banked[view] = path
                                print(f"  banked {view:6} yaw {yaw:+6.1f}  pitch {pitch:+6.1f}"
                                      f"  sharpness {sharp:.0f}")
                        else:
                            hold[view] = 0
                    status = f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  sharp {sharp:.0f}"
                else:
                    status = "landmarks failed"
            else:
                status = f"low confidence {face[-1]:.2f}"

        view = frame.copy()
        cv2.putText(view, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 240, 120), 2)
        for i, name in enumerate(wanted):
            done = name in banked
            cv2.putText(view, f"[{'x' if done else ' '}] {name}", (16, 74 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (60, 240, 120) if done else (200, 200, 200), 2)
        cv2.imshow("avatarforge capture - q to finish", view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(banked) == len(wanted):
            print("all views captured")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{len(banked)} of {len(wanted)} views saved to {args.out}")
    for name, path in banked.items():
        print(f"  {name:6} {path}")
    if not banked:
        print("nothing captured — try better lighting, or lower MIN_SHARPNESS")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
