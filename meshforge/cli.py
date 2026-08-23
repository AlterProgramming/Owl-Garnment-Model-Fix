"""Command-line entry point wiring clean -> segment -> skin -> animate ->
export -> validate -> preview into one run against a real GLB.

    python3 -m meshforge in.glb --out out.glb --rig wing --clip wave
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from meshforge.animate import blink_clip, wave_clip
from meshforge.clean import weld_and_clean
from meshforge.export import write_multi_skinned_glb
from meshforge.eyes import (
    LEFT_EYE_SOCKET,
    RIGHT_EYE_SOCKET,
    build_eyeball,
    build_eyelid,
    eye_pivot,
    eyelid_blink_axis_angle,
    neutralize_old_eye_paint,
)
from meshforge.segment import BBoxRegion, find_articulated_part
from meshforge.skin import compute_multi_part_skin_weights, pose_vertices_multi_lbs
from meshforge.preview import render_orthographic
from meshforge.validate import validate

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])

# Fractional bbox region for the owl's raised wing (mesh -X side; screen-left
# in the front render), determined empirically against the real
# owl-mascot.glb asset — see task-8-report.md for the render-by-render
# narrowing process. Isolates a single elongated, feather-textured part
# attached at one flat-cut edge (10629 faces out of 149880 total); confirmed
# by rendering the isolated part alone (front and 90-degree views) and
# looking at it, not just checking that find_articulated_part didn't raise.
# A looser region (full Z half, wider X/Y) pulled in a slab of white belly
# along the cut edge — this tighter box excludes that bleed-through.
WING_REGION = BBoxRegion(x=(0.0, 0.30), y=(0.35, 0.70), z=(0.0, 1.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meshforge", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="raw TRELLIS-style GLB to process")
    parser.add_argument("--out", type=Path, required=True, help="output rigged/animated GLB path")
    parser.add_argument("--rig", choices=["wing"], default="wing", help="which part to rig (only 'wing' implemented)")
    parser.add_argument("--clip", choices=["wave"], default="wave", help="which animation clip to bake (only 'wave' implemented)")
    parser.add_argument("--preview-dir", type=Path, default=None, help="directory to write sanity-check PNG renders into")
    parser.add_argument(
        "--region", type=str, default=None,
        help="override WING_REGION: 'x0,x1,y0,y1,z0,z1' fractional bbox coordinates",
    )
    return parser


def _parse_region(spec: str) -> BBoxRegion:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"--region must be six comma-separated floats 'x0,x1,y0,y1,z0,z1', got {spec!r}"
        )
    x0, x1, y0, y1, z0, z1 = (float(p) for p in parts)
    return BBoxRegion(x=(x0, x1), y=(y0, y1), z=(z0, z1))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    raw = trimesh.load(str(args.input))
    if isinstance(raw, trimesh.Scene):
        # dump(concatenate=True) applies each geometry's scene-graph node
        # transform before combining, so a mesh with a non-identity node
        # transform is processed in the right coordinate space, and no
        # geometry is silently dropped the way picking geometry.values()[0]
        # would.
        raw = raw.dump(concatenate=True)

    region = _parse_region(args.region) if args.region is not None else WING_REGION

    cleaned = weld_and_clean(raw)
    part_result = find_articulated_part(cleaned, region)
    body_wing_weights = compute_multi_part_skin_weights(cleaned, {"wing": part_result.part_face_mask})
    wing_keyframes = wave_clip(part_result.pivot_axis)

    # The wing skin is blended (compute_multi_part_skin_weights) because
    # it's cut out of the body's own surface. Eyes and lids are
    # different: new geometry appended alongside the body, not carved
    # from it, so each is simply 100% its own joint — there's no seam to
    # blend, and no shared vertices with the body to derive a blend band
    # from.
    eye_sockets = [RIGHT_EYE_SOCKET, LEFT_EYE_SOCKET]
    body_mesh = neutralize_old_eye_paint(cleaned, eye_sockets)

    lid_axis, lid_angle_rad = eyelid_blink_axis_angle()
    blink_keyframes = blink_clip(lid_axis, close_degrees=np.degrees(lid_angle_rad))

    rigid_parts = [
        ("eye_right", build_eyeball(RIGHT_EYE_SOCKET), eye_pivot(RIGHT_EYE_SOCKET), None),
        ("eye_left", build_eyeball(LEFT_EYE_SOCKET), eye_pivot(LEFT_EYE_SOCKET), None),
        ("lid_right", build_eyelid(RIGHT_EYE_SOCKET), eye_pivot(RIGHT_EYE_SOCKET), blink_keyframes),
        ("lid_left", build_eyelid(LEFT_EYE_SOCKET), eye_pivot(LEFT_EYE_SOCKET), blink_keyframes),
    ]
    combined = trimesh.util.concatenate([body_mesh] + [mesh for _, mesh, _, _ in rigid_parts])

    n_body = len(body_mesh.vertices)
    total_rigid = sum(len(mesh.vertices) for _, mesh, _, _ in rigid_parts)

    def _pad_body_weight(body_weight: np.ndarray) -> np.ndarray:
        return np.concatenate([body_weight, np.zeros(total_rigid)])

    joint_weights = {
        "body": _pad_body_weight(body_wing_weights["body"]),
        "wing": _pad_body_weight(body_wing_weights["wing"]),
    }
    joint_pivots = {"wing": part_result.pivot_point}
    # Eyes get no animation entry: they're runtime-aimed (a real bone,
    # rotated by the viewer toward the cursor/target), not a baked clip,
    # so they export skinned but static at their identity bind pose.
    # Lids share one baked blink clip each, in sync (real blinks are).
    animations = {"wing": wing_keyframes}

    offset = 0
    for name, mesh, pivot, keyframes in rigid_parts:
        n = len(mesh.vertices)
        weight = np.zeros(n_body + total_rigid)
        weight[n_body + offset: n_body + offset + n] = 1.0
        joint_weights[name] = weight
        joint_pivots[name] = pivot
        if keyframes is not None:
            animations[name] = keyframes
        offset += n

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_multi_skinned_glb(combined, joint_weights, joint_pivots, animations, out_path=str(args.out))

    print(f"wrote {args.out}")
    result = validate(str(args.out))
    if result != 0:
        return result

    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

        vertices = combined.vertices
        faces = combined.faces
        colors = combined.visual.vertex_colors[:, :3].astype(np.float32) / 255.0

        def _posed(rotations: dict) -> np.ndarray:
            full = {name: _IDENTITY_QUAT for name in joint_pivots}
            full.update(rotations)
            return pose_vertices_multi_lbs(vertices, joint_weights, joint_pivots, full)

        # Sample rest (time=0), roughly mid-clip, and peak of the wing's
        # wave so a wrong pivot or a bad weight blend is visible in the
        # previews, not just the unanimated rest pose. Also render the
        # blink's peak (fully closed) as its own frame — the wing preview
        # loop can't reach it, since it only ever samples wing rotations.
        n = len(wing_keyframes)
        wing_sample_indices = sorted({
            0, n // 2, int(np.argmax([np.linalg.norm(kf.rotation[:3]) for kf in wing_keyframes])),
        })
        wing_sample_labels = {0: "rest", n // 2: "mid"}
        blink_only = blink_keyframes[:-1]  # exclude the trailing held-open keyframe
        blink_peak_idx = len(blink_only) // 2

        poses = {
            wing_sample_labels.get(i, f"t{wing_keyframes[i].time:.2f}"): {"wing": wing_keyframes[i].rotation}
            for i in wing_sample_indices
        }
        poses["blink"] = {
            "lid_right": blink_only[blink_peak_idx].rotation,
            "lid_left": blink_only[blink_peak_idx].rotation,
        }

        # A shared bounds across all sampled poses keeps frame-to-frame
        # comparisons apples-to-apples (no per-frame recentering/rescaling).
        stacked = np.concatenate([_posed(r) for r in poses.values()], axis=0)
        shared_bounds = np.array([stacked.min(axis=0), stacked.max(axis=0)])

        for label, rotations in poses.items():
            posed_vertices = _posed(rotations)
            for angle in (0, 90, 180, 270):
                image = render_orthographic(
                    posed_vertices, faces, colors, rotation_deg_y=angle, bounds=shared_bounds,
                )
                Image.fromarray(image).save(args.preview_dir / f"pose_{label}_{angle:03d}.png")
        print(f"wrote previews to {args.preview_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
