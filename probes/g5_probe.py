#!/usr/bin/env python3
"""Measure G5 root coverage and rest-pose visibility without changing meshforge.

Run from the repository root:

    PYTHONPATH=. python3 out/owl_kente/probes/g5_probe.py
    PYTHONPATH=. python3 out/owl_kente/probes/g5_probe.py analyze

The first form runs the shipped wrap build through ``owl_pipeline.main`` and
then analyses the captured arrays.  The second form only reads ``g5.npz``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import meshforge.gates as gatesmod
import meshforge.wrap as wrapmod


REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "out" / "owl_kente" / "probes"
BUILD_PATH = PROBE_DIR / "g5-build.glb"
NPZ_PATH = PROBE_DIR / "g5.npz"
REPORT_PATH = PROBE_DIR / "G5.md"

# These are intentionally module globals: the shims are the measurement
# boundary, and keeping the originals and captures visible makes the probe
# easy to audit in a debugger or from a follow-on script.
ORIGINAL_GATE_ROOT_COVERED = gatesmod.gate_root_covered
ORIGINAL_BUILD_KENTE_WRAP = wrapmod.build_kente_wrap
CAPTURED: dict[str, Any] = {}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _gate_coverage_mask(root_ring: np.ndarray, axis_xz: np.ndarray,
                        garments: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Recompute the per-vertex mask using gates.gate_root_covered's logic."""
    ring = np.asarray(root_ring, dtype=np.float64)
    axis = np.asarray(axis_xz, dtype=np.float64)
    foot = np.stack([np.full(len(ring), float(axis[0])), ring[:, 1],
                     np.full(len(ring), float(axis[1]))], axis=1)
    dirs = ring - foot
    dist = np.linalg.norm(dirs, axis=1)
    good = dist > 1e-6
    if good.any():
        dirs[good] /= dist[good][:, None]
    covered = np.zeros(len(ring), dtype=bool)
    if garments and good.any():
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
            for vertices, faces in garments
        ])
        locations, ray_index, _ = mesh.ray.intersects_location(
            ray_origins=foot[good], ray_directions=dirs[good], multiple_hits=True
        )
        if len(ray_index):
            t = np.einsum("ij,ij->i", locations - foot[good][ray_index], dirs[good][ray_index])
            beyond = t > dist[good][ray_index] + 0.002
            hit = np.zeros(int(good.sum()), dtype=bool)
            np.logical_or.at(hit, ray_index[beyond], True)
            covered[good] = hit
    return covered


def _capture_gate(root_ring, axis_xz, garments):
    ring = np.asarray(root_ring, dtype=np.float64).copy()
    axis = np.asarray(axis_xz, dtype=np.float64).copy()
    garment_copy = [(np.asarray(vertices).copy(), np.asarray(faces).copy())
                    for vertices, faces in garments]
    result = ORIGINAL_GATE_ROOT_COVERED(root_ring, axis_xz, garments)
    CAPTURED["gate_root_ring"] = ring
    CAPTURED["gate_axis_xz"] = axis
    CAPTURED["gate_garments"] = garment_copy
    CAPTURED["gate"] = result
    # gates.py returns only the aggregate count.  Repeating its short ray
    # calculation here provides the exact per-ring mask needed for analysis.
    CAPTURED["gate_covered"] = _gate_coverage_mask(ring, axis, garment_copy)
    return result


def _capture_wrap(*args, **kwargs):
    result = ORIGINAL_BUILD_KENTE_WRAP(*args, **kwargs)
    CAPTURED["wrap"] = result
    return result


def _capture_export(*args, **kwargs):
    """Capture the body's rest-pose primitive immediately before GLB export."""
    primitives = args[0] if args else kwargs.get("primitives", ())
    for primitive in primitives:
        if getattr(primitive, "name", None) == "body":
            CAPTURED["body_vertices"] = np.asarray(primitive.vertices).copy()
            CAPTURED["body_faces"] = np.asarray(primitive.faces).copy()
            CAPTURED["body_source"] = "pipeline exporter body primitive"
            break
    return CAPTURED["original_export"](*args, **kwargs)


def _fallback_body_from_glb(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Fallback required if an exporter shim cannot see the body primitive."""
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return np.asarray(loaded.vertices), np.asarray(loaded.faces), "loaded GLB mesh"
    candidates = [(name, geometry) for name, geometry in loaded.geometry.items()
                  if "body" in str(name).lower()]
    if not candidates:
        candidates = list(loaded.geometry.items())
    if not candidates:
        raise RuntimeError(f"no mesh geometry found in {path}")
    name, geometry = max(candidates, key=lambda item: len(item[1].faces))
    return np.asarray(geometry.vertices), np.asarray(geometry.faces), f"loaded GLB geometry {name}"


def _collect_arrays(value: Any, prefix: str, out: dict[str, np.ndarray], seen: set[int]) -> None:
    """Collect numeric arrays from the captured wrap/support object graph."""
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            out.setdefault(prefix, np.asarray(value).copy())
        return
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return
    ident = id(value)
    if ident in seen:
        return
    seen.add(ident)
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_arrays(item, f"{prefix}_{key}", out, seen)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _collect_arrays(getattr(value, field.name), f"{prefix}_{field.name}", out, seen)
        return
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        for field in value._fields:
            _collect_arrays(getattr(value, field), f"{prefix}_{field}", out, seen)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_arrays(item, f"{prefix}_{index}", out, seen)


def _captured_arrays() -> dict[str, np.ndarray]:
    if "gate_root_ring" not in CAPTURED or "wrap" not in CAPTURED:
        raise RuntimeError("build did not capture both gate arguments and wrap return")
    if "body_vertices" not in CAPTURED:
        body_v, body_f, source = _fallback_body_from_glb(BUILD_PATH)
        CAPTURED["body_vertices"] = body_v.copy()
        CAPTURED["body_faces"] = body_f.copy()
        CAPTURED["body_source"] = source

    arrays: dict[str, np.ndarray] = {
        "gate_root_ring": CAPTURED["gate_root_ring"],
        "gate_axis_xz": CAPTURED["gate_axis_xz"],
        "gate_covered": np.asarray(CAPTURED["gate_covered"], dtype=bool),
        "body_vertices": np.asarray(CAPTURED["body_vertices"]),
        "body_faces": np.asarray(CAPTURED["body_faces"]),
    }
    for index, (vertices, faces) in enumerate(CAPTURED["gate_garments"]):
        arrays[f"gate_garment_{index}_vertices"] = vertices
        arrays[f"gate_garment_{index}_faces"] = faces

    # The explicit aliases make the analysis keys stable.  The recursive
    # capture below also preserves every numeric array reachable from the wrap
    # return (support, body measure, cloth patterns, primitives, pins, sets).
    wrap = CAPTURED["wrap"]
    arrays.update({
        "sheet_vertices": np.asarray(wrap.sheet.vertices),
        "sheet_faces": np.asarray(wrap.sheet.faces),
        "tail_vertices": np.asarray(wrap.tail.vertices),
        "tail_faces": np.asarray(wrap.tail.faces),
        "knot_vertices": np.asarray(wrap.knot.vertices),
        "knot_faces": np.asarray(wrap.knot.faces),
        "support_root_ring": np.asarray(wrap.support.root_ring),
    })
    _collect_arrays(wrap, "wrap", arrays, set())
    return arrays


def _save_capture_npz() -> None:
    arrays = _captured_arrays()
    gate = CAPTURED["gate"]
    metadata = {
        "gate": {
            "id": gate.id,
            "name": gate.name,
            "value": gate.value,
            "threshold": gate.threshold,
            "passed": gate.passed,
            "detail": gate.detail,
        },
        "body_source": CAPTURED.get("body_source", "unknown"),
        "build_path": str(BUILD_PATH),
        "master_args": [
            "--hires", os.path.join(os.environ["HOME"], "Downloads", "AI-CCORE Owl.glb"),
            "--out", "out/owl_kente/probes/g5-build.glb", "--cache", ".owl_cache",
            "--kente", "--kente-style", "wrap", "--beads", "--cloth-res", "76,42",
        ],
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, default=_json_default))
    np.savez_compressed(NPZ_PATH, **arrays)


def run_build() -> None:
    if not PROBE_DIR.is_dir():
        raise RuntimeError(f"probe directory is missing: {PROBE_DIR}")
    os.chdir(REPO_ROOT)
    import meshforge.owl_pipeline as pipeline

    # The pipeline imports write_rigged_glb into its own module namespace, so
    # patch that bound name as well as the two requested module functions.
    gatesmod.gate_root_covered = _capture_gate
    wrapmod.build_kente_wrap = _capture_wrap
    CAPTURED["original_export"] = pipeline.write_rigged_glb
    pipeline.write_rigged_glb = _capture_export
    try:
        hires = os.path.join(os.environ["HOME"], "Downloads", "AI-CCORE Owl.glb")
        args = [
            "--hires", hires,
            "--out", "out/owl_kente/probes/g5-build.glb",
            "--cache", ".owl_cache",
            "--kente", "--kente-style", "wrap", "--beads", "--cloth-res", "76,42",
        ]
        rc = pipeline.main(args)
        if rc not in (None, 0):
            raise RuntimeError(f"owl_pipeline.main returned {rc}")
    finally:
        gatesmod.gate_root_covered = ORIGINAL_GATE_ROOT_COVERED
        wrapmod.build_kente_wrap = ORIGINAL_BUILD_KENTE_WRAP
        pipeline.write_rigged_glb = CAPTURED["original_export"]
    if not BUILD_PATH.is_file():
        raise RuntimeError(f"build did not write {BUILD_PATH}")
    gate = CAPTURED.get("gate")
    if gate is None:
        raise RuntimeError("gate_root_covered was not called")
    if int(CAPTURED["gate_covered"].sum()) != int(gate.detail["covered"]):
        raise RuntimeError("captured per-vertex coverage disagrees with Gate.detail")
    _save_capture_npz()


def _load_metadata(data: Any) -> dict[str, Any]:
    raw = data["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return json.loads(str(raw))


def _visibility_counts(root_ring: np.ndarray, body: trimesh.Trimesh,
                       elevation_deg: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (unoccluded[n,8], unit_camera_directions[8,3])."""
    yaw_deg = np.arange(0.0, 360.0, 45.0)
    yaw = np.radians(yaw_deg)
    elevation = np.radians(elevation_deg)
    directions = np.stack([
        np.sin(yaw) * np.cos(elevation),
        np.full(len(yaw), np.sin(elevation)),
        np.cos(yaw) * np.cos(elevation),
    ], axis=1)
    n = len(root_ring)
    origins = np.repeat(np.asarray(root_ring, dtype=np.float64), len(directions), axis=0)
    rays = np.tile(directions, (n, 1))
    extent = float(np.linalg.norm(np.asarray(body.bounds[1]) - np.asarray(body.bounds[0])))
    offset = max(extent, 1.0) * 1.0e-6
    # Starting epsilon outside the queried surface suppresses the triangle at
    # the source vertex.  A ray that points into the owl still exits it at a
    # positive t and is therefore correctly classified as occluded.
    ray_origins = origins + rays * offset
    hit = np.zeros(len(ray_origins), dtype=bool)
    locations, ray_index, _ = body.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=rays, multiple_hits=True
    )
    if len(ray_index):
        t = np.einsum("ij,ij->i", locations - ray_origins[ray_index], rays[ray_index])
        positive = t > offset * 0.1
        np.logical_or.at(hit, ray_index[positive], True)
    return (~hit).reshape(n, len(directions)), directions


def _theta_histogram(theta_deg: np.ndarray, covered: np.ndarray) -> list[dict[str, Any]]:
    edges = np.arange(-180.0, 181.0, 10.0)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (theta_deg >= lo) & ((theta_deg < hi) if hi < 180 else (theta_deg <= hi))
        rows.append({"theta": f"{lo:g}..{hi:g}", "covered": int((in_bin & covered).sum()),
                     "bare": int((in_bin & ~covered).sum())})
    return rows


def _height_deciles(y: np.ndarray, covered: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(y, kind="stable")
    groups = np.array_split(order, 10)
    rows = []
    for index, group in enumerate(groups):
        rows.append({
            "decile": index + 1,
            "y_min": float(y[group].min()),
            "y_max": float(y[group].max()),
            "covered": int(covered[group].sum()),
            "bare": int((~covered[group]).sum()),
        })
    return rows


def _circular_extent(theta_deg: np.ndarray) -> tuple[float, float, float]:
    """Return the smallest occupied arc as [start, end] in [0, 360]."""
    values = np.sort(np.mod(np.asarray(theta_deg, dtype=np.float64), 360.0))
    if not len(values):
        return 0.0, 0.0, 0.0
    gaps = np.diff(np.concatenate([values, values[:1] + 360.0]))
    cut = (int(np.argmax(gaps)) + 1) % len(values)
    ordered = np.concatenate([values[cut:], values[:cut] + 360.0])
    return float(ordered[0]), float(ordered[-1]), float(ordered[-1] - ordered[0])


def _signed_relation(dy: np.ndarray, radial_delta: np.ndarray, distance: np.ndarray,
                     body_height: float) -> np.ndarray:
    """Label the nearest sheet/tail vertex without hiding signed measurements."""
    y_tol = 0.005 * body_height
    absent = 0.10 * body_height
    relation = np.full(len(dy), "same-height", dtype="U16")
    relation[distance > absent] = "absent"
    relation[(distance <= absent) & (dy < -y_tol)] = "below"
    relation[(distance <= absent) & (dy > y_tol)] = "above"
    relation[(distance <= absent) & (np.abs(dy) <= y_tol) & (radial_delta < -y_tol)] = "behind"
    return relation


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _markdown_report(stats: dict[str, Any]) -> str:
    lines: list[str] = []
    s = stats
    lines.extend([
        "# G5 root coverage probe",
        "",
        "This report was generated from `g5.npz`; rerunning `g5_probe.py analyze` does not rebuild the owl.",
        "",
        "## Summary",
        "",
        "| measure | result |",
        "|---|---:|",
        f"| ring vertices | {s['ring_count']} |",
        f"| covered / bare | {s['covered_count']} / {s['bare_count']} |",
        f"| G5 value | {s['covered_fraction']:.6f} (gate rounded {s['gate_value']}) |",
        f"| bare theta raw range (deg) | [{_fmt(s['bare_theta_min'], 6)}, {_fmt(s['bare_theta_max'], 6)}] |",
        f"| bare theta minimal wrapped arc | [{_fmt(s['bare_theta_arc_start'], 6)}, {_fmt(s['bare_theta_arc_end'], 6)}] mod 360 ({_fmt(s['bare_theta_arc_span'], 6)} deg) |",
        f"| bare y range | [{_fmt(s['bare_y_min'], 9)}, {_fmt(s['bare_y_max'], 9)}] |",
        f"| bare visible from >=1 view | {s['visible_any_count']}/{s['bare_count']} ({s['visible_any_fraction']:.6f}) |",
        f"| mean unoccluded views / bare vertex | {s['visible_mean']:.6f} / 8 |",
        "",
        "Angles use the requested `degrees(atan2(z-axis_z, x-axis_x))`, in [-180, 180]. Height is world y.",
        "Visibility uses the exported rest-pose body primitive, eight yaw directions (0..315 degrees), elevation +15 degrees, and a 1e-6 model-extent start offset to ignore the source triangle.",
        "",
        "## 10-degree theta histogram",
        "",
        "| theta bin (deg) | covered | bare |",
        "|---:|---:|---:|",
    ])
    for row in s["theta_hist"]:
        lines.append(f"| {row['theta']} | {row['covered']} | {row['bare']} |")
    lines.extend([
        "",
        "## Height rank deciles",
        "",
        "Deciles are ten equal-count rank groups sorted by world y (the displayed ranges are their actual extents).",
        "",
        "| decile | y min | y max | covered | bare |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in s["height_deciles"]:
        lines.append(f"| {row['decile']} | {_fmt(row['y_min'], 9)} | {_fmt(row['y_max'], 9)} | {row['covered']} | {row['bare']} |")
    lines.extend([
        "",
        "## Nearest-cloth summary",
        "",
        "The nearest-cloth query uses sheet + tail vertices only, as requested (the knot is excluded).",
        "",
        "| measure | result |",
        "|---|---:|",
        f"| nearest distance min / median / max | {_fmt(s['nearest_distance_min'], 9)} / {_fmt(s['nearest_distance_median'], 9)} / {_fmt(s['nearest_distance_max'], 9)} |",
        f"| signed dy < 0 / = 0 / > 0 | {s['dy_negative_count']} / {s['dy_zero_count']} / {s['dy_positive_count']} |",
        f"| relation below / behind / same-height / absent | {s['relation_counts'].get('below', 0)} / {s['relation_counts'].get('behind', 0)} / {s['relation_counts'].get('same-height', 0)} / {s['relation_counts'].get('absent', 0)} |",
        "",
        "## Nearest sheet/tail vertex and visibility per bare vertex",
        "",
        "`dy = nearest_cloth_y - ring_y`; negative means below, near-zero with a negative radial delta means behind/toward the owl axis. `absent` means nearest sheet/tail vertex is farther than 0.10 body heights; this label is supplemental, while the signed values are the measurement.",
        "",
        "| ring index | theta deg | y | nearest distance | dy | radial delta | relation | unoccluded views |",
        "|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for row in s["bare_rows"]:
        lines.append(
            f"| {row['index']} | {_fmt(row['theta_deg'], 6)} | {_fmt(row['y'], 9)} | {_fmt(row['distance'], 9)} | "
            f"{_fmt(row['dy'], 9)} | {_fmt(row['radial_delta'], 9)} | {row['relation']} | {row['visible_views']} |"
        )
    lines.extend([
        "",
        "## Visibility summary",
        "",
        "| unoccluded view count | bare vertices |",
        "|---:|---:|",
    ])
    for count in range(9):
        lines.append(f"| {count} | {s['visible_hist'][count]} |")
    lines.extend([
        "",
        "## Diagnosis",
        "",
        s["diagnosis"],
        "",
        "## Options",
        "",
        "1. **Preferred garment fix:** `meshforge/wrap.py::measure_support`, decrease `WrapParams.arm_margin` below its current 0.025. The implementation is `y = low - arm_margin*H`, so this raises the flank/back diagonal toward the wing; with 195/202 nearest cloth vertices below the ring, G5 should increase. Recheck G3/G7 because the change can increase contact and stretch.",
        "2. **Targeted garment fix:** `meshforge/wrap.py::build_kente_wrap`, decrease `WrapParams.tail_offset_deg` below 5.0 and/or increase `WrapParams.tail_span_deg` above 45.0 so the tail pin arc overlaps more of the ring's back edge. G5 should increase in that angular sector; recheck G6 for wing clearance.",
        "3. **Measurement-only option:** `meshforge/gates.py::gate_root_covered(root_ring, axis_xz, garments)`, narrow the caller-supplied `root_ring` to an outward-visible subset. This changes G5 rather than the garment; it is not justified by this test because 202/202 bare vertices are visible from at least one of eight views.",
        "",
        "The per-vertex arrays are in `g5.npz`: `bare_indices`, `bare_theta_deg`, `bare_y`, `bare_nearest_cloth_distance`, `bare_cloth_height_delta`, `bare_cloth_radial_delta`, `bare_relation`, and `bare_visible_views`.",
        "",
    ])
    return "\n".join(lines)


def analyze_npz() -> dict[str, Any]:
    if not NPZ_PATH.is_file():
        raise RuntimeError(f"missing {NPZ_PATH}; run the build form first")
    data = np.load(NPZ_PATH, allow_pickle=False)
    ring = np.asarray(data["gate_root_ring"], dtype=np.float64)
    axis = np.asarray(data["gate_axis_xz"], dtype=np.float64)
    covered = np.asarray(data["gate_covered"], dtype=bool)
    theta_deg = np.degrees(np.arctan2(ring[:, 2] - axis[1], ring[:, 0] - axis[0]))
    y = ring[:, 1]
    bare_mask = ~covered
    bare_indices = np.flatnonzero(bare_mask)

    cloth = np.vstack([np.asarray(data["sheet_vertices"], dtype=np.float64),
                       np.asarray(data["tail_vertices"], dtype=np.float64)])
    nearest_distance, nearest_index = cKDTree(cloth).query(ring[bare_mask], workers=-1)
    nearest_points = cloth[nearest_index]
    delta = nearest_points - ring[bare_mask]
    dy = delta[:, 1]
    ring_radius = np.hypot(ring[bare_mask, 0] - axis[0], ring[bare_mask, 2] - axis[1])
    cloth_radius = np.hypot(nearest_points[:, 0] - axis[0], nearest_points[:, 2] - axis[1])
    radial_delta = cloth_radius - ring_radius

    body = trimesh.Trimesh(vertices=np.asarray(data["body_vertices"], dtype=np.float64),
                           faces=np.asarray(data["body_faces"], dtype=np.int64), process=False)
    visible, camera_directions = _visibility_counts(ring[bare_mask], body)
    visible_views = visible.sum(axis=1).astype(np.int16)
    visible_hist = np.bincount(visible_views, minlength=9)[:9].astype(int).tolist()

    metadata = _load_metadata(data)
    body_height = float(np.ptp(np.asarray(data["body_vertices"])[:, 1]))
    bare_theta = theta_deg[bare_mask]
    bare_y = y[bare_mask]
    relation = _signed_relation(dy, radial_delta, nearest_distance, body_height)
    relation_counts = {name: int((relation == name).sum()) for name in np.unique(relation)}
    theta_arc_start, theta_arc_end, theta_arc_span = _circular_extent(bare_theta)
    any_visible = visible_views > 0
    gate_detail = metadata["gate"]["detail"]
    if int(covered.sum()) != int(gate_detail["covered"]):
        raise RuntimeError("g5.npz coverage mask disagrees with captured Gate.detail")

    stats: dict[str, Any] = {
        "ring_count": int(len(ring)),
        "covered_count": int(covered.sum()),
        "bare_count": int(bare_mask.sum()),
        "covered_fraction": float(covered.mean()) if len(covered) else 0.0,
        "gate_value": metadata["gate"]["value"],
        "bare_theta_min": float(bare_theta.min()),
        "bare_theta_max": float(bare_theta.max()),
        "bare_theta_arc_start": theta_arc_start,
        "bare_theta_arc_end": theta_arc_end,
        "bare_theta_arc_span": theta_arc_span,
        "bare_y_min": float(bare_y.min()),
        "bare_y_max": float(bare_y.max()),
        "visible_any_count": int(any_visible.sum()),
        "visible_any_fraction": float(any_visible.mean()) if len(any_visible) else 0.0,
        "visible_mean": float(visible_views.mean()) if len(visible_views) else 0.0,
        "theta_hist": _theta_histogram(theta_deg, covered),
        "height_deciles": _height_deciles(y, covered),
        "visible_hist": visible_hist,
        "relation_counts": relation_counts,
        "nearest_distance_min": float(nearest_distance.min()),
        "nearest_distance_median": float(np.median(nearest_distance)),
        "nearest_distance_max": float(nearest_distance.max()),
        "dy_negative_count": int((dy < 0.0).sum()),
        "dy_zero_count": int((dy == 0.0).sum()),
        "dy_positive_count": int((dy > 0.0).sum()),
        "bare_rows": [],
        "diagnosis": "",
    }
    for i in range(len(bare_indices)):
        stats["bare_rows"].append({
            "index": int(bare_indices[i]),
            "theta_deg": float(bare_theta[i]),
            "y": float(bare_y[i]),
            "distance": float(nearest_distance[i]),
            "dy": float(dy[i]),
            "radial_delta": float(radial_delta[i]),
            "relation": str(relation[i]),
            "visible_views": int(visible_views[i]),
        })

    # Keep the decisive per-vertex measurements in the same npz so all of the
    # report can be regenerated without rebuilding or relying on markdown.
    original = {key: data[key] for key in data.files}
    original.update({
        "ring_theta_deg": theta_deg,
        "ring_y": y,
        "bare_indices": bare_indices.astype(np.int64),
        "bare_theta_deg": bare_theta,
        "bare_y": bare_y,
        "bare_nearest_cloth_index": nearest_index.astype(np.int64),
        "bare_nearest_cloth_distance": nearest_distance,
        "bare_cloth_delta_xyz": delta,
        "bare_cloth_height_delta": dy,
        "bare_cloth_radial_delta": radial_delta,
        "bare_relation": relation,
        "bare_visible_views": visible_views,
        "bare_visible_by_view": visible,
        "camera_yaw_deg": np.arange(0.0, 360.0, 45.0),
        "camera_directions": camera_directions,
    })
    np.savez_compressed(NPZ_PATH, **original)

    # A conservative diagnosis from the measured visibility and proximity:
    # the measurement-only option is only credible for vertices hidden in all
    # views; any visible bare vertices are a garment gap by this test.
    if stats["visible_any_count"] == 0:
        diagnosis = ("The bare set is the raised wing root's self-occluded underside/armpit in this eight-view "
                     "rest-pose test: no bare vertex is visible from any sampled turntable direction, so the "
                     "failure is not an externally visible gap.")
    elif stats["visible_any_count"] < stats["bare_count"]:
        diagnosis = (f"The bare set is mixed: {stats['visible_any_count']} of {stats['bare_count']} bare vertices "
                     "are visible from at least one sampled direction, while the remainder are the wing root's "
                     "self-occluded underside/armpit. The visible subset is a genuine external garment gap; the "
                     "hidden subset is not decisive for viewer-facing coverage.")
    else:
        diagnosis = ("The bare set is a genuinely visible garment gap: every bare root vertex is visible from at "
                     "least one sampled turntable direction, so narrowing the measurement ring would mask an "
                     "external defect rather than describe the owl's visible surface.")
    stats["diagnosis"] = diagnosis
    REPORT_PATH.write_text(_markdown_report(stats), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("build", "analyze", "all"), default="all")
    args = parser.parse_args()
    if args.mode in ("build", "all"):
        run_build()
    if args.mode in ("analyze", "all"):
        stats = analyze_npz()
        print(json.dumps({
            "ring": stats["ring_count"],
            "covered": stats["covered_count"],
            "bare": stats["bare_count"],
            "bare_theta_deg": [stats["bare_theta_min"], stats["bare_theta_max"]],
            "bare_theta_arc_mod_360": [stats["bare_theta_arc_start"], stats["bare_theta_arc_end"], stats["bare_theta_arc_span"]],
            "bare_y": [stats["bare_y_min"], stats["bare_y_max"]],
            "visible_any": [stats["visible_any_count"], stats["bare_count"], stats["visible_any_fraction"]],
            "visible_mean_views": stats["visible_mean"],
            "relations": stats["relation_counts"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
