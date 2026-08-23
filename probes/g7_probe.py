#!/usr/bin/env python3
"""Capture and analyse the shipped G7 stretch measurement.

Run from the repository root:

    PYTHONPATH=. python3 out/owl_kente/probes/g7_probe.py
    PYTHONPATH=. python3 out/owl_kente/probes/g7_probe.py analyze

The ``analyze`` mode only reads ``g7.npz`` and regenerates ``G7.md``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import meshforge.gates as gatesmod
import meshforge.wrap as wrapmod


REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "out" / "owl_kente" / "probes"
BUILD_PATH = PROBE_DIR / "g7-build.glb"
NPZ_PATH = PROBE_DIR / "g7.npz"
REPORT_PATH = PROBE_DIR / "G7.md"

ORIGINAL_GATE_STRETCH = gatesmod.gate_stretch
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


def _capture_gate_stretch(P, sets, stretch_k, cols=None):
    """Capture the gate's inputs before delegating to the real gate."""
    positions = np.asarray(P, dtype=np.float64).copy()
    records = []
    for set_index, s in enumerate(sets):
        i = np.asarray(s.i, dtype=np.int64).copy()
        j = np.asarray(s.j, dtype=np.int64).copy()
        rest = np.asarray(s.rest, dtype=np.float64).copy()
        current = np.linalg.norm(positions[j] - positions[i], axis=1)
        records.append({
            "set_index": int(set_index),
            "i": i,
            "j": j,
            "rest": rest,
            "current": current,
            "k": float(s.k),
        })
    result = ORIGINAL_GATE_STRETCH(P, sets, stretch_k, cols)
    CAPTURED["gate_positions"] = positions
    CAPTURED["gate_sets"] = records
    CAPTURED["stretch_k"] = float(stretch_k)
    CAPTURED["gate_cols"] = None if cols is None else int(cols)
    CAPTURED["gate"] = result
    return result


def _capture_wrap(*args, **kwargs):
    result = ORIGINAL_BUILD_KENTE_WRAP(*args, **kwargs)
    CAPTURED["wrap"] = result
    return result


def _save_capture_npz() -> None:
    if "wrap" not in CAPTURED or "gate" not in CAPTURED:
        raise RuntimeError("build did not capture both the wrap and G7 gate")
    wrap = CAPTURED["wrap"]
    records = CAPTURED["gate_sets"]
    starts = [0]
    for record in records:
        starts.append(starts[-1] + len(record["i"]))
    if records:
        constraint_i = np.concatenate([record["i"] for record in records])
        constraint_j = np.concatenate([record["j"] for record in records])
        constraint_rest = np.concatenate([record["rest"] for record in records])
        constraint_current = np.concatenate([record["current"] for record in records])
        constraint_k = np.concatenate([
            np.full(len(record["i"]), record["k"], dtype=np.float64) for record in records
        ])
        constraint_set_index = np.concatenate([
            np.full(len(record["i"]), record["set_index"], dtype=np.int64) for record in records
        ])
    else:
        constraint_i = constraint_j = np.zeros(0, dtype=np.int64)
        constraint_rest = constraint_current = constraint_k = np.zeros(0, dtype=np.float64)
        constraint_set_index = np.zeros(0, dtype=np.int64)

    rows, cols = (int(v) for v in wrap.sheet_grid)
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
        "master_args": [
            "--hires", os.path.join(os.environ["HOME"], "Downloads", "AI-CCORE Owl.glb"),
            "--out", "out/owl_kente/probes/g7-build.glb", "--cache", ".owl_cache",
            "--kente", "--kente-style", "wrap", "--beads", "--cloth-res", "76,42",
        ],
        "support_info": wrap.support.info,
        "grid": [rows, cols],
    }
    arrays = {
        "positions": np.asarray(CAPTURED["gate_positions"], dtype=np.float64),
        "constraint_i": constraint_i,
        "constraint_j": constraint_j,
        "constraint_rest": constraint_rest,
        "constraint_current": constraint_current,
        "constraint_k": constraint_k,
        "constraint_set_index": constraint_set_index,
        "constraint_set_starts": np.asarray(starts, dtype=np.int64),
        "grid_index": np.arange(rows * cols, dtype=np.int64).reshape(rows, cols),
        "theta_cols": np.asarray(wrap.sheet_pattern.theta_cols, dtype=np.float64),
        "support_theta": np.asarray(wrap.support.theta, dtype=np.float64),
        "support_r": np.asarray(wrap.support.r, dtype=np.float64),
        "support_gather": np.asarray(wrap.support.gather, dtype=np.float64),
        "support_y": np.asarray(wrap.support.y, dtype=np.float64),
        "body_H": np.asarray(float(wrap.body.size_y), dtype=np.float64),
        "stretch_k": np.asarray(float(CAPTURED["stretch_k"]), dtype=np.float64),
        "gate_cols": np.asarray(-1 if CAPTURED["gate_cols"] is None else CAPTURED["gate_cols"], dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, default=_json_default)),
    }
    np.savez_compressed(NPZ_PATH, **arrays)


def run_build() -> None:
    os.chdir(REPO_ROOT)
    import meshforge.owl_pipeline as pipeline

    gatesmod.gate_stretch = _capture_gate_stretch
    wrapmod.build_kente_wrap = _capture_wrap
    try:
        hires = os.path.join(os.environ["HOME"], "Downloads", "AI-CCORE Owl.glb")
        args = [
            "--hires", hires,
            "--out", "out/owl_kente/probes/g7-build.glb",
            "--cache", ".owl_cache",
            "--kente", "--kente-style", "wrap", "--beads", "--cloth-res", "76,42",
        ]
        rc = pipeline.main(args)
        if rc not in (None, 0):
            raise RuntimeError(f"owl_pipeline.main returned {rc}")
    finally:
        gatesmod.gate_stretch = ORIGINAL_GATE_STRETCH
        wrapmod.build_kente_wrap = ORIGINAL_BUILD_KENTE_WRAP
    if not BUILD_PATH.is_file():
        raise RuntimeError(f"build did not write {BUILD_PATH}")
    _save_capture_npz()


def _load_npz() -> dict[str, np.ndarray]:
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_metadata(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    value = raw["metadata_json"]
    return json.loads(str(value.item() if isinstance(value, np.ndarray) else value))


def _percentile(values: list[float] | np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if len(values) else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _theta_deg(theta: np.ndarray) -> np.ndarray:
    return (np.degrees(theta) + 180.0) % 360.0 - 180.0


def _part_labels(theta_deg: np.ndarray, support_info: dict[str, Any]) -> np.ndarray:
    """Classify by the loop's forward-from-knot construction in measure_support."""
    k = float(support_info["theta_k_deg"])
    a = float(support_info["theta_a_deg"])
    b = float(support_info["theta_b_deg"])
    d = (np.asarray(theta_deg) - k) % 360.0
    da = (a - k) % 360.0
    db = (b - k) % 360.0
    labels = np.full(len(d), "back diagonal", dtype="U20")
    labels[(d <= 25.0) | (d >= 335.0)] = "knot arc"
    labels[(d > 25.0) & (d < da)] = "front diagonal"
    labels[(d >= da) & (d <= db)] = "wing-base flank"
    return labels


def _analyse(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    metadata = _load_metadata(raw)
    positions = raw["positions"]
    grid = raw["grid_index"]
    rows, cols = (int(v) for v in grid.shape)
    stretch_k = float(raw["stretch_k"])
    starts = raw["constraint_set_starts"]
    set_index = raw["constraint_set_index"]
    ci = raw["constraint_i"]
    cj = raw["constraint_j"]
    rest = raw["constraint_rest"]
    current = raw["constraint_current"]
    ratio = current / np.maximum(rest, 1e-12)
    ri, coli = np.divmod(ci, cols)
    rj, colj = np.divmod(cj, cols)

    warp_by_col: list[list[float]] = [[] for _ in range(cols)]
    weft_by_col: list[list[float]] = [[] for _ in range(cols)]
    warp_rest_by_col: list[list[float]] = [[] for _ in range(cols)]
    weft_rest_by_col: list[list[float]] = [[] for _ in range(cols)]
    all_by_col: list[list[float]] = [[] for _ in range(cols)]
    top_weft: dict[tuple[int, int], tuple[float, float]] = {}
    set_rows = []
    kind_code = np.zeros(len(ci), dtype=np.int8)
    for si in range(len(starts) - 1):
        lo, hi = int(starts[si]), int(starts[si + 1])
        mask = (set_index == si)
        # The grid index is the definition: same row, changing column is
        # around-body weft; same column, changing row is top-to-hem warp.
        same_row = bool(np.all(ri[mask] == rj[mask])) if mask.any() else False
        same_col = bool(np.all(coli[mask] == colj[mask])) if mask.any() else False
        if same_row and not same_col:
            kind = "weft"
            code = 2
        elif same_col and not same_row:
            kind = "warp"
            code = 1
        else:
            kind = "other"
            code = 0
        kind_code[lo:hi] = code
        selected = np.isclose(np.asarray(raw["constraint_k"])[lo:hi], stretch_k, atol=1e-9)
        set_rows.append({
            "set": si,
            "k": float(np.asarray(raw["constraint_k"])[lo]) if hi > lo else float("nan"),
            "edges": hi - lo,
            "grid_delta": [
                int(np.median(rj[mask] - ri[mask])) if mask.any() else 0,
                int(np.median(colj[mask] - coli[mask])) if mask.any() else 0,
            ],
            "kind": kind,
            "g7": bool(selected.any()),
        })
        if not selected.any():
            continue
        edge_idx = np.arange(lo, hi)[selected]
        for edge in edge_idx:
            c = int(ci[edge] % cols)
            all_by_col[c].append(float(ratio[edge]))
            if kind == "warp":
                warp_by_col[c].append(float(ratio[edge]))
                warp_rest_by_col[c].append(float(rest[edge]))
            elif kind == "weft":
                weft_by_col[c].append(float(ratio[edge]))
                weft_rest_by_col[c].append(float(rest[edge]))
                if int(ri[edge]) == 0:
                    top_weft[(0, c)] = (float(rest[edge]), float(current[edge]))

    theta_cols = raw["theta_cols"]
    theta_deg = _theta_deg(theta_cols)
    support_theta = raw["support_theta"]
    support_r = np.interp(theta_cols, support_theta, raw["support_r"], period=2 * np.pi)
    support_gather = np.interp(theta_cols, support_theta, raw["support_gather"], period=2 * np.pi)

    def column_stats(c: int) -> dict[str, Any]:
        w = np.asarray(warp_by_col[c], dtype=np.float64)
        f = np.asarray(weft_by_col[c], dtype=np.float64)
        both = np.asarray(all_by_col[c], dtype=np.float64)
        real_c = c % max(cols - 1, 1)
        left = top_weft.get((0, (real_c - 1) % max(cols - 1, 1)), (float("nan"), float("nan")))
        right = top_weft.get((0, real_c), (float("nan"), float("nan")))
        return {
            "column": c,
            "theta_deg": float(theta_deg[c]),
            "support_r": float(support_r[c]),
            "gather": float(support_gather[c]),
            "top": positions[grid[0, c]].tolist(),
            "warp_mean": float(w.mean()) if len(w) else float("nan"),
            "warp_p90": _percentile(w, 90),
            "weft_mean": float(f.mean()) if len(f) else float("nan"),
            "weft_p90": _percentile(f, 90),
            "overall_p90": _percentile(both, 90),
            "left_rest": left[0],
            "left_current": left[1],
            "right_rest": right[0],
            "right_current": right[1],
            "drafted_warp_mean": float(np.mean(warp_rest_by_col[c])) if warp_rest_by_col[c] else float("nan"),
            "drafted_weft_mean": float(np.mean(weft_rest_by_col[c])) if weft_rest_by_col[c] else float("nan"),
        }

    columns = [column_stats(c) for c in range(cols)]
    worst = sorted(columns, key=lambda row: (-row["overall_p90"], row["column"]))[:10]
    failing = [columns[c] for c in range(12, 20)]
    H = float(raw["body_H"])
    median_warp_p90 = float(np.nanmedian([row["warp_p90"] for row in failing]))
    median_weft_p90 = float(np.nanmedian([row["weft_p90"] for row in failing]))
    dominant_kind = "warp" if median_warp_p90 >= median_weft_p90 else "weft"
    other_kind = "weft" if dominant_kind == "warp" else "warp"
    for row in failing:
        scale = max(1.0, row[f"{dominant_kind}_p90"] / 1.12)
        row["extra_scale"] = scale
        row["extra_pct"] = (scale - 1.0) * 100.0
        row["extra_H"] = row[f"drafted_{dominant_kind}_mean"] * (scale - 1.0) / H
        row["dominant_p90"] = row[f"{dominant_kind}_p90"]
        row["part"] = str(_part_labels(np.asarray([row["theta_deg"]]), metadata["support_info"])[0])

    selected_ratio = ratio[np.isclose(np.asarray(raw["constraint_k"]), stretch_k, atol=1e-9)]
    fail_theta = theta_deg[12:20]
    parts = _part_labels(fail_theta, metadata["support_info"])
    diagnosis = {
        "gate_median": float(np.median(selected_ratio)),
        "gate_p90": float(np.percentile(selected_ratio, 90)),
        "gate_p99": float(np.percentile(selected_ratio, 99)),
        "theta_range": [float(fail_theta.min()), float(fail_theta.max())],
        "parts": sorted(set(str(x) for x in parts)),
        "part_counts": {str(x): int((parts == x).sum()) for x in np.unique(parts)},
        "dominant_kind": dominant_kind,
        "other_kind": other_kind,
        "median_dominant_p90": median_warp_p90 if dominant_kind == "warp" else median_weft_p90,
        "median_other_p90": median_weft_p90 if dominant_kind == "warp" else median_warp_p90,
        "warp_p90_range": [float(min(row["warp_p90"] for row in failing)),
                           float(max(row["warp_p90"] for row in failing))],
        "weft_p90_range": [float(min(row["weft_p90"] for row in failing)),
                           float(max(row["weft_p90"] for row in failing))],
        "extra_pct_range": [float(min(row["extra_pct"] for row in failing)),
                            float(max(row["extra_pct"] for row in failing))],
        "extra_H_range": [float(min(row["extra_H"] for row in failing)),
                          float(max(row["extra_H"] for row in failing))],
        "extra_pct_mean": float(np.mean([row["extra_pct"] for row in failing])),
        "extra_H_mean": float(np.mean([row["extra_H"] for row in failing])),
    }
    return {
        "metadata": metadata,
        "rows": rows,
        "cols": cols,
        "H": H,
        "set_rows": set_rows,
        "kind_code": kind_code,
        "columns": columns,
        "worst": worst,
        "failing": failing,
        "diagnosis": diagnosis,
    }


def _markdown_report(stats: dict[str, Any]) -> str:
    d = stats["diagnosis"]
    info = stats["metadata"]["support_info"]
    worst_rows = stats["metadata"]["gate"]["detail"].get("worst_rows_p90", {})
    lines = [
        "# G7 stretch probe",
        "",
        "Generated from `g7.npz`; `PYTHONPATH=. python3 out/owl_kente/probes/g7_probe.py analyze` re-runs only the analysis.",
        "",
        "## Measurement",
        "",
        f"Grid: {stats['rows']} rows × {stats['cols']} columns; body height H = {_fmt(stats['H'], 6)}.",
        f"Gate ratios: median {_fmt(d['gate_median'], 6)}, p90 {_fmt(d['gate_p90'], 6)}, p99 {_fmt(d['gate_p99'], 6)}.",
        "",
        "## Constraint split",
        "",
        "Classification uses the captured grid index: an edge with the same row and changing column is weft (around the body); an edge with the same column and changing row is warp (top to hem). Other grid deltas are shear/bend and are not included by G7 because their k differs from stretch_k.",
        "",
        "| set | k | edges | grid delta (row,col) | classification | G7 |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in stats["set_rows"]:
        lines.append(f"| {row['set']} | {_fmt(row['k'], 3)} | {row['edges']} | {row['grid_delta']} | {row['kind']} | {'yes' if row['g7'] else 'no'} |")
    lines.extend([
        "",
        "## Ten worst columns",
        "",
        "Rows are ranked by the same combined warp+weft p90 convention used by G7. Neighbor spacing is `rest → draped` for the top-row weft edge to the left and right; lengths are world units.",
        "",
        "| col | θ° | r | gather | warp mean/p90 | weft mean/p90 | all p90 | pinned top (x,y,z) | left rest→draped | right rest→draped |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for row in stats["worst"]:
        top = ", ".join(_fmt(float(v), 4) for v in row["top"])
        lines.append(
            f"| {row['column']} | {_fmt(row['theta_deg'], 2)} | {_fmt(row['support_r'], 4)} | {_fmt(row['gather'], 3)} | "
            f"{_fmt(row['warp_mean'], 3)}/{_fmt(row['warp_p90'], 3)} | {_fmt(row['weft_mean'], 3)}/{_fmt(row['weft_p90'], 3)} | "
            f"{_fmt(row['overall_p90'], 3)} | ({top}) | {_fmt(row['left_rest'], 4)}→{_fmt(row['left_current'], 4)} | "
            f"{_fmt(row['right_rest'], 4)}→{_fmt(row['right_current'], 4)} |"
        )
    lines.extend([
        "",
        f"## Columns 12–19 {d['dominant_kind']} deficit",
        "",
        f"The required ease below scales each column's drafted {d['dominant_kind']} rest lengths by `{d['dominant_kind']}_p90 / 1.12`; `extra H` is that extra length divided by H, using the column's mean drafted {d['dominant_kind']} rest length across its rows.",
        "",
        f"| col | θ° | support part | {d['dominant_kind']} p90 | extra rest % | extra H |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    for row in stats["failing"]:
        lines.append(f"| {row['column']} | {_fmt(row['theta_deg'], 2)} | {row['part']} | {_fmt(row['dominant_p90'], 4)} | {_fmt(row['extra_pct'], 2)}% | {_fmt(row['extra_H'], 5)} |")
    lines.extend([
        "",
        "## Diagnosis",
        "",
        f"The failure is {d['dominant_kind']}, not {d['other_kind']}: columns 12–19 have median {d['dominant_kind']} p90 {_fmt(d['median_dominant_p90'], 3)} versus median {d['other_kind']} p90 {_fmt(d['median_other_p90'], 3)}. The full warp p90 range is {_fmt(d['warp_p90_range'][0], 3)}–{_fmt(d['warp_p90_range'][1], 3)} and weft range is {_fmt(d['weft_p90_range'][0], 3)}–{_fmt(d['weft_p90_range'][1], 3)}; the isolated weft outlier is retained in the table but does not define the run. Their θ range is {_fmt(d['theta_range'][0], 2)}° to {_fmt(d['theta_range'][1], 2)}°. By `measure_support`'s forward-from-knot intervals (knot {info['theta_k_deg']}°, a {info['theta_a_deg']}°, b {info['theta_b_deg']}°), they occupy {', '.join(d['parts'])}; the ring is {info['root_ring_theta_deg'][0]}° to {info['root_ring_theta_deg'][1]}°, so all are on the back-diagonal/root-ring side, with the knot-arc overlap shown explicitly and none on the front diagonal or wing-base flank. The gate's worst rows are {', '.join(str(x) for x in worst_rows)}; these are mid-column warp edges, consistent with the drape reaching farther down/around the root than the drafted fall allows. The drafted {d['dominant_kind']} edges are shorter than the distances the drape reaches, hence l/l0 ≈ 1.4: the rows are pulled down/away from their drafted fall while the pinned top row is not primarily being pulled apart around the body. Bringing these columns to p90 ≤ 1.12 requires {_fmt(d['extra_pct_range'][0], 2)}–{_fmt(d['extra_pct_range'][1], 2)}% extra {d['dominant_kind']} rest length, {_fmt(d['extra_H_range'][0], 5)}–{_fmt(d['extra_H_range'][1], 5)} H (mean {_fmt(d['extra_pct_mean'], 2)}%, {_fmt(d['extra_H_mean'], 5)} H).",
        "",
        "## Options",
        "",
        f"1. **First choice — local ease:** add a `local_{d['dominant_kind']}_ease` parameter to `meshforge/drape.py::constraint_sets` and apply it to the root-ring columns' {d['dominant_kind']} `rest` lengths in the `meshforge/wrap.py::build_kente_wrap` call. Start with the measured per-column scale (about the mean shown above), then recheck G3/G4/G8. This directly repairs the measured drafted-vs-required deficit without moving the support.",
        "2. **Spacing correction:** change the `seg` spacing input/logic in `meshforge/drape.py::column_angles` so the columns are allocated from the actual pinned top-loop arc, including the root-ring stand-off. Expected effect: a better-matched drafted grid and a lower G7 p90, at the cost of moving the column pattern around the loop; this is secondary because G7's dominant error here is warp rather than weft.",
        "3. **Support smoothing:** increase `WrapParams.loop_smooth_cols` in `meshforge/wrap.py::measure_support` above 1.0 and remeasure the root transition. Expected effect: a less abrupt support-radius/angle transition and less abrupt fall variation at the root ring; it is a lower-confidence, global change that can affect pin clearance and G3/G5.",
        "",
    ])
    return "\n".join(lines)


def analyze_npz() -> dict[str, Any]:
    raw = _load_npz()
    stats = _analyse(raw)
    derived = {
        "constraint_ratio": raw["constraint_current"] / np.maximum(raw["constraint_rest"], 1e-12),
        "constraint_kind_code": stats["kind_code"],
        "column_theta_deg": np.asarray([row["theta_deg"] for row in stats["columns"]]),
        "column_support_r": np.asarray([row["support_r"] for row in stats["columns"]]),
        "column_gather": np.asarray([row["gather"] for row in stats["columns"]]),
        "column_warp_mean": np.asarray([row["warp_mean"] for row in stats["columns"]]),
        "column_warp_p90": np.asarray([row["warp_p90"] for row in stats["columns"]]),
        "column_weft_mean": np.asarray([row["weft_mean"] for row in stats["columns"]]),
        "column_weft_p90": np.asarray([row["weft_p90"] for row in stats["columns"]]),
        "column_overall_p90": np.asarray([row["overall_p90"] for row in stats["columns"]]),
        "failing_columns": np.asarray([row["column"] for row in stats["failing"]], dtype=np.int64),
        "failing_extra_pct": np.asarray([row["extra_pct"] for row in stats["failing"]]),
        "failing_extra_H": np.asarray([row["extra_H"] for row in stats["failing"]]),
    }
    raw.update(derived)
    np.savez_compressed(NPZ_PATH, **raw)
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
        d = stats["diagnosis"]
        print(json.dumps({
            "grid": [stats["rows"], stats["cols"]],
            "gate": [d["gate_median"], d["gate_p90"], d["gate_p99"]],
            "failure_kind": d["dominant_kind"],
            "failing_theta_deg": d["theta_range"],
            "parts": d["part_counts"],
            "warp_p90": d["warp_p90_range"],
            "weft_p90": d["weft_p90_range"],
            "extra_pct": d["extra_pct_range"],
            "extra_H": d["extra_H_range"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
