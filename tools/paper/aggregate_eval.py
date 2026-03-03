#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "outside_route_lanes",
    "red_light",
    "route_dev",
    "route_timeout",
    "stop_infraction",
    "vehicle_blocked",
]

# Bike blueprints that are classified as "bike" by the CBF filter.
# "vehicle.diamondback.century" is the only blueprint matched by the
# CBF/RL code (checks for 'diamondback' in type_id).
BIKE_SUBSTRINGS = ["diamondback"]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _std(values):
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / float(len(values) - 1)
    return math.sqrt(var)


def _ci95(values):
    n = len(values)
    if n <= 1:
        return 0.0
    return 1.96 * _std(values) / math.sqrt(float(n))


def _extract_route_num(route_id):
    if route_id is None:
        return -1
    m = re.search(r"(\d+)$", str(route_id))
    return int(m.group(1)) if m else -1


def _extract_stage_steps(method_name):
    # stage_025k -> 25000, stage_100k -> 100000
    m_k = re.match(r".*?(\d+)k$", method_name)
    if m_k:
        return int(m_k.group(1)) * 1000
    m_num = re.match(r".*?(\d+)$", method_name)
    if m_num:
        return int(m_num.group(1))
    return ""


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_from_manifest(manifest_path):
    manifest = _load_json(manifest_path)
    if "methods" not in manifest:
        raise ValueError("Manifest missing 'methods': {}".format(manifest_path))

    runtime = manifest.get("runtime", {})
    num_routes = _safe_int(runtime.get("num_routes", 0), 0)

    runs = []
    for method in manifest["methods"]:
        method_name = method.get("name")
        if not method_name:
            raise ValueError("Manifest method missing name")
        for pass_run in method.get("pass_runs", []):
            results_json = pass_run.get("results_json")
            pass_index = _safe_int(pass_run.get("pass_index", 0), 0)
            if not results_json:
                continue
            runs.append(
                {
                    "method": method_name,
                    "pass_index": pass_index,
                    "results_json": results_json,
                    "num_routes": num_routes,
                }
            )

    reference_method = manifest.get("reference_method")
    return runs, reference_method


def _discover_from_root(runs_root):
    runs = []
    if not os.path.isdir(runs_root):
        raise ValueError("runs_root does not exist: {}".format(runs_root))

    for method_name in sorted(os.listdir(runs_root)):
        method_root = os.path.join(runs_root, method_name)
        if not os.path.isdir(method_root):
            continue
        for pass_name in sorted(os.listdir(method_root)):
            pass_root = os.path.join(method_root, pass_name)
            if not os.path.isdir(pass_root):
                continue
            results_json = os.path.join(pass_root, "eval", "ego_vehicle_0", "results.json")
            if not os.path.exists(results_json):
                continue
            m = re.match(r"pass(\d+)", pass_name)
            pass_index = int(m.group(1)) if m else 0
            runs.append(
                {
                    "method": method_name,
                    "pass_index": pass_index,
                    "results_json": results_json,
                    "num_routes": 0,
                }
            )
    return runs


def _load_route_rows(run):
    results_json = run["results_json"]
    if not os.path.exists(results_json):
        raise FileNotFoundError("Missing results file: {}".format(results_json))

    data = _load_json(results_json)
    records = data.get("_checkpoint", {}).get("records", [])
    if not isinstance(records, list):
        raise ValueError("Invalid records format in {}".format(results_json))

    method = run["method"]
    pass_index = _safe_int(run.get("pass_index", 0), 0)
    num_routes = _safe_int(run.get("num_routes", 0), 0)
    stage_steps = _extract_stage_steps(method)

    rows = []
    for rec in records:
        idx = _safe_int(rec.get("index", -1), -1)
        route_slot = idx % num_routes if num_routes > 0 and idx >= 0 else idx
        route_id = rec.get("route_id", "")
        route_num = _extract_route_num(route_id)
        route_key = "pass{0:02d}_slot{1:03d}".format(pass_index, max(route_slot, 0))

        infractions = rec.get("infractions", {}) or {}
        inf_counts = {key: len(infractions.get(key, []) or []) for key in INFRACTION_KEYS}

        # Split vehicle collisions into bike (diamondback) and non-bike vehicles
        bike_count = 0
        for s in (infractions.get("collisions_vehicle", []) or []):
            s_lower = s.lower()
            if any(tag in s_lower for tag in BIKE_SUBSTRINGS):
                bike_count += 1
        vehicle_only_count = inf_counts["collisions_vehicle"] - bike_count

        collisions_total = (
            inf_counts["collisions_layout"]
            + inf_counts["collisions_pedestrian"]
            + inf_counts["collisions_vehicle"]
        )

        meta = rec.get("meta", {}) or {}
        score_route = _safe_float(rec.get("scores", {}).get("score_route", 0.0), 0.0)
        route_length_m = _safe_float(meta.get("route_length", 0.0), 0.0)
        traveled_km = max((score_route / 100.0) * route_length_m / 1000.0, 0.0)

        status = str(rec.get("status", ""))
        completed = 1 if status.startswith("Completed") else 0

        row = {
            "method": method,
            "stage_steps": stage_steps,
            "pass_index": pass_index,
            "record_index": idx,
            "route_id_raw": route_id,
            "route_id_num": route_num,
            "route_slot": route_slot,
            "route_key": route_key,
            "status": status,
            "completed": completed,
            "score_composed": _safe_float(rec.get("scores", {}).get("score_composed", 0.0), 0.0),
            "score_route": score_route,
            "score_penalty": _safe_float(rec.get("scores", {}).get("score_penalty", 0.0), 0.0),
            "duration_system": _safe_float(meta.get("duration_system", 0.0), 0.0),
            "duration_game": _safe_float(meta.get("duration_game", 0.0), 0.0),
            "route_length_m": route_length_m,
            "traveled_km": traveled_km,
            "collisions_layout": inf_counts["collisions_layout"],
            "collisions_pedestrian": inf_counts["collisions_pedestrian"],
            "collisions_vehicle": inf_counts["collisions_vehicle"],
            "collisions_bike": bike_count,
            "collisions_vehicle_only": vehicle_only_count,
            "collisions_total": collisions_total,
            "outside_route_lanes": inf_counts["outside_route_lanes"],
            "red_light": inf_counts["red_light"],
            "route_dev": inf_counts["route_dev"],
            "route_timeout": inf_counts["route_timeout"],
            "stop_infraction": inf_counts["stop_infraction"],
            "vehicle_blocked": inf_counts["vehicle_blocked"],
            "results_json": results_json,
        }
        rows.append(row)

    return rows


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CARLA leaderboard results for paper plots")
    parser.add_argument("--manifest", type=str, default="", help="run_manifest.json generated by run_paper_eval.sh")
    parser.add_argument("--runs-root", type=str, default="", help="Fallback discovery root: results/paper_eval/<exp>")
    parser.add_argument("--output-dir", type=str, default="paper_outputs/main", help="Output folder for CSV tables")
    parser.add_argument("--reference-method", type=str, default="", help="Reference method for paired deltas")
    args = parser.parse_args()

    if not args.manifest and not args.runs_root:
        raise SystemExit("Provide either --manifest or --runs-root")

    if args.manifest:
        runs, manifest_ref = _discover_from_manifest(args.manifest)
        ref_method = args.reference_method or manifest_ref
    else:
        runs = _discover_from_root(args.runs_root)
        ref_method = args.reference_method

    if not runs:
        raise SystemExit("No run results discovered")

    all_rows = []
    for run in runs:
        all_rows.extend(_load_route_rows(run))

    if not all_rows:
        raise SystemExit("No route rows found in discovered results")

    all_rows = sorted(all_rows, key=lambda r: (r["method"], r["pass_index"], r["record_index"]))

    route_csv = os.path.join(args.output_dir, "route_metrics.csv")
    route_fields = list(all_rows[0].keys())
    _write_csv(route_csv, route_fields, all_rows)

    by_method = defaultdict(list)
    for row in all_rows:
        by_method[row["method"]].append(row)

    summary_rows = []
    for method, rows in sorted(by_method.items()):
        score_comp = [r["score_composed"] for r in rows]
        score_route = [r["score_route"] for r in rows]
        duration_sys = [r["duration_system"] for r in rows]
        duration_game = [r["duration_game"] for r in rows]
        completed = [r["completed"] for r in rows]

        collisions_total = [r["collisions_total"] for r in rows]
        collisions_vehicle = [r["collisions_vehicle"] for r in rows]
        collisions_bike = [r["collisions_bike"] for r in rows]
        collisions_vehicle_only = [r["collisions_vehicle_only"] for r in rows]
        collisions_pedestrian = [r["collisions_pedestrian"] for r in rows]
        collisions_layout = [r["collisions_layout"] for r in rows]
        red_light = [r["red_light"] for r in rows]
        stop_infraction = [r["stop_infraction"] for r in rows]
        route_dev = [r["route_dev"] for r in rows]
        route_timeout = [r["route_timeout"] for r in rows]

        total_traveled_km = sum(r["traveled_km"] for r in rows)
        collisions_per_km = sum(collisions_total) / max(total_traveled_km, 1e-6)

        summary_rows.append(
            {
                "method": method,
                "stage_steps": _extract_stage_steps(method),
                "n_samples": len(rows),
                "score_composed_mean": _mean(score_comp),
                "score_composed_std": _std(score_comp),
                "score_composed_ci95": _ci95(score_comp),
                "score_route_mean": _mean(score_route),
                "score_route_std": _std(score_route),
                "score_route_ci95": _ci95(score_route),
                "completion_rate": _mean(completed),
                "duration_system_mean": _mean(duration_sys),
                "duration_game_mean": _mean(duration_game),
                "collisions_total_per_route_mean": _mean(collisions_total),
                "collisions_per_km": collisions_per_km,
                "collisions_vehicle_per_route_mean": _mean(collisions_vehicle),
                "collisions_bike_per_route_mean": _mean(collisions_bike),
                "collisions_vehicle_only_per_route_mean": _mean(collisions_vehicle_only),
                "collisions_pedestrian_per_route_mean": _mean(collisions_pedestrian),
                "collisions_layout_per_route_mean": _mean(collisions_layout),
                "red_light_per_route_mean": _mean(red_light),
                "stop_infraction_per_route_mean": _mean(stop_infraction),
                "route_dev_per_route_mean": _mean(route_dev),
                "route_timeout_per_route_mean": _mean(route_timeout),
            }
        )

    summary_csv = os.path.join(args.output_dir, "method_summary.csv")
    summary_fields = list(summary_rows[0].keys())
    _write_csv(summary_csv, summary_fields, sorted(summary_rows, key=lambda r: r["method"]))

    methods = sorted(by_method.keys())
    if not ref_method:
        ref_method = methods[0]
    if ref_method not in by_method:
        raise SystemExit("Reference method '{}' not found in discovered methods".format(ref_method))

    key_maps = {}
    for method, rows in by_method.items():
        key_maps[method] = {r["route_key"]: r for r in rows}

    paired_rows = []
    ref_map = key_maps[ref_method]
    for method in methods:
        if method == ref_method:
            continue
        cmp_map = key_maps[method]
        common_keys = sorted(set(ref_map.keys()) & set(cmp_map.keys()))
        for key in common_keys:
            ref_row = ref_map[key]
            cmp_row = cmp_map[key]
            paired_rows.append(
                {
                    "reference_method": ref_method,
                    "compare_method": method,
                    "route_key": key,
                    "pass_index": ref_row["pass_index"],
                    "route_slot": ref_row["route_slot"],
                    "score_composed_ref": ref_row["score_composed"],
                    "score_composed_cmp": cmp_row["score_composed"],
                    "delta_score_composed": ref_row["score_composed"] - cmp_row["score_composed"],
                    "score_route_ref": ref_row["score_route"],
                    "score_route_cmp": cmp_row["score_route"],
                    "delta_score_route": ref_row["score_route"] - cmp_row["score_route"],
                    "completed_ref": ref_row["completed"],
                    "completed_cmp": cmp_row["completed"],
                    "delta_completed": ref_row["completed"] - cmp_row["completed"],
                    "collisions_total_ref": ref_row["collisions_total"],
                    "collisions_total_cmp": cmp_row["collisions_total"],
                    "delta_collisions_total": ref_row["collisions_total"] - cmp_row["collisions_total"],
                }
            )

    paired_csv = os.path.join(args.output_dir, "paired_deltas.csv")
    if paired_rows:
        paired_fields = list(paired_rows[0].keys())
        _write_csv(paired_csv, paired_fields, paired_rows)
    else:
        _write_csv(
            paired_csv,
            [
                "reference_method",
                "compare_method",
                "route_key",
                "pass_index",
                "route_slot",
                "score_composed_ref",
                "score_composed_cmp",
                "delta_score_composed",
                "score_route_ref",
                "score_route_cmp",
                "delta_score_route",
                "completed_ref",
                "completed_cmp",
                "delta_completed",
                "collisions_total_ref",
                "collisions_total_cmp",
                "delta_collisions_total",
            ],
            [],
        )

    print("Wrote: {}".format(route_csv))
    print("Wrote: {}".format(summary_csv))
    print("Wrote: {}".format(paired_csv))
    print("Reference method: {}".format(ref_method))


if __name__ == "__main__":
    main()
