#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _save(fig, out_base):
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)


def plot_overall(summary_rows, out_dir):
    methods = [r["method"] for r in summary_rows]

    score_comp = [_to_float(r["score_composed_mean"]) for r in summary_rows]
    score_comp_ci = [_to_float(r["score_composed_ci95"]) for r in summary_rows]
    score_route = [_to_float(r["score_route_mean"]) for r in summary_rows]
    score_route_ci = [_to_float(r["score_route_ci95"]) for r in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = list(range(len(methods)))

    axes[0].bar(x, score_comp, yerr=score_comp_ci, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=25, ha="right")
    axes[0].set_ylabel("Driving Score")
    axes[0].set_title("Overall Driving Score (mean ± 95% CI)")

    axes[1].bar(x, score_route, yerr=score_route_ci, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=25, ha="right")
    axes[1].set_ylabel("Route Completion Score")
    axes[1].set_title("Route Completion (mean ± 95% CI)")

    _save(fig, os.path.join(out_dir, "overall_scores"))


def plot_collisions(summary_rows, out_dir):
    methods = [r["method"] for r in summary_rows]
    collisions_per_km = [_to_float(r["collisions_per_km"]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(methods)))
    ax.bar(x, collisions_per_km)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Collisions per km")
    ax.set_title("Collision Rate")
    _save(fig, os.path.join(out_dir, "overall_collisions_per_km"))


def plot_infraction_breakdown(summary_rows, out_dir):
    methods = [r["method"] for r in summary_rows]
    metrics = [
        ("collisions_vehicle_per_route_mean", "veh_collision"),
        ("collisions_pedestrian_per_route_mean", "ped_collision"),
        ("collisions_layout_per_route_mean", "static_collision"),
        ("route_timeout_per_route_mean", "timeout"),
        ("route_dev_per_route_mean", "route_dev"),
        ("red_light_per_route_mean", "red_light"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.12
    x = list(range(len(methods)))

    for idx, (col, label) in enumerate(metrics):
        vals = [_to_float(r[col]) for r in summary_rows]
        offsets = [xi + (idx - (len(metrics) - 1) / 2.0) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Infractions per route")
    ax.set_title("Infraction Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, os.path.join(out_dir, "infraction_breakdown"))


def plot_paired_deltas(paired_rows, out_dir):
    by_cmp = defaultdict(list)
    for row in paired_rows:
        by_cmp[row["compare_method"]].append(_to_float(row["delta_score_composed"]))

    if not by_cmp:
        return

    for cmp_method, deltas in sorted(by_cmp.items()):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(deltas, bins=20)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Delta Driving Score (reference - baseline)")
        ax.set_ylabel("Route count")
        ax.set_title("Paired route deltas vs {}".format(cmp_method))
        safe_name = cmp_method.replace("/", "_")
        _save(fig, os.path.join(out_dir, "paired_delta_hist_{}".format(safe_name)))


def plot_stage_curve(summary_rows, out_dir):
    stage_rows = []
    for row in summary_rows:
        stage_steps = _to_int(row.get("stage_steps", ""), -1)
        if stage_steps >= 0:
            stage_rows.append((stage_steps, row))

    if len(stage_rows) < 2:
        return

    stage_rows.sort(key=lambda item: item[0])
    x = [item[0] for item in stage_rows]
    y_score = [_to_float(item[1]["score_composed_mean"]) for item in stage_rows]
    y_collision = [_to_float(item[1]["collisions_per_km"]) for item in stage_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(x, y_score, marker="o")
    axes[0].set_xlabel("Checkpoint steps")
    axes[0].set_ylabel("Driving Score")
    axes[0].set_title("Checkpoint Stage Curve: Score")

    axes[1].plot(x, y_collision, marker="o")
    axes[1].set_xlabel("Checkpoint steps")
    axes[1].set_ylabel("Collisions per km")
    axes[1].set_title("Checkpoint Stage Curve: Collision")

    _save(fig, os.path.join(out_dir, "checkpoint_stage_curve"))


def main():
    parser = argparse.ArgumentParser(description="Plot paper figures from aggregated eval CSVs")
    parser.add_argument("--summary-csv", default="paper_outputs/main/method_summary.csv")
    parser.add_argument("--paired-csv", default="paper_outputs/main/paired_deltas.csv")
    parser.add_argument("--output-dir", default="paper_outputs/main/figures")
    args = parser.parse_args()

    if not os.path.exists(args.summary_csv):
        raise SystemExit("Missing summary CSV: {}".format(args.summary_csv))

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = _read_csv(args.summary_csv)
    paired_rows = _read_csv(args.paired_csv) if os.path.exists(args.paired_csv) else []

    if not summary_rows:
        raise SystemExit("Summary CSV has no rows: {}".format(args.summary_csv))

    plot_overall(summary_rows, args.output_dir)
    plot_collisions(summary_rows, args.output_dir)
    plot_infraction_breakdown(summary_rows, args.output_dir)
    plot_paired_deltas(paired_rows, args.output_dir)
    plot_stage_curve(summary_rows, args.output_dir)

    print("Wrote figures to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
