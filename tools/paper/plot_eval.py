#!/usr/bin/env python3
"""Generate paper-quality evaluation figures from aggregated CSV outputs."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------
METHOD_ORDER = ["pid_only", "fixed_cbf", "rl_cbf_adaptive"]
METHOD_LABELS = {
    "pid_only": "No CBF",
    "fixed_cbf": "Fixed CBF",
    "rl_cbf_adaptive": "Adaptive CBF",
}
COLORS = {
    "pid_only": "#de2d26",        # No CBF
    "fixed_cbf": "#8856a7",       # Fixed CBF
    "rl_cbf_adaptive": "#2c7fb8",  # Adaptive CBF
}
INFRACTION_COLORS = {
    "Bicycle": "#2ca25f",
    "Vehicle": "#2b8cbe",
    "Pedestrian": "#756bb1",
    "Stop Sign": "#fec44f",
    "Static": "#dd1c77",
    "Red Light": "#a8ddb5",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


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
    fig.savefig(out_base + ".png", dpi=200)
    plt.close(fig)


def _sort_rows(rows):
    """Return summary rows in canonical METHOD_ORDER."""
    lut = {r["method"]: r for r in rows}
    return [lut[m] for m in METHOD_ORDER if m in lut]


def _method_colors(methods):
    return [COLORS.get(m, "#333333") for m in methods]


# ---------------------------------------------------------------------------
# Fig 1 — Driving score bar chart (single panel, no route completion)
# ---------------------------------------------------------------------------
def plot_driving_score(summary_rows, out_dir):
    rows = _sort_rows(summary_rows)
    methods = [r["method"] for r in rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    scores = [_to_float(r["score_composed_mean"]) for r in rows]
    ci = [_to_float(r["score_composed_ci95"]) for r in rows]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(labels, scores, yerr=ci, capsize=5,
                  color=_method_colors(methods), edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Driving Score")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    # value labels
    for bar, s, c in zip(bars, scores, ci):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + c + 1.5,
                f"{s:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "overall_scores"))


# ---------------------------------------------------------------------------
# Fig 2 — Collision breakdown: stacked bar per class (vehicle, bike, ped, static)
# ---------------------------------------------------------------------------
def plot_collision_breakdown(summary_rows, out_dir):
    rows = _sort_rows(summary_rows)
    methods = [r["method"] for r in rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    # If bike column exists use it; otherwise fall back to 0
    has_bike = "collisions_bike_per_route_mean" in rows[0]
    veh = []
    for r in rows:
        if has_bike:
            veh.append(_to_float(r.get("collisions_vehicle_only_per_route_mean", 0)))
        else:
            veh.append(_to_float(r["collisions_vehicle_per_route_mean"]))
    bike = [_to_float(r.get("collisions_bike_per_route_mean", 0)) for r in rows] if has_bike else [0]*len(rows)
    ped = [_to_float(r["collisions_pedestrian_per_route_mean"]) for r in rows]
    layout = [_to_float(r["collisions_layout_per_route_mean"]) for r in rows]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    x = range(len(labels))
    bottom = [0.0] * len(labels)

    # Layering order requested: Bicycle -> Vehicle -> Pedestrian (then Static).
    categories = [
        (bike, "Bicycle", INFRACTION_COLORS["Bicycle"]),
        (veh, "Vehicle", INFRACTION_COLORS["Vehicle"]),
        (ped, "Pedestrian", INFRACTION_COLORS["Pedestrian"]),
        (layout, "Static", INFRACTION_COLORS["Static"]),
    ]
    for vals, label, color in categories:
        if max(vals) < 1e-6:
            continue
        ax.bar(x, vals, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.5, width=0.55)
        bottom = [b + v for b, v in zip(bottom, vals)]

    # Total label on top
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 0.03, f"{tot:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Collisions per Route")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "collision_breakdown"))


# ---------------------------------------------------------------------------
# Fig 3 — Collision-free route rate
# ---------------------------------------------------------------------------
def plot_collision_free_rate(route_rows, out_dir):
    by_method = defaultdict(list)
    for r in route_rows:
        by_method[r["method"]].append(_to_int(r["collisions_total"]))

    methods = [m for m in METHOD_ORDER if m in by_method]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    rates = []
    for m in methods:
        colls = by_method[m]
        rates.append(sum(1 for c in colls if c == 0) / len(colls) * 100)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(labels, rates, color=_method_colors(methods),
                  edgecolor="black", linewidth=0.6, width=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Collision-Free Routes (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "collision_free_rate"))


# ---------------------------------------------------------------------------
# Fig 4 — Collisions per km
# ---------------------------------------------------------------------------
def plot_collisions_per_km(summary_rows, out_dir):
    rows = _sort_rows(summary_rows)
    methods = [r["method"] for r in rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    cpk = [_to_float(r["collisions_per_km"]) for r in rows]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(labels, cpk, color=_method_colors(methods),
                  edgecolor="black", linewidth=0.6, width=0.5)
    for bar, v in zip(bars, cpk):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Collisions / km")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "overall_collisions_per_km"))


# ---------------------------------------------------------------------------
# Fig 5 — Infraction breakdown (grouped bars)
# ---------------------------------------------------------------------------
def plot_infraction_breakdown(summary_rows, out_dir):
    rows = _sort_rows(summary_rows)
    methods = [r["method"] for r in rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    has_bike = "collisions_bike_per_route_mean" in rows[0]
    vehicle_key = "collisions_vehicle_only_per_route_mean" if has_bike else "collisions_vehicle_per_route_mean"

    # Requested order and colors:
    # Bicycle, Vehicle, Pedestrian, Stop Sign, Static
    # Red-light class removed; its previous color is now used for Pedestrian.
    metrics = [
        ("collisions_bike_per_route_mean", "Bicycle"),
        (vehicle_key, "Vehicle"),
        ("collisions_pedestrian_per_route_mean", "Pedestrian"),
        ("stop_infraction_per_route_mean", "Stop Sign"),
        ("collisions_layout_per_route_mean", "Static"),
    ]

    n_groups = len(labels)
    n_bars = len(metrics)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for idx, (col, mlabel) in enumerate(metrics):
        vals = [_to_float(r.get(col, 0)) for r in rows]
        offsets = [i + (idx - (n_bars - 1) / 2.0) * width for i in range(n_groups)]
        ax.bar(offsets, vals, width=width, label=mlabel,
               color=INFRACTION_COLORS.get(mlabel, "#999999"), edgecolor="white", linewidth=0.4)

    ax.set_xticks(list(range(n_groups)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Infractions per Route")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "infraction_breakdown"))


# ---------------------------------------------------------------------------
# Fig 5b — Individual actor-collision plots (per route)
# ---------------------------------------------------------------------------
def _plot_single_collision_metric(summary_rows, out_dir, metric_key, title, out_name, bar_color):
    rows = _sort_rows(summary_rows)
    if not rows:
        return
    if metric_key not in rows[0]:
        return

    methods = [r["method"] for r in rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    vals = [_to_float(r.get(metric_key, 0)) for r in rows]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(labels, vals, color=bar_color, edgecolor="black", linewidth=0.6, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Collisions per Route")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, out_name))


def plot_individual_actor_collisions(summary_rows, out_dir):
    rows = _sort_rows(summary_rows)
    if not rows:
        return

    has_bike = "collisions_bike_per_route_mean" in rows[0]
    vehicle_key = "collisions_vehicle_only_per_route_mean" if has_bike else "collisions_vehicle_per_route_mean"

    _plot_single_collision_metric(
        summary_rows,
        out_dir,
        "collisions_bike_per_route_mean",
        "Bicycle Collisions per Route",
        "bicycle_collisions_per_route",
        INFRACTION_COLORS["Bicycle"],
    )
    _plot_single_collision_metric(
        summary_rows,
        out_dir,
        vehicle_key,
        "Vehicle Collisions per Route",
        "vehicle_collisions_per_route",
        INFRACTION_COLORS["Vehicle"],
    )
    _plot_single_collision_metric(
        summary_rows,
        out_dir,
        "collisions_pedestrian_per_route_mean",
        "Pedestrian Collisions per Route",
        "pedestrian_collisions_per_route",
        INFRACTION_COLORS["Pedestrian"],
    )


# ---------------------------------------------------------------------------
# Fig 6b — All infractions in one graph (grouped horizontal bars)
# ---------------------------------------------------------------------------
def plot_all_infractions(route_rows, out_dir):
    if not route_rows:
        return

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in route_rows)]
    if not methods:
        return

    has_bike = "collisions_bike" in route_rows[0]
    metrics = []
    if has_bike:
        # Requested vertical order (bottom -> top):
        # Bicycle, Vehicle, Pedestrian, Stop Sign, Static
        metrics.extend([
            ("collisions_bike", "Bicycle"),
            ("collisions_vehicle_only", "Vehicle"),
        ])
    else:
        metrics.append(("collisions_vehicle", "Vehicle"))
    metrics.extend([
        ("collisions_pedestrian", "Pedestrian"),
        ("stop_infraction", "Stop Sign"),
        ("collisions_layout", "Static"),
    ])

    # Keep only metrics that actually exist in route CSV columns.
    metrics = [(k, lbl) for k, lbl in metrics if k in route_rows[0]]
    if not metrics:
        return

    labels = [lbl for _, lbl in metrics]
    y = list(range(len(metrics)))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bar_h = 0.8 / max(len(methods), 1)
    max_val = 0.0

    for idx, method in enumerate(methods):
        rows = [r for r in route_rows if r["method"] == method]
        vals = []
        for key, _ in metrics:
            key_vals = [_to_float(r.get(key, 0)) for r in rows]
            vals.append(sum(key_vals) / max(len(key_vals), 1))
        max_val = max(max_val, max(vals) if vals else 0.0)

        offset = (idx - (len(methods) - 1) / 2.0) * bar_h
        y_pos = [yy + offset for yy in y]
        ax.barh(
            y_pos,
            vals,
            height=bar_h,
            color=COLORS.get(method, "#333333"),
            edgecolor="white",
            linewidth=0.4,
            label=METHOD_LABELS.get(method, method),
        )
        for yp, v in zip(y_pos, vals):
            ax.text(
                v + max(max_val * 0.012, 0.01),
                yp,
                f"{v:.3f}",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Infractions per Route")
    ax.set_xlim(0, max_val * 1.20 + 0.05)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "all_infractions_per_route"))


# ---------------------------------------------------------------------------
# Fig 7 — Per-route score distributions (violin)
# ---------------------------------------------------------------------------
def plot_score_distributions(route_rows, out_dir):
    by_method = defaultdict(list)
    for r in route_rows:
        by_method[r["method"]].append(_to_float(r["score_composed"]))

    methods = [m for m in METHOD_ORDER if m in by_method]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    data = [by_method[m] for m in methods]
    colors = _method_colors(methods)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    vp = ax.violinplot(data, positions=range(len(methods)), showmeans=True,
                       showmedians=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.6)
    vp["cmeans"].set_color("black")
    vp["cmedians"].set_color("red")
    vp["cmedians"].set_linewidth(1.5)
    vp["cmeans"].set_linewidth(1.5)

    ax.set_xticks(list(range(len(methods))))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Driving Score")
    ax.set_ylim(-5, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "score_distribution"))


# ---------------------------------------------------------------------------
# Fig 8 — Per-route driving score box plot
# ---------------------------------------------------------------------------
def plot_overall_scores_boxplot(route_rows, out_dir):
    by_method_score = defaultdict(list)
    for r in route_rows:
        m = r["method"]
        by_method_score[m].append(_to_float(r.get("score_composed", 0)))

    methods = [m for m in METHOD_ORDER if m in by_method_score]
    if not methods:
        return

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = _method_colors(methods)
    score_data = [by_method_score[m] for m in methods]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bp = ax.boxplot(
        score_data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        boxprops={"edgecolor": "#333333", "linewidth": 1.0},
    )
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.55)

    means = [sum(vals) / max(len(vals), 1) for vals in score_data]
    ax.scatter(range(1, len(methods) + 1), means, marker="D", color="black", s=16, zorder=3)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Driving Score")
    ax.set_ylim(-5, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "overall_scores_boxplot"))


# ---------------------------------------------------------------------------
# Fig 9 — Safety-performance Pareto scatter
# ---------------------------------------------------------------------------
def plot_pareto(summary_rows, out_dir):
    rows = _sort_rows(summary_rows)
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for r in rows:
        method = r["method"]
        x = _to_float(r.get("collisions_per_km", 0))
        y = _to_float(r.get("score_composed_mean", 0))
        yerr = _to_float(r.get("score_composed_ci95", 0))
        color = COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Collisions / km (lower is better)")
    ax.set_ylabel("Driving Score (higher is better)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "pareto_score_vs_collisions"))


def _status_bucket(status):
    if status.startswith("Completed"):
        return "Completed"
    if "blocked" in status.lower():
        return "Blocked"
    if "timed out" in status.lower() or "timeout" in status.lower():
        return "Timeout"
    if "deviated" in status.lower():
        return "Route Deviation"
    if status.startswith("Failed"):
        return "Other Failure"
    return "Other"


# ---------------------------------------------------------------------------
# Fig 10 — Failure mode composition
# ---------------------------------------------------------------------------
def plot_failure_modes(route_rows, out_dir):
    if not route_rows:
        return

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in route_rows)]
    if not methods:
        return

    categories = ["Completed", "Blocked", "Timeout", "Route Deviation", "Other Failure"]
    cat_colors = {
        "Completed": "#2ecc71",
        "Blocked": "#e74c3c",
        "Timeout": "#f39c12",
        "Route Deviation": "#3498db",
        "Other Failure": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = list(range(len(methods)))
    bottom = [0.0] * len(methods)

    for cat in categories:
        vals = []
        for m in methods:
            rows_m = [r for r in route_rows if r["method"] == m]
            if not rows_m:
                vals.append(0.0)
                continue
            cnt = sum(1 for r in rows_m if _status_bucket(str(r.get("status", ""))) == cat)
            vals.append(cnt / len(rows_m) * 100.0)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=cat_colors[cat],
            width=0.55,
            edgecolor="white",
            linewidth=0.4,
            label=cat,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylabel("Route Outcomes (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", ncol=2, fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "failure_modes"))


# ---------------------------------------------------------------------------
# Fig 11 — Win / Tie / Loss vs baselines (paired score deltas)
# ---------------------------------------------------------------------------
def plot_win_tie_loss(paired_rows, out_dir):
    if not paired_rows:
        return

    by_cmp = defaultdict(list)
    for r in paired_rows:
        by_cmp[r.get("compare_method", "")].append(_to_float(r.get("delta_score_composed", 0)))

    baselines = [m for m in METHOD_ORDER if m in by_cmp and m != "rl_cbf_adaptive"]
    if not baselines:
        baselines = [m for m in sorted(by_cmp.keys()) if m]
    if not baselines:
        return

    wins, ties, losses = [], [], []
    for b in baselines:
        vals = by_cmp[b]
        n = max(len(vals), 1)
        wins.append(sum(1 for v in vals if v > 0) / n * 100.0)
        ties.append(sum(1 for v in vals if v == 0) / n * 100.0)
        losses.append(sum(1 for v in vals if v < 0) / n * 100.0)

    x = list(range(len(baselines)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.bar([i - width for i in x], wins, width=width, color="#2ecc71", label="Win")
    ax.bar(x, ties, width=width, color="#95a5a6", label="Tie")
    ax.bar([i + width for i in x], losses, width=width, color="#e74c3c", label="Loss")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in baselines])
    ax.set_ylabel("Paired Routes (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "win_tie_loss"))


# ---------------------------------------------------------------------------
# Stage curve (optional, for checkpoint sweeps)
# ---------------------------------------------------------------------------
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(x, y_score, marker="o", color="#e74c3c", linewidth=1.5)
    axes[0].set_xlabel("Checkpoint Steps")
    axes[0].set_ylabel("Driving Score")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(x, y_collision, marker="o", color="#3498db", linewidth=1.5)
    axes[1].set_xlabel("Checkpoint Steps")
    axes[1].set_ylabel("Collisions / km")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    _save(fig, os.path.join(out_dir, "checkpoint_stage_curve"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot paper figures from aggregated eval CSVs")
    parser.add_argument("--summary-csv", default="paper_outputs/main/method_summary.csv")
    parser.add_argument("--route-csv", default="paper_outputs/main/route_metrics.csv")
    parser.add_argument("--paired-csv", default="paper_outputs/main/paired_deltas.csv")
    parser.add_argument("--output-dir", default="paper_outputs/main/figures")
    args = parser.parse_args()

    if not os.path.exists(args.summary_csv):
        raise SystemExit("Missing summary CSV: {}".format(args.summary_csv))

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = _read_csv(args.summary_csv)
    route_rows = _read_csv(args.route_csv) if os.path.exists(args.route_csv) else []
    paired_rows = _read_csv(args.paired_csv) if os.path.exists(args.paired_csv) else []

    if not summary_rows:
        raise SystemExit("Summary CSV has no rows: {}".format(args.summary_csv))

    plot_driving_score(summary_rows, args.output_dir)
    plot_collision_breakdown(summary_rows, args.output_dir)
    plot_collisions_per_km(summary_rows, args.output_dir)
    plot_infraction_breakdown(summary_rows, args.output_dir)
    plot_individual_actor_collisions(summary_rows, args.output_dir)
    plot_pareto(summary_rows, args.output_dir)
    if route_rows:
        plot_collision_free_rate(route_rows, args.output_dir)
        plot_score_distributions(route_rows, args.output_dir)
        plot_overall_scores_boxplot(route_rows, args.output_dir)
        plot_all_infractions(route_rows, args.output_dir)
    plot_stage_curve(summary_rows, args.output_dir)

    print("Wrote figures to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
