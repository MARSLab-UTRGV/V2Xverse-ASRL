#!/usr/bin/env python3
"""
paper_percentages.py
Prints every percentage (and related statistic) cited in the Results section
of the paper, computed directly from the evaluation CSV files.

Usage:
    python tools/paper_percentages.py
    python tools/paper_percentages.py --data paper_outputs/final_eval3
"""

import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(data_dir: str):
    route   = pd.read_csv(os.path.join(data_dir, "route_metrics.csv"))
    summary = pd.read_csv(os.path.join(data_dir, "method_summary.csv"))
    paired  = pd.read_csv(os.path.join(data_dir, "paired_deltas.csv"))
    return route, summary, paired


def pct(num, den):
    return 100.0 * num / den


def reduction(a, b):
    """Percentage reduction from a to b."""
    return 100.0 * (a - b) / a


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def row(label: str, value, unit: str = "%"):
    if isinstance(value, float):
        print(f"  {label:<52}  {value:8.3f} {unit}")
    else:
        print(f"  {label:<52}  {value} {unit}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Print all paper percentages.")
    parser.add_argument(
        "--data",
        default=os.path.join(ROOT, "paper_outputs", "final_eval3"),
        help="Directory containing the three evaluation CSV files.",
    )
    args = parser.parse_args()

    route_df, summary_df, paired_df = load_data(args.data)

    methods = {
        "pid_only":        "PID-only",
        "fixed_cbf":       "Fixed CBF",
        "rl_cbf_adaptive": "Adaptive CBF",
    }

    # Index summary by method for easy lookup
    summ = summary_df.set_index("method")

    n = {m: int(summ.loc[m, "n_samples"]) for m in methods}

    # Per-method route DataFrames
    rt = {m: route_df[route_df["method"] == m].copy() for m in methods}

    # -----------------------------------------------------------------------
    section("Score Distribution")
    # -----------------------------------------------------------------------
    for key, label in methods.items():
        df   = rt[key]
        total = len(df)
        perfect = (df["score_composed"] == 100.0).sum()
        pct_perfect = pct(perfect, total)
        below10 = (df["score_composed"] < 10).sum()
        med = df["score_composed"].median()
        row(f"{label}: perfect score (==100)", pct_perfect)
        row(f"{label}: routes below 10",       pct(below10, total))
        row(f"{label}: median score",           med, "pts")
        row(f"{label}: mean score",             df["score_composed"].mean(), "pts")

    # -----------------------------------------------------------------------
    section("Collision-Free Route Rate")
    # -----------------------------------------------------------------------
    for key, label in methods.items():
        df      = rt[key]
        cf_rate = pct((df["collisions_total"] == 0).sum(), len(df))
        row(f"{label}: collision-free routes", cf_rate)

    # Gain adaptive vs others
    adp_cf = pct((rt["rl_cbf_adaptive"]["collisions_total"] == 0).sum(), n["rl_cbf_adaptive"])
    fix_cf = pct((rt["fixed_cbf"]["collisions_total"] == 0).sum(), n["fixed_cbf"])
    pid_cf = pct((rt["pid_only"]["collisions_total"] == 0).sum(), n["pid_only"])
    row("Gain adaptive vs fixed CBF (+pp)",   adp_cf - fix_cf, "pp")
    row("Gain adaptive vs PID-only (+pp)",    adp_cf - pid_cf, "pp")

    # -----------------------------------------------------------------------
    section("Collision Reduction: Adaptive CBF vs PID-only")
    # -----------------------------------------------------------------------
    pid_cpr  = summ.loc["pid_only",        "collisions_total_per_route_mean"]
    adp_cpr  = summ.loc["rl_cbf_adaptive", "collisions_total_per_route_mean"]
    pid_cpkm = summ.loc["pid_only",        "collisions_per_km"]
    adp_cpkm = summ.loc["rl_cbf_adaptive", "collisions_per_km"]
    pid_bike = summ.loc["pid_only",        "collisions_bike_per_route_mean"]
    adp_bike = summ.loc["rl_cbf_adaptive", "collisions_bike_per_route_mean"]
    pid_veh  = summ.loc["pid_only",        "collisions_vehicle_per_route_mean"]
    adp_veh  = summ.loc["rl_cbf_adaptive", "collisions_vehicle_per_route_mean"]
    pid_ped  = summ.loc["pid_only",        "collisions_pedestrian_per_route_mean"]
    adp_ped  = summ.loc["rl_cbf_adaptive", "collisions_pedestrian_per_route_mean"]

    row("Collisions/route reduction",   reduction(pid_cpr,  adp_cpr))
    row("Collisions/km reduction",      reduction(pid_cpkm, adp_cpkm))
    row("Bicycle collisions reduction", reduction(pid_bike, adp_bike))
    row("Vehicle collisions reduction", reduction(pid_veh,  adp_veh))
    row("Pedestrian coll. reduction",   reduction(pid_ped,  adp_ped))

    # -----------------------------------------------------------------------
    section("Collision Reduction: Adaptive CBF vs Fixed CBF")
    # -----------------------------------------------------------------------
    fix_cpr  = summ.loc["fixed_cbf", "collisions_total_per_route_mean"]
    fix_cpkm = summ.loc["fixed_cbf", "collisions_per_km"]
    fix_bike = summ.loc["fixed_cbf", "collisions_bike_per_route_mean"]
    fix_veh  = summ.loc["fixed_cbf", "collisions_vehicle_per_route_mean"]
    fix_ped  = summ.loc["fixed_cbf", "collisions_pedestrian_per_route_mean"]

    row("Collisions/route reduction",   reduction(fix_cpr,  adp_cpr))
    row("Collisions/km reduction",      reduction(fix_cpkm, adp_cpkm))
    row("Bicycle collisions reduction", reduction(fix_bike, adp_bike))
    row("Vehicle collisions reduction", reduction(fix_veh,  adp_veh))
    row("Pedestrian coll. reduction",   reduction(fix_ped,  adp_ped))

    # -----------------------------------------------------------------------
    section("Route Completion")
    # -----------------------------------------------------------------------
    for key, label in methods.items():
        row(f"{label}: route completion rate", summ.loc[key, "completion_rate"] * 100)

    adp_rc  = summ.loc["rl_cbf_adaptive", "completion_rate"] * 100
    pid_rc  = summ.loc["pid_only",        "completion_rate"] * 100
    fix_rc  = summ.loc["fixed_cbf",       "completion_rate"] * 100
    row("Adaptive CBF vs PID-only (+pp)",  adp_rc - pid_rc, "pp")
    row("Adaptive CBF vs Fixed CBF (+pp)", adp_rc - fix_rc, "pp")

    # Terminal status counts
    for key, label in methods.items():
        compl   = (rt[key]["status"] == "Completed").sum()
        blocked = (rt[key]["status"] == "Failed - Agent got blocked").sum()
        timed   = (rt[key]["status"] == "Failed - Agent timed out").sum()
        row(f"{label}: completed", int(compl),   "routes")
        row(f"{label}: blocked",   int(blocked), "routes")
        row(f"{label}: timeouts",  int(timed),   "routes")

    # -----------------------------------------------------------------------
    section("Driving Score Penalty Factor  (driving / route score)")
    # -----------------------------------------------------------------------
    for key, label in methods.items():
        ds = summ.loc[key, "score_composed_mean"]
        rs = summ.loc[key, "score_route_mean"]
        row(f"{label}: penalty factor", ds / rs, "")

    # -----------------------------------------------------------------------
    section("Per-Route Paired Comparison (Adaptive CBF as reference)")
    # -----------------------------------------------------------------------
    for cmp_key, cmp_label in [("fixed_cbf", "Fixed CBF"), ("pid_only", "PID-only")]:
        sub = paired_df[
            (paired_df["reference_method"] == "rl_cbf_adaptive") &
            (paired_df["compare_method"]   == cmp_key)
        ]
        total = len(sub)

        # Score
        wins_score   = (sub["delta_score_composed"] > 0).sum()
        losses_score = (sub["delta_score_composed"] < 0).sum()
        mean_delta   = sub["delta_score_composed"].mean()
        std_delta    = sub["delta_score_composed"].std()

        # Collisions
        fewer_coll = (sub["delta_collisions_total"] < 0).sum()
        more_coll  = (sub["delta_collisions_total"] > 0).sum()

        print(f"\n  vs {cmp_label}  ({total} paired routes)")
        row("  Score: wins  (adaptive higher)", pct(wins_score,   total))
        row("  Score: loses (adaptive lower)",  pct(losses_score, total))
        row("  Score: mean delta",              mean_delta, "pts")
        row("  Score: std  delta",              std_delta,  "pts")
        row("  Collisions: fewer (adaptive)",   pct(fewer_coll, total))
        row("  Collisions: more  (adaptive)",   pct(more_coll,  total))

    # -----------------------------------------------------------------------
    section("Infraction Breakdown (mean per route)")
    # -----------------------------------------------------------------------
    for key, label in methods.items():
        df = rt[key]
        stop   = df["stop_infraction"].mean()
        red    = df["red_light"].mean()
        timeout = df["route_timeout"].mean()
        row(f"{label}: stop-sign infractions/route", stop, "")
        row(f"{label}: red-light infractions/route", red,  "")
        row(f"{label}: timeouts/route",              timeout, "")

    print()


if __name__ == "__main__":
    main()
