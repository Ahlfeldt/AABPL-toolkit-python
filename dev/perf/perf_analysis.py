"""
Analyse dev/perf/perf_test/ results.

Run cells top-to-bottom in a Jupyter notebook, or execute as a script.
Requires: pandas, matplotlib, numpy.

Timing columns available in the saved JSONs
-------------------------------------------
func_build_disk_region_lookups   -- geometry build (0 when geometry_cached=True)
func_search_and_aggregate        -- per-call search loop
func_set_target                  -- target grid setup (assign + aggregate)
func_set_source                  -- source assignment + micro-region classification
func_perform_search              -- inner search (sub-component of search_and_aggregate)
total_cpu_s                      -- sum of all measured CPU time
total_wall_s                     -- wall time (unreliable under parallel load)

Note: micro_region_stats are currently empty in the saved files (silently suppressed).
Geometric insight comes from radius_over_spacing + nest_depth instead.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PERF_FOLDER = os.path.join(os.path.dirname(__file__), "perf_test")
PERF_GEN_FOLDER = os.path.join(os.path.dirname(__file__), "perf_test_gen")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(folder: str = PERF_FOLDER) -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception:
            continue
        row = {}
        row["_file"] = fname
        row.update(d.get("config", {}))
        row.update(d.get("timing", {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    # derived columns
    if "func_search_and_aggregate" in df.columns and "n_source" in df.columns:
        df["search_ms_per_pt"] = df["func_search_and_aggregate"] / df["n_source"] * 1000
    if "func_set_target" in df.columns and "n_target" in df.columns:
        df["target_ms_per_pt"] = df["func_set_target"] / df["n_target"] * 1000
    if "func_set_source" in df.columns and "n_source" in df.columns:
        df["source_ms_per_pt"] = df["func_set_source"] / df["n_source"] * 1000
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_rs(r_over_s: float) -> str:
    return f"r/s={r_over_s:.3g}"


def _pivot_mean(df: pd.DataFrame, val: str):
    """Mean of val, pivoted: rows=nest_depth, cols=radius_over_spacing."""
    return (
        df.groupby(["nest_depth", "radius_over_spacing"])[val]
        .mean()
        .unstack("radius_over_spacing")
    )


# ---------------------------------------------------------------------------
# Section 1 – Geometry build time
# ---------------------------------------------------------------------------

def plot_build_time(df: pd.DataFrame):
    """
    func_build_disk_region_lookups vs nest_depth, coloured by radius_over_spacing.
    Only uses rows where geometry was NOT cached (first-ever build for that config).
    """
    builds = df[~df["geometry_cached"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Geometry build time  (geometry_cached=False only)", fontsize=13)

    # -- left: raw build time heatmap -----------------------------------------
    ax = axes[0]
    piv = _pivot_mean(builds, "func_build_disk_region_lookups")
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:.3g}" for v in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("radius / spacing")
    ax.set_ylabel("nest_depth")
    ax.set_title("Build time (CPU s)")
    plt.colorbar(im, ax=ax)
    for r in range(len(piv.index)):
        for c in range(len(piv.columns)):
            v = piv.values[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if v < piv.values.max() * 0.6 else "white")

    # -- right: build time vs nest_depth line plot per r/s --------------------
    ax = axes[1]
    r_over_s_vals = sorted(builds["radius_over_spacing"].unique())
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(r_over_s_vals) - 1)
    for i, rs in enumerate(r_over_s_vals):
        sub = builds[builds["radius_over_spacing"] == rs].groupby("nest_depth")["func_build_disk_region_lookups"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=_label_rs(rs), color=cmap(norm(i)))
    ax.set_xlabel("nest_depth")
    ax.set_ylabel("Build time (CPU s)")
    ax.set_title("Build time vs nest_depth")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 2 – Search time per source point
# ---------------------------------------------------------------------------

def plot_search_time(df: pd.DataFrame):
    """
    search_ms_per_pt = func_search_and_aggregate / n_source * 1000
    Separated by trynew, vs (radius_over_spacing, nest_depth).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Search time per source point (ms)  — bias-free metric", fontsize=13)

    for ax, tn in zip(axes, [0, 1]):
        sub = df[df["trynew"] == tn]
        piv = _pivot_mean(sub, "search_ms_per_pt")
        im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.3g}" for v in piv.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("radius / spacing")
        ax.set_ylabel("nest_depth")
        ax.set_title(f"trynew={tn}  (ms / source pt)")
        plt.colorbar(im, ax=ax)
        for r in range(len(piv.index)):
            for c in range(len(piv.columns)):
                v = piv.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
                            color="black" if v < piv.values[~np.isnan(piv.values)].max() * 0.6 else "white")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 3 – Timing breakdown (stacked bar per config)
# ---------------------------------------------------------------------------

def plot_timing_breakdown(df: pd.DataFrame):
    """
    Stacked bars showing which functions dominate total_cpu_s.
    Rows = geometry_cached=False (full run with build).
    Grouped by (radius_over_spacing, nest_depth, trynew).
    """
    COMPONENTS = [
        ("func_build_disk_region_lookups", "Geometry build", "#d62728"),
        ("func_set_target",               "Set target",     "#1f77b4"),
        ("func_set_source",               "Set source",     "#ff7f0e"),
        ("func_search_and_aggregate",     "Search",         "#2ca02c"),
    ]

    sub = df[~df["geometry_cached"]].copy()
    sub["_label"] = (
        "r/s=" + sub["radius_over_spacing"].apply(lambda x: f"{x:.3g}")
        + " nd=" + sub["nest_depth"].astype(str)
        + " tn=" + sub["trynew"].astype(str)
    )
    stat = sub.groupby("_label")[[c for c, _, _ in COMPONENTS]].mean()

    fig, ax = plt.subplots(figsize=(max(10, len(agg) * 0.6), 5))
    bottoms = np.zeros(len(agg))
    for col, label, color in COMPONENTS:
        if col in agg.columns:
            vals = agg[col].values
            ax.bar(range(len(agg)), vals, bottom=bottoms, label=label, color=color, alpha=0.85)
            bottoms += vals

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("CPU seconds")
    ax.set_title("Timing breakdown by config  (geometry_cached=False)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 4 – trynew=0 vs trynew=1 comparison
# ---------------------------------------------------------------------------

def plot_trynew_comparison(df: pd.DataFrame):
    """
    For each (radius_over_spacing, nest_depth): compare mean search_ms_per_pt
    for trynew=0 vs trynew=1.  Points above diagonal → trynew=1 is faster.
    """
    grp = (
        df.groupby(["radius_over_spacing", "nest_depth", "trynew"])["search_ms_per_pt"]
        .mean()
        .unstack("trynew")
        .rename(columns={0: "tn0", 1: "tn1"})
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(grp["tn0"], grp["tn1"],
                    c=grp.index.get_level_values("nest_depth"),
                    cmap="plasma", s=60, zorder=3)
    lim = max(grp[["tn0", "tn1"]].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("search ms/pt  (trynew=0)")
    ax.set_ylabel("search ms/pt  (trynew=1)")
    ax.set_title("trynew=0 vs trynew=1  (each dot = one r/s × nd combo)")
    plt.colorbar(sc, ax=ax, label="nest_depth")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # which is faster?
    grp["faster"] = np.where(grp["tn0"] < grp["tn1"], "tn0", "tn1")
    print("\ntrynew winner per config:")
    print(grp.to_string())


# ---------------------------------------------------------------------------
# Section 5 – Best config ranking per radius
# ---------------------------------------------------------------------------

def best_config_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each radius_over_spacing, rank configs by mean search_ms_per_pt.
    Returns a DataFrame sorted from fastest to slowest within each r/s group.
    """
    grp = (
        df.groupby(["radius_over_spacing", "nest_depth", "trynew"])["search_ms_per_pt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp.columns = ["radius_over_spacing", "nest_depth", "trynew",
                   "search_ms_per_pt_mean", "search_ms_per_pt_std", "n_runs"]
    grp["rank"] = grp.groupby("radius_over_spacing")["search_ms_per_pt_mean"].rank()
    grp = grp.sort_values(["radius_over_spacing", "rank"])
    return grp


# ---------------------------------------------------------------------------
# Section 6 – Build time vs r/spacing analytical structure
# ---------------------------------------------------------------------------

def plot_build_vs_rs(df: pd.DataFrame):
    """
    Show how geometry build time scales with radius_over_spacing on a log scale.
    Helps understand the structural inflection points (sqrt(2), 1.0 thresholds).
    """
    builds = df[~df["geometry_cached"]].copy()
    grp = builds.groupby(["radius_over_spacing", "nest_depth"])["func_build_disk_region_lookups"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    for nd, sub in grp.groupby("nest_depth"):
        ax.plot(sub["radius_over_spacing"], sub["func_build_disk_region_lookups"],
                marker="o", label=f"nest_depth={nd}")

    # mark geometric thresholds
    for xv, lbl in [(1/np.sqrt(2), "1/√2≈0.707"), (1.0, "1.0"), (np.sqrt(2), "√2≈1.414")]:
        ax.axvline(xv, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.text(xv, 0.97, lbl, transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=7, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("radius / spacing  (log scale)")
    ax.set_ylabel("Build time (CPU s)")
    ax.set_title("Geometry build time vs radius/spacing")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main – run all sections
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading results from {PERF_FOLDER} …")
    df = load_results(PERF_FOLDER)
    print(f"  {len(df)} rows loaded")
    print(f"  geometry_cached distribution: {df['geometry_cached'].value_counts().to_dict()}")
    print(f"  radius_over_spacing values: {sorted(df['radius_over_spacing'].unique())}")
    print(f"  nest_depth values: {sorted(df['nest_depth'].unique())}")
    print()

    print("=== Section 1: Geometry build time ===")
    plot_build_time(df)

    print("=== Section 2: Search time per source point ===")
    plot_search_time(df)

    print("=== Section 3: Timing breakdown ===")
    plot_timing_breakdown(df)

    print("=== Section 4: trynew comparison ===")
    plot_trynew_comparison(df)

    print("=== Section 5: Best config ranking ===")
    ranking = best_config_table(df)
    print(ranking.to_string(index=False))

    print("=== Section 6: Build time vs r/spacing ===")
    plot_build_vs_rs(df)
