"""
plotting.py  —  Benchmark 3: Fixed-N vs Adaptive-N figures
================================================================================

Reads the three CSV files produced by bench3_fixed_vs_adaptive.cpp and writes
publication-quality figures suitable for a Computer Physics Communications paper.

Input files (must be in the same directory, or pass --datadir):
    bench3_adaptive.csv
    bench3_fixed.csv
    bench3_comparison.csv

Output files (written to --outdir, default: current directory):
    fig1_bar_N_adaptive.pdf      — horizontal bar chart, the main figure
    fig2_waste_factor.pdf        — waste-factor panel per fixed-N value
    fig3_relsem_heatmap.pdf      — achieved rel-SEM heat-map across (obs, N)
    fig4_runtime.pdf             — runtime comparison: adaptive vs fixed

Usage:
    python plotting.py
    python plotting.py --datadir results/ --outdir figures/ --fmt pdf
    python plotting.py --fmt png --dpi 300

Dependencies:
    numpy, pandas, matplotlib  (all standard in any scientific Python stack)
================================================================================
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, NullFormatter

# -----------------------------------------------------------------------------
# Style — matches CPC / Elsevier single-column width (86 mm) and
# double-column width (180 mm).  All font sizes are set explicitly so the
# figure looks correct at 1:1 in the PDF without further scaling.
# -----------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          8,
    "axes.titlesize":     8,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.6,
    "text.usetex":        False,   # set True if LaTeX is available
})

# -----------------------------------------------------------------------------
# Colour palette  (colour-blind-safe, works in greyscale too)
# -----------------------------------------------------------------------------

C_ADAPTIVE   = "#2166ac"   # blue  — adaptive bar
C_WASTE      = "#d6604d"   # red   — over-sampled
C_MATCHED    = "#4dac26"   # green — matched
C_UNDER      = "#f4a582"   # salmon — under-sampled
C_REFLINE    = "#666666"   # grey  — fixed-N reference lines
C_GRID       = "#dddddd"

# Observable display labels (LaTeX-style for axis tick labels)
OBS_LABEL = {
    "Pe": r"$P_e$",
    "Pg": r"$P_g$",
    "sz": r"$\sigma_z$",
}

FIXED_N_LABELS = {
    1000:  r"$N{=}1\,000$",
    5000:  r"$N{=}5\,000$",
    10000: r"$N{=}10\,000$",
    50000: r"$N{=}50\,000$",
}

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_csvs(datadir: str):
    def rd(name):
        path = os.path.join(datadir, name)
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found.  Run bench3_fixed_vs_adaptive first.")
        return pd.read_csv(path)

    adapt = rd("bench3_adaptive.csv")
    fixed = rd("bench3_fixed.csv")
    cmp   = rd("bench3_comparison.csv")
    return adapt, fixed, cmp


def save(fig, outdir, stem, fmt):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{stem}.{fmt}")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Derived data helpers
# -----------------------------------------------------------------------------

def adaptive_medians(adapt_df):
    """Return dict obs -> (N_median, rel_sem_median, runtime_median)."""
    out = {}
    for obs, grp in adapt_df.groupby("observable"):
        out[obs] = (
            int(grp["N_adaptive"].median()),
            grp["rel_sem"].median(),
            grp["runtime_s"].median(),
        )
    return out


def obs_order(df):
    """Return observables sorted by ascending median N_adaptive."""
    med = df.groupby("observable")["N_adaptive"].median()
    return list(med.sort_values().index)


# -----------------------------------------------------------------------------
# Figure 1 — Horizontal bar chart  (THE main figure)
#
#   Bars   = N_adaptive (median, with min/max whiskers from 3 replicas)
#   Vlines = each fixed-N value (styled differently)
#   Colour = single blue; insufficient fixed-N gets a warning annotation.
#
#   Single-column width: 86 mm → 3.39 in
# -----------------------------------------------------------------------------

def fig1_bar_N_adaptive(adapt_df, outdir, fmt):
    fixed_Ns = [1000, 5000, 10000, 50000]

    obs_list  = obs_order(adapt_df)   # ascending N
    n_obs     = len(obs_list)

    # Aggregate replicas
    agg = (adapt_df.groupby("observable")["N_adaptive"]
                   .agg(["median", "min", "max"])
                   .reindex(obs_list))

    fig, ax = plt.subplots(figsize=(3.39, 0.9 + 0.55 * n_obs))

    y_pos = np.arange(n_obs)
    bar_h = 0.45

    # -- Bars -----------------------------------------------------------
    bars = ax.barh(
        y_pos,
        agg["median"].values,
        height=bar_h,
        color=C_ADAPTIVE,
        zorder=3,
        label="Adaptive $N$ (median)",
    )

    # Whiskers (min–max range from 3 replicas)
    for i, obs in enumerate(obs_list):
        lo  = agg.loc[obs, "min"]
        hi  = agg.loc[obs, "max"]
        med = agg.loc[obs, "median"]
        ax.plot([lo, hi], [y_pos[i], y_pos[i]],
                color="white", lw=1.2, zorder=4)
        ax.plot([lo, hi], [y_pos[i], y_pos[i]],
                color=C_ADAPTIVE, lw=0.8, zorder=5,
                solid_capstyle="round")
        ax.plot([lo, lo], [y_pos[i] - bar_h*0.3, y_pos[i] + bar_h*0.3],
                color=C_ADAPTIVE, lw=0.8, zorder=5)
        ax.plot([hi, hi], [y_pos[i] - bar_h*0.3, y_pos[i] + bar_h*0.3],
                color=C_ADAPTIVE, lw=0.8, zorder=5)

    # -- Fixed-N reference lines -------------------------------------------------
    line_styles = [
        (1000,  ":",  0.9),
        (5000,  "--", 0.9),
        (10000, "-",  1.1),
        (50000, "-.", 0.9),
    ]
    for N_fix, ls, lw in line_styles:
        ax.axvline(N_fix, color=C_REFLINE, linestyle=ls, linewidth=lw,
                   zorder=2, label=FIXED_N_LABELS[N_fix])

    # -- Value labels inside / outside bars -------------------------------------
    x_max = ax.get_xlim()[1]
    for i, obs in enumerate(obs_list):
        med = agg.loc[obs, "median"]
        label_text = f"{med:,}"
        x_label = med + x_max * 0.01
        ax.text(x_label, y_pos[i], label_text,
                va="center", ha="left", fontsize=6.5, color=C_ADAPTIVE)

    # -- Axes formatting -----------------------------------------------------
    ax.set_yticks(y_pos)
    ax.set_yticklabels([OBS_LABEL.get(o, o) for o in obs_list], fontsize=8)
    ax.set_xlabel("Number of trajectories $N$")
    ax.set_title(r"Adaptive $N$ vs fixed-$N$ reference lines ($\varepsilon = 1\%$)",
                 pad=4)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.grid(axis="x", color=C_GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # -- Legend ---------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc="lower right", framealpha=0.9,
              edgecolor="0.8", fontsize=6.5, ncol=1)

    fig.tight_layout()
    save(fig, outdir, "fig1_bar_N_adaptive", fmt)


# -----------------------------------------------------------------------------
# Figure 2 — Waste-factor panel
#
#   4 sub-panels, one per fixed-N value.
#   Bar height = waste_factor = N_fixed / N_adaptive.
#   Colour: red if > 1.05 (waste), green if ≈ 1 (matched), salmon if < 0.95.
#   Horizontal line at waste_factor = 1.
#
#   Double-column: 180 mm → 7.09 in
# -----------------------------------------------------------------------------

def fig2_waste_factor(cmp_df, adapt_df, outdir, fmt):
    fixed_Ns = sorted(cmp_df["N_fixed"].unique())
    obs_list  = obs_order(adapt_df)
    n_obs     = len(obs_list)

    fig, axes = plt.subplots(1, len(fixed_Ns),
                             figsize=(7.09, 1.8),
                             sharey=True)

    for ax, N_fix in zip(axes, fixed_Ns):
        sub = cmp_df[cmp_df["N_fixed"] == N_fix].set_index("observable")
        sub = sub.reindex(obs_list)

        wf     = sub["waste_factor"].values
        colors = []
        for w in wf:
            if   w > 1.05: colors.append(C_WASTE)
            elif w < 0.95: colors.append(C_UNDER)
            else:          colors.append(C_MATCHED)

        x = np.arange(n_obs)
        ax.bar(x, wf, color=colors, width=0.55, zorder=3)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=4)

        # Annotate bars with numeric value
        for xi, w in enumerate(wf):
            va  = "bottom" if w >= 1 else "top"
            off = 0.04 if w >= 1 else -0.04
            ax.text(xi, w + off, f"{w:.1f}×",
                    ha="center", va=va, fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels([OBS_LABEL.get(o, o) for o in obs_list])
        ax.set_title(FIXED_N_LABELS[N_fix], pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=C_GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, n_obs - 0.4)

    axes[0].set_ylabel(r"Waste factor $N_{\mathrm{fixed}}/N_{\mathrm{adaptive}}$")

    # -- Shared legend -----------------------------------------------------------
    legend_patches = [
        mpatches.Patch(color=C_WASTE,   label="Over-sampled  (> 1.05×)"),
        mpatches.Patch(color=C_MATCHED, label="Matched  (0.95–1.05×)"),
        mpatches.Patch(color=C_UNDER,   label="Under-sampled  (< 0.95×)"),
    ]
    fig.legend(handles=legend_patches,
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04),
               framealpha=0.9, edgecolor="0.8", fontsize=6.5)

    fig.suptitle(
        r"Waste factor for each fixed-$N$ choice ($\varepsilon = 1\%$)",
        y=1.10, fontsize=8)
    fig.tight_layout()
    save(fig, outdir, "fig2_waste_factor", fmt)


# -----------------------------------------------------------------------------
# Figure 3 — Achieved rel-SEM heat-map
#
#   Rows    = observables
#   Columns = fixed-N values + adaptive column
#   Cell colour: green if target met, red if not.
#   Cell text: achieved rel-SEM in percent.
#
#   Single-column: 86 mm → 3.39 in (tall enough for 3 rows)
# -----------------------------------------------------------------------------

def fig3_relsem_heatmap(cmp_df, adapt_df, outdir, fmt):
    fixed_Ns = sorted(cmp_df["N_fixed"].unique())
    obs_list  = obs_order(adapt_df)

    # Build matrix: rows = obs, cols = fixed_N values + adaptive
    adapt_med = adaptive_medians(adapt_df)

    cols_fixed = fixed_Ns
    col_labels = [FIXED_N_LABELS[N] for N in cols_fixed] + ["Adaptive"]
    n_cols = len(col_labels)
    n_rows = len(obs_list)

    relsem_mat = np.zeros((n_rows, n_cols))
    met_mat    = np.zeros((n_rows, n_cols), dtype=bool)

    EPS = 0.01   # 1% target

    for ri, obs in enumerate(obs_list):
        # Fixed-N columns
        for ci, N_fix in enumerate(cols_fixed):
            row = cmp_df[(cmp_df["observable"] == obs) &
                         (cmp_df["N_fixed"] == N_fix)]
            if len(row):
                relsem_mat[ri, ci] = row.iloc[0]["fixed_rel_sem"]
                met_mat[ri, ci]    = bool(row.iloc[0]["fixed_target_met"])
        # Adaptive column
        _, rs, _ = adapt_med[obs]
        relsem_mat[ri, n_cols - 1] = rs
        met_mat[ri, n_cols - 1]    = rs <= EPS * 1.01

    fig, ax = plt.subplots(figsize=(3.39, 0.6 + 0.55 * n_rows))

    for ri in range(n_rows):
        for ci in range(n_cols):
            rs   = relsem_mat[ri, ci]
            met  = met_mat[ri, ci]
            bg   = "#c7e9c0" if met else "#fcbba1"   # light green / light red
            rect = mpatches.FancyBboxPatch(
                (ci - 0.45, ri - 0.38), 0.90, 0.76,
                boxstyle="round,pad=0.02",
                linewidth=0.5,
                edgecolor="0.7",
                facecolor=bg,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(ci, ri, f"{rs*100:.2f}%",
                    ha="center", va="center", fontsize=7,
                    color="black", zorder=3)

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([OBS_LABEL.get(o, o) for o in obs_list])
    ax.set_title(r"Achieved rel-SEM  ($\varepsilon_{\mathrm{target}} = 1\%$)", pad=4)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    # Legend patches
    met_patch  = mpatches.Patch(color="#c7e9c0", label="Target met")
    miss_patch = mpatches.Patch(color="#fcbba1", label="Target missed")
    ax.legend(handles=[met_patch, miss_patch],
              loc="upper left", bbox_to_anchor=(0, -0.22),
              ncol=2, frameon=False, fontsize=6.5)

    fig.tight_layout()
    save(fig, outdir, "fig3_relsem_heatmap", fmt)


# -----------------------------------------------------------------------------
# Figure 4 — Runtime comparison
#
#   For each observable: scatter of (N, runtime) for fixed-N runs,
#   plus a horizontal band for the adaptive runtime (median ± IQR from
#   3 replicas).  Log x-axis to show the full N range.
#
#   Three-panel row, one panel per observable.
# -----------------------------------------------------------------------------

def fig4_runtime(fixed_df, adapt_df, outdir, fmt):
    obs_list = obs_order(adapt_df)
    n_obs    = len(obs_list)

    fig, axes = plt.subplots(1, n_obs, figsize=(7.09, 2.0), sharey=False)

    for ax, obs in zip(axes, obs_list):
        sub = fixed_df[fixed_df["observable"] == obs].sort_values("N_fixed")
        N_vals = sub["N_fixed"].values
        rt_vals = sub["runtime_s"].values

        # Adaptive band
        grp = adapt_df[adapt_df["observable"] == obs]["runtime_s"]
        rt_med  = grp.median()
        rt_lo   = grp.min()
        rt_hi   = grp.max()

        # Fixed-N scatter + line
        ax.plot(N_vals, rt_vals, "o-",
                color=C_REFLINE, ms=4, lw=0.8, label="Fixed $N$", zorder=3)

        # Adaptive band
        ax.axhspan(rt_lo, rt_hi, color=C_ADAPTIVE, alpha=0.20, zorder=1)
        ax.axhline(rt_med, color=C_ADAPTIVE, linewidth=1.0,
                   linestyle="--", label="Adaptive (median)", zorder=2)

        ax.set_xscale("log")
        ax.set_xlabel("$N$")
        ax.set_title(OBS_LABEL.get(obs, obs), pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=C_GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1,
                                               numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())

    axes[0].set_ylabel("Wall-clock time (s)")

    # Shared legend from first panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06),
               framealpha=0.9, edgecolor="0.8", fontsize=6.5)

    fig.suptitle("Runtime: adaptive stopping vs fixed-$N$ ensemble", y=1.10, fontsize=8)
    fig.tight_layout()
    save(fig, outdir, "fig4_runtime", fmt)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plotting for bench3_fixed_vs_adaptive results.")
    p.add_argument("--datadir", default=".",
                   help="Directory containing bench3_*.csv files (default: .)")
    p.add_argument("--outdir", default=".",
                   help="Directory for output figures (default: .)")
    p.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg", "eps"],
                   help="Output format (default: pdf)")
    p.add_argument("--dpi", type=int, default=300,
                   help="DPI for raster formats (default: 300)")
    p.add_argument("--figs", default="1234",
                   help="Which figures to produce, e.g. --figs 12 (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    mpl.rcParams["savefig.dpi"] = args.dpi

    print(f"Loading CSVs from: {os.path.abspath(args.datadir)}")
    adapt_df, fixed_df, cmp_df = load_csvs(args.datadir)

    # -- Validate columns -----------------------------------------------------
    required = {
        "bench3_adaptive.csv":   {"observable", "N_adaptive", "runtime_s",
                                  "mean", "sem", "rel_sem"},
        "bench3_fixed.csv":      {"observable", "N_fixed", "runtime_s",
                                  "mean", "sem", "rel_sem", "target_met"},
        "bench3_comparison.csv": {"observable", "N_adaptive_med", "N_fixed",
                                  "fixed_rel_sem", "fixed_target_met",
                                  "waste_factor"},
    }
    col_map = {
        "bench3_adaptive.csv":   adapt_df,
        "bench3_fixed.csv":      fixed_df,
        "bench3_comparison.csv": cmp_df,
    }
    for fname, cols in required.items():
        df   = col_map[fname]
        miss = cols - set(df.columns)
        if miss:
            sys.exit(f"ERROR: {fname} is missing columns: {miss}")

    print(f"  Observables found: {sorted(adapt_df['observable'].unique())}")
    print(f"  Fixed-N values:    {sorted(cmp_df['N_fixed'].unique().tolist())}")
    print()

    figs = args.figs
    if "1" in figs:
        print("Figure 1: Horizontal bar chart")
        fig1_bar_N_adaptive(adapt_df, args.outdir, args.fmt)

    if "2" in figs:
        print("Figure 2: Waste-factor panel")
        fig2_waste_factor(cmp_df, adapt_df, args.outdir, args.fmt)

    if "3" in figs:
        print("Figure 3: Rel-SEM heat-map")
        fig3_relsem_heatmap(cmp_df, adapt_df, args.outdir, args.fmt)

    if "4" in figs:
        print("Figure 4: Runtime comparison")
        fig4_runtime(fixed_df, adapt_df, args.outdir, args.fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
