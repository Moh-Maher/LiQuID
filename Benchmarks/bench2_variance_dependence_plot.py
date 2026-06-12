#!/usr/bin/env python3
"""
bench2_plot.py
-----------------------------------------------------------------------------
Publication-quality plots for LiQuID Benchmark 2: Variance Dependence.

Reads three CSV files produced by bench2_variance_dependence:
    bench2_main.csv          — N_adaptive / N_optimal / efficiency per (obs, ε)
    bench2_convergence.csv   — RelSEM(N) history for all 4 observables ([not needed])
    bench2_variance.csv      — σ², |μ|, σ/|μ| per observable

Produces four figures:
    bench2_fig1_Nadaptive_bar.pdf   Plot 1 — centerpiece bar chart
    bench2_fig2_convergence.pdf     Plot 2 — convergence history overlay (not needed)
    bench2_fig3_variance.pdf        Plot 3 — variance bar chart (physics link)  (not needed)
    bench2_fig4_efficiency.pdf      Plot 4 — N_adaptive vs N_optimal scatter

Run:
    python bench2_plot.py                   # reads CSVs in current directory
    python bench2_plot.py --data /path/to   # read CSVs from another directory
    python bench2_plot.py --show            # display interactively instead of saving
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # headless by default; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

# Palette: colourblind-safe, works in print and on screen
PALETTE = {
    "Pe":         "#4477AA",   # blue
    "Pg":         "#66CCEE",   # cyan / light blue
    "sz":         "#EE6677",   # red
    "sx":         "#228833",   # green
    "jump_count": "#CCBB44",   # gold/yellow
}
LABEL_MAP = {
    "Pe":         r"Population $P_e$",
    "Pg":         r"Population $P_g$",
    "sz":         r"$\sigma_z$",
    "sx":         r"$\sigma_x$ (energy proxy)",
    "jump_count": r"Jump count",
}
# Added "Pg" near "Pe" as they typically share similar low-variance profiles
OBS_ORDER = ["Pe", "Pg", "sz", "sx", "jump_count"]   

# Matplotlib rcParams for a journal-style figure
plt.rcParams.update({
    "figure.dpi":          150,
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.titlesize":      12,
    "legend.fontsize":     10,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "axes.linewidth":      0.8,
    "lines.linewidth":     1.6,
    "savefig.bbox":        "tight",
    "savefig.dpi":         300,
})

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_csv(directory: str, name: str) -> pd.DataFrame:
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.  Run the C++ benchmark first.")
        sys.exit(1)
    
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"WARNING: {path} is empty. Skipping data processing for this file.")
        return pd.DataFrame()


def save_or_show(fig, path: str, show: bool):
    if show:
        plt.show()
    else:
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def get_color(obs: str) -> str:
    """Safely fetch color from PALETTE with a dark grey fallback."""
    return PALETTE.get(obs, "#555555")


def get_label(obs: str) -> str:
    """Safely fetch label from LABEL_MAP with a fallback."""
    return LABEL_MAP.get(obs, f"Observable {obs}")


# -----------------------------------------------------------------------------
# Plot 1 — Centerpiece bar chart: N_adaptive at ε = 1%
# -----------------------------------------------------------------------------

def plot1_Nadaptive_bar(df_main: pd.DataFrame,
                        out_dir: str, show: bool):
    """
    Horizontal bar chart showing required trajectory count for each observable
    at ε = 1%.  Error whiskers from the min/max of the 3 replicas.
    Annotated with N values.  Log x-axis so the ~order-of-magnitude spread
    is clearly visible.
    """
    if df_main.empty:
        print("  Skipping Plot 1: bench2_main.csv data is empty.")
        return

    TARGET_EPS = 0.01

    sub = df_main[np.isclose(df_main["target_rel_sem"], TARGET_EPS)].copy()
    if sub.empty:
        print(f"  Skipping Plot 1: No data found matching target_rel_sem = {TARGET_EPS}")
        return

    # Enforce display order; use a high fallback index if unknown
    sub["obs_order"] = sub["observable"].map(
        {k: i for i, k in enumerate(OBS_ORDER)}).fillna(99)
    sub = sub.sort_values("obs_order", ascending=False)   # bottom = highest N

    fig, ax = plt.subplots(figsize=(7, 3.8))

    y_pos  = np.arange(len(sub))
    colors = [get_color(o) for o in sub["observable"]]

    bars = ax.barh(
        y_pos,
        sub["N_adaptive_med"],
        xerr=[
            sub["N_adaptive_med"] - sub["N_adaptive_lo"],
            sub["N_adaptive_hi"] - sub["N_adaptive_med"],
        ],
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        error_kw=dict(elinewidth=1.0, ecolor="0.3", capsize=4, capthick=1.0),
        height=0.55,
        zorder=3,
    )

    # Annotate with the median N value
    x_max = sub["N_adaptive_hi"].max()
    for y, row in zip(y_pos, sub.itertuples()):
        n = row.N_adaptive_med
        ax.text(
            n * 1.15, y,
            f"{n:,}",
            va="center", ha="left",
            fontsize=9.5, color="0.2",
        )

    ax.set_xscale("log")
    ax.set_xlim(left=10)
    ax.set_ylim(-0.6, len(sub) - 0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([get_label(o) for o in sub["observable"]],
                       fontsize=11)

    ax.set_xlabel("Required trajectories  $N_{\\rm adaptive}$")
    ax.set_title(
        f"Required trajectories for {int(TARGET_EPS*100)}% relative SEM\n"
        r"Different observables require different ensemble sizes"
        "\n(driven qubit: $\Omega = 0.5, \gamma = 1,\; T = 20$)",
        fontsize=11, pad=8,
    )

    # Light vertical grid lines on log axis
    ax.xaxis.grid(True, which="both", linestyle=":", linewidth=0.5,
                  color="0.75", zorder=0)
    ax.set_axisbelow(True)

    # Caption note at bottom
    fig.text(
        0.13, -0.04,
        r"Error bars: min/max of 3 independent replicas.",
        fontsize=8.5, color="0.45",
    )

    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench2_fig1_Nadaptive_bar.pdf"), show)


# -----------------------------------------------------------------------------
# Plot 2 — Convergence histories overlay
# -----------------------------------------------------------------------------
'''
def plot2_convergence(df_conv: pd.DataFrame,
                      out_dir: str, show: bool):
    """
    For each observable we take the convergence history collected while that
    observable drove stopping.  We then overlay its own RelSEM curve so each
    line is "the best possible" view of that observable's convergence.
    A horizontal dashed line marks the 1% target.
    """
    if df_conv.empty:
        print("  Skipping Plot 2: bench2_convergence.csv data is empty.")
        return

    TARGET_EPS = 0.01

    fig, ax = plt.subplots(figsize=(7, 4.5))

    N_max_plotted = 0
    
    # Check what observables are dynamically present in case OBS_ORDER misses one
    unique_obs = set(df_conv["obs_name"].unique()).union(OBS_ORDER)

    for obs in unique_obs:
        # rows where this observable was BOTH driving and being plotted
        mask = (df_conv["driving_obs"] == obs) & (df_conv["obs_name"] == obs)
        sub  = df_conv[mask].sort_values("N")
        if sub.empty:
            continue

        ax.plot(
            sub["N"], sub["rel_sem"] * 100.0,
            color=get_color(obs),
            label=get_label(obs),
            lw=1.8,
            zorder=4,
        )

        # Mark the stopping point (last point of that run)
        stop = sub.iloc[-1]
        ax.scatter(
            stop["N"], stop["rel_sem"] * 100.0,
            color=get_color(obs),
            s=50, zorder=5, marker="o",
            edgecolors="white", linewidths=0.6,
        )

        N_max_plotted = max(N_max_plotted, sub["N"].max())

    if N_max_plotted == 0:
        print("  Skipping Plot 2: No valid observable convergence sequences found inside data.")
        plt.close(fig)
        return

    # Target line
    ax.axhline(
        TARGET_EPS * 100.0, color="0.3",
        linestyle="--", lw=1.4, zorder=3,
        label=f"Target $\\epsilon = {int(TARGET_EPS*100)}\\%$",
    )

    # Add N^{-1/2} reference slope anchored at the primary curve's first point
    # Try Pe first, fallback to ground state Pg, then first available
    ref_obs = "Pe" if "Pe" in df_conv["obs_name"].values else ("Pg" if "Pg" in df_conv["obs_name"].values else df_conv["obs_name"].iloc[0])
    sub_ref = df_conv[
        (df_conv["driving_obs"] == ref_obs) & (df_conv["obs_name"] == ref_obs)
    ].sort_values("N")
    
    if not sub_ref.empty:
        n0  = sub_ref["N"].iloc[0]
        s0  = sub_ref["rel_sem"].iloc[0] * 100.0
        N_r = np.logspace(np.log10(n0), np.log10(N_max_plotted), 80)
        ax.plot(
            N_r, s0 * np.sqrt(n0 / N_r),
            color="0.6", lw=1.0, ls=(0, (4, 3)),
            label=r"$N^{-1/2}$ reference", zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of trajectories  $N$")
    ax.set_ylabel(r"Relative SEM  [\%]")
    ax.set_title(
        r"Convergence of RelSEM$(N)$ per observable"
        "\nAll run on the same system; target = 1%",
        fontsize=11, pad=8,
    )

    # Y-axis: percent formatter
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g}%"))

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="0.78")

    # Annotate with the key message
    ax.text(
        0.02, 0.05,
        "Population converges first;\nJump count converges last.",
        transform=ax.transAxes,
        fontsize=8.5, color="0.35",
        verticalalignment="bottom",
    )

    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench2_fig2_convergence.pdf"), show)

'''
# -----------------------------------------------------------------------------
# Plot 3 — Variance bar chart (physics connection)
# -----------------------------------------------------------------------------

def plot3_variance(df_var: pd.DataFrame,
                   out_dir: str, show: bool):
    """
    Two-panel figure:
      Left  — σ² (population variance) per observable
      Right — σ/|μ| (relative standard deviation), which directly scales
              N_optimal at fixed ε.
    Both panels share the same colour scheme as Plot 1.
    """
    if df_var.empty:
        print("  Skipping Plot 3: bench2_variance.csv data is empty.")
        return

    df_var = df_var.copy()
    df_var["obs_order"] = df_var["observable"].map(
        {k: i for i, k in enumerate(OBS_ORDER)}).fillna(99)
    df_var = df_var.sort_values("obs_order")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    y_pos  = np.arange(len(df_var))
    colors = [get_color(o) for o in df_var["observable"]]
    labels = [get_label(o) for o in df_var["observable"]]

    # -- Left: σ² --------------------------------------------------------
    ax = axes[0]
    bars = ax.barh(
        y_pos, df_var["variance"],
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5,
        height=0.55, zorder=3,
    )
    for y, row in zip(y_pos, df_var.itertuples()):
        ax.text(row.variance * 1.06, y,
                f"{row.variance:.3f}",
                va="center", ha="left", fontsize=9)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r"Population variance  $\sigma^2$")
    ax.set_title(r"Variance per observable", fontsize=11)
    ax.xaxis.grid(True, which="both", ls=":", lw=0.5, color="0.75", zorder=0)
    ax.set_axisbelow(True)

    # -- Right: σ/|μ| -------------------------------------------------------
    ax = axes[1]
    # For σ_x (mean ≈ 0) σ/|μ| diverges — cap it for display
    snr_vals = df_var["sigma_over_absmean"].clip(upper=50)
    bars2 = ax.barh(
        y_pos, snr_vals,
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5,
        height=0.55, zorder=3,
    )
    for y, (row, val) in zip(y_pos,
                              zip(df_var.itertuples(), snr_vals)):
        real = row.sigma_over_absmean
        txt  = f"{real:.2f}" if real < 40 else r"$\gg1$"
        ax.text(val * 1.06, y, txt,
                va="center", ha="left", fontsize=9)
    ax.set_xlabel(r"Relative std dev  $\sigma / |\mu|$")
    ax.set_title(r"$\sigma/|\mu|$ drives $N_{\rm optimal}$ at fixed $\varepsilon$",
                 fontsize=11)
    ax.xaxis.grid(True, which="both", ls=":", lw=0.5, color="0.75", zorder=0)
    ax.set_axisbelow(True)

    # Shared annotation
    fig.text(
        0.5, -0.03,
        r"Large $\sigma^2$ or large $\sigma/|\mu|$ $\;\Rightarrow\;$ more trajectories needed"
        r" at the same target $\varepsilon$.",
        ha="center", fontsize=9.5, color="0.35",
    )

    fig.suptitle(
        "Observable statistics from reference run  "
        r"($N_{\rm ref} = 10^6$, driven two-level atom)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench2_fig3_variance.pdf"), show)


# -----------------------------------------------------------------------------
# Plot 4 — N_adaptive vs N_optimal scatter (efficiency)
# -----------------------------------------------------------------------------

def plot4_efficiency(df_main: pd.DataFrame,
                     out_dir: str, show: bool):
    """
    Scatter of N_adaptive (y) vs N_optimal (x) across all observables and all
    ε values.  Perfect efficiency = the diagonal.  ±20% band shaded.
    Each observable uses a distinct marker and colour.
    """
    if df_main.empty:
        print("  Skipping Plot 4: bench2_main.csv data is empty.")
        return

    MARKERS = {"Pe": "o", "Pg": "v", "sz": "s", "sx": "^", "jump_count": "D"}

    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    # Remove rows with |μ| ≈ 0 (skipped in C++ output)
    df = df_main[df_main["N_optimal"] > 0].copy()

    if df.empty:
        print("  Skipping Plot 4: No rows found where N_optimal > 0.")
        plt.close(fig)
        return

    all_N = np.concatenate([df["N_optimal"].values, df["N_adaptive_med"].values])
    lo, hi = all_N.min() * 0.5, all_N.max() * 2.0

    # Perfect efficiency diagonal and ±20% band
    x_diag = np.array([lo, hi])
    ax.fill_between(x_diag, x_diag * 0.8, x_diag * 1.2,
                    color="0.88", zorder=1)
    ax.plot(x_diag, x_diag,
            color="0.35", lw=1.2, ls="--",
            label="Perfect efficiency", zorder=2)

    unique_main_obs = set(df["observable"].unique()).union(OBS_ORDER)

    for obs in unique_main_obs:
        sub = df[df["observable"] == obs]
        if sub.empty:
            continue
        ax.errorbar(
            sub["N_optimal"],
            sub["N_adaptive_med"],
            yerr=[
                sub["N_adaptive_med"] - sub["N_adaptive_lo"],
                sub["N_adaptive_hi"] - sub["N_adaptive_med"],
            ],
            fmt=MARKERS.get(obs, "o"),  # Circle fallback for unknown markers
            color=get_color(obs),
            markersize=7,
            linewidth=1.0,
            elinewidth=0.9,
            capsize=3,
            label=get_label(obs),
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\rm optimal} = \lceil\,\sigma^2\,/\,(\varepsilon|\mu|)^2\rceil$")
    ax.set_ylabel(r"$N_{\rm adaptive}$  (median of 3 replicas)")
    ax.set_title(
        r"Adaptive stopping vs.\ theoretical optimum" "\n"
        r"Shaded band: $\pm 20\%$ of perfect efficiency",
        fontsize=11, pad=8,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="0.78")

    # Efficiency annotation
    effs = df["efficiency"].dropna().values
    if len(effs):
        med_eff = np.median(effs)
        ax.text(
            0.97, 0.04,
            f"Median efficiency: {med_eff:.2f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, color="0.25",
        )

    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench2_fig4_efficiency.pdf"), show)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data", default=".",
        help="Directory containing the three CSV files (default: .)")
    parser.add_argument(
        "--out", default=None,
        help="Output directory for PDF files (default: same as --data)")
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively instead of saving PDF files")
    args = parser.parse_args()

    data_dir = args.data
    out_dir  = args.out if args.out else data_dir

    if args.show:
        matplotlib.use("TkAgg")   # switch to interactive backend

    print("Loading CSV data …")
    df_main = load_csv(data_dir, "bench2_main.csv")
    #df_conv = load_csv(data_dir, "bench2_convergence.csv")
    df_var  = load_csv(data_dir, "bench2_variance.csv")

    # Strip any accidental whitespace in column names
    for df in [df_main, df_var]:
        if not df.empty:
            df.columns = df.columns.str.strip()

    print("\nGenerating figures …")

    plot1_Nadaptive_bar(df_main, out_dir, args.show)
    #plot2_convergence  (df_conv, out_dir, args.show)
    #plot3_variance     (df_var,  out_dir, args.show)
    plot4_efficiency   (df_main, out_dir, args.show)

    # -- Quick sanity report -----------------------------------------------
    print("\n-- Benchmark 2 summary ---------------------------------------")

    if not df_main.empty:
        TARGET_EPS = 0.01
        sub1 = df_main[np.isclose(df_main["target_rel_sem"], TARGET_EPS)]
        if not sub1.empty:
            print(f"\n  Required trajectories at ε = {int(TARGET_EPS*100)}%:")
            print(f"  {'Observable':<28} {'N_adaptive':>12}  {'N_optimal':>12}  {'Efficiency':>10}")
            print("  " + "─" * 68)
            
            unique_summary_obs = [o for o in OBS_ORDER if o in sub1["observable"].values]
            # Capture any remaining tracked observables not explicit in order
            for o in sub1["observable"].unique():
                if o not in unique_summary_obs:
                    unique_summary_obs.append(o)

            for obs in unique_summary_obs:
                row = sub1[sub1["observable"] == obs]
                if row.empty:
                    continue
                r = row.iloc[0]
                print(f"  {get_label(obs):<28} {int(r['N_adaptive_med']):>12,}  "
                      f"{int(r['N_optimal']):>12,}  {r['efficiency']:>10.3f}")
            
            if len(sub1["N_adaptive_med"]) > 1:
                ratio = sub1["N_adaptive_med"].max() / sub1["N_adaptive_med"].min()
                print(f"\n  Spread (max/min N): {ratio:.1f}×  — "
                      "more than an order of magnitude"
                      if ratio > 10 else f"{ratio:.1f}×")

        effs = df_main["efficiency"].dropna()
        if len(effs):
            n_pass = ((effs >= 0.8) & (effs <= 1.2)).sum()
            print(f"\n  Efficiency ∈ [0.8, 1.2]: {n_pass} / {len(effs)} rows")
    else:
        print("  Summary statistics unavailable: main dataset is empty.")
    print("\nDone.")


if __name__ == "__main__":
    main()
