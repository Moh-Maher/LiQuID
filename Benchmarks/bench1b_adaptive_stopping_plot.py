#!/usr/bin/env python3
"""
bench1b_adaptive_stopping_plot.py
------------------------------------------------------------------
Produces Figure 1 for the LiQuID adaptive-stopping validation.

Run:
    python bench1b_adaptive_stopping_plot.py                   # reads CSV in current directory
    python bench1b_adaptive_stopping_plot.py --data results/   # read CSV from another directory
    python bench1b_adaptive_stopping_plot.py --out figs/       # write figures to another directory
    python bench1b_adaptive_stopping_plot.py --data results/ --out figs/
    python bench1b_adaptive_stopping_plot.py --show            # display interactively instead of saving

Requires: numpy, matplotlib, pandas
Output  : bench_adaptive_stopping_fig1.pdf
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # headless by default; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


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
        print(f"WARNING: {path} is empty.")
        return pd.DataFrame()


def save_or_show(fig, path: str, show: bool):
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 1 — scatter + efficiency bar chart
# -----------------------------------------------------------------------------

def plot_adaptive_stopping(df: pd.DataFrame, out_dir: str, show: bool):
    """
    Left panel  : N_adaptive vs N_optimal scatter with y = x diagonal.
    Right panel : efficiency bar chart per target ε.
    """
    if df.empty:
        print("  Skipping figure: bench_adaptive_stopping.csv data is empty.")
        return

    N_opt   = df["N_optimal"].values
    N_adapt = df["N_adaptive"].values
    eff     = df["efficiency"].values
    eps     = df["target_rel_sem"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # -- left panel: scatter + y = x ------------------------------------------
    ax = axes[0]

    lo = min(N_opt.min(), N_adapt.min()) * 0.6
    hi = max(N_opt.max(), N_adapt.max()) * 1.6

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label=r"$y = x$ (ideal)")
    sc = ax.scatter(N_opt, N_adapt, c=np.log10(eps),
                    cmap="viridis_r", s=90, zorder=5, edgecolors="k", lw=0.6)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(\varepsilon)$", fontsize=10)

    # Annotate each point with ε label
    # NOTE: % kept outside the LaTeX math block to prevent parser crashes
    for i, e in enumerate(eps):
        label = rf"$\varepsilon={e * 100:.1f}$%"
        ax.annotate(label, (N_opt[i], N_adapt[i]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color="#333333")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\mathrm{optimal}}  =  \lceil \sigma^2 / (\varepsilon |\mu|)^2 \rceil$",
                  fontsize=11)
    ax.set_ylabel(r"$N_{\mathrm{adaptive}}$  (LiQuID, median of 3)", fontsize=11)
    ax.set_title(r"Adaptive stopping: $N_{\mathrm{adaptive}}$ vs $N_{\mathrm{optimal}}$",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # -- right panel: efficiency bar chart ------------------------------------
    ax2 = axes[1]
    bar_labels = [rf"{e * 100:.1f}%" for e in eps]
    colors = ["#2ecc71" if 0.9 <= e <= 1.1 else "#e74c3c" for e in eff]

    x_positions = np.arange(len(eps))
    bars = ax2.bar(x_positions, eff, color=colors, edgecolor="k", lw=0.6, width=0.55)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(bar_labels, fontsize=9)

    ax2.axhline(1.0, color="k",  lw=1.2, ls="--", label="Ideal (1.0)")
    ax2.axhspan(0.9, 1.1, color="green", alpha=0.10, label="[0.9, 1.1] band")
    ax2.set_ylim(0, max(eff.max() * 1.25, 1.3))
    ax2.set_xlabel(r"Target relative SEM  $\varepsilon$", fontsize=11)
    ax2.set_ylabel(r"Efficiency  $= N_{\mathrm{optimal}} / N_{\mathrm{adaptive}}$",
                   fontsize=11)
    ax2.set_title("Stopping efficiency per target", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    for bar, e in zip(bars, eff):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{e:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    save_or_show(fig, os.path.join(out_dir, "bench_adaptive_stopping_fig1.pdf"), show)


# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    if df.empty:
        print("  Summary unavailable: dataset is empty.")
        return

    print("\n  Adaptive Stopping Validation — Summary Table")
    print(f"  {'Target ε':>10}  {'N_optimal':>12}  {'N_adaptive':>12}  "
          f"{'Efficiency':>12}  {'Achieved ε':>12}  Status")
    print("  " + "─" * 70)
    for i in range(len(df)):
        row = df.iloc[i]
        ok  = "✓ PASS" if 0.9 <= float(row.efficiency) <= 1.1 else "✗ FAIL"
        print(f"  {float(row.target_rel_sem):>10.3f}  {int(row.N_optimal):>12d}  "
              f"{int(row.N_adaptive):>12d}  {float(row.efficiency):>12.4f}  "
              f"{float(row.achieved_rel_sem):>12.4f}  {ok}")
    print()
    n_pass = ((df.efficiency >= 0.9) & (df.efficiency <= 1.1)).sum()
    print(f"  Targets passing [0.9, 1.1]: {n_pass} / {len(df)}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", default=".",
        help="Directory containing bench_adaptive_stopping.csv (default: .)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for figure files (default: same as --data)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively instead of saving to disk",
    )
    args = parser.parse_args()

    data_dir = args.data
    out_dir  = args.out if args.out else data_dir

    if args.show:
        matplotlib.use("TkAgg")   # switch to interactive backend

    os.makedirs(out_dir, exist_ok=True)

    print("Loading CSV data …")
    df = load_csv(data_dir, "bench_adaptive_stopping.csv")
    if not df.empty:
        df.columns = df.columns.str.strip()   # defensive: remove accidental whitespace

    print("\nGenerating figures …")
    plot_adaptive_stopping(df, out_dir, args.show)

    print_summary(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
