#!/usr/bin/env python3
"""
bench4_higher_dim_variance_plot.py
-----------------------------------------------------------------------------
Publication-quality plots for LiQuID Benchmark 4: Higher-Dimensional (d=4)
Variance Dependence — two coupled dissipative qubits.

Reads four CSV files produced by bench4_higher_dim_variance:
    bench4_variance.csv   sigma^2, |mu|, sigma/|mu| per observable (ref run)
    bench4_main.csv       N_optimal / N_adaptive / efficiency per (obs, eps)
    bench4_ratio.csv      N_max/N_min across observables, per eps target
    bench4_fixed.csv      fixed-N comparison (small N vs large N, all obs)

Produces four figures:
    bench4_fig1_Nadaptive_bar.pdf     Required N per observable at eps=1%
    bench4_fig2_ratio_vs_eps.pdf      N_max/N_min ratio as eps tightens
    bench4_fig3_variance.pdf          sigma^2 and sigma/|mu| per observable
    bench4_fig4_fixed_comparison.pdf  Fixed-N failure / oversampling panel

Run:
    python bench4_higher_dim_variance_plot.py                 # CSVs in cwd
    python bench4_higher_dim_variance_plot.py --data results/ --out figs/
    python bench4_higher_dim_variance_plot.py --show
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Style — consistent with bench2_variance_dependence_plot.py
# -----------------------------------------------------------------------------

PALETTE = {
    "Pe1": "#4477AA",   # blue
    "Pe2": "#66CCEE",   # cyan
    "Pee": "#CCBB44",   # gold
    "sz1": "#EE6677",   # red
    "sz2": "#228833",   # green
}
LABEL_MAP = {
    "Pe1": r"$P_{e1}$",
    "Pe2": r"$P_{e2}$",
    "Pee": r"$P_{ee}$",
    "sz1": r"$\langle\sigma_z^{(1)}\rangle$",
    "sz2": r"$\langle\sigma_z^{(2)}\rangle$",
}
OBS_ORDER = ["Pe1", "Pe2", "Pee", "sz1", "sz2"]

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
        print(f"ERROR: {path} not found.  Run bench4_higher_dim_variance first "
              f"(stages: ref, sweep 0..4, finalize).")
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
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def get_color(obs: str) -> str:
    return PALETTE.get(obs, "#555555")


def get_label(obs: str) -> str:
    return LABEL_MAP.get(obs, f"Observable {obs}")


def ordered(obs_iterable):
    """Sort an iterable of observable keys by the canonical OBS_ORDER."""
    obs_set = set(obs_iterable)
    return [o for o in OBS_ORDER if o in obs_set] + \
           [o for o in obs_set if o not in OBS_ORDER]


# -----------------------------------------------------------------------------
# Plot 1 — Centerpiece bar chart: N_adaptive at eps = 1%
# -----------------------------------------------------------------------------

def plot1_Nadaptive_bar(df_main: pd.DataFrame, out_dir: str, show: bool):
    if df_main.empty:
        print("  Skipping Plot 1: bench4_main.csv is empty.")
        return

    TARGET_EPS = 0.01
    sub = df_main[np.isclose(df_main["target_rel_sem"], TARGET_EPS)].copy()
    if sub.empty:
        print(f"  Skipping Plot 1: no rows at target_rel_sem={TARGET_EPS}")
        return

    obs_list = ordered(sub["observable"])
    sub["obs_order"] = sub["observable"].map({k: i for i, k in enumerate(obs_list)})
    sub = sub.sort_values("obs_order", ascending=False)  # bottom = highest N

    fig, ax = plt.subplots(figsize=(7, 3.8))
    y_pos = np.arange(len(sub))
    colors = [get_color(o) for o in sub["observable"]]

    ax.barh(y_pos, sub["N_adaptive"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, height=0.55, zorder=3)

    x_max = sub["N_adaptive"].max()
    for y, row in zip(y_pos, sub.itertuples()):
        ax.text(row.N_adaptive * 1.12, y, f"{row.N_adaptive:,}",
                va="center", ha="left", fontsize=9.5, color="0.2")

    ax.set_xscale("log")
    ax.set_xlim(50, x_max * 3.0)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([get_label(o) for o in sub["observable"]], fontsize=12)
    ax.set_xlabel(r"Required trajectories  $N_{\rm adaptive}$")
    ax.set_title(
        f"Required trajectories for {int(TARGET_EPS*100)}% relative SEM\n"
        r"$d=4$: two coupled dissipative qubits — observable dependence persists",
        fontsize=11, pad=8)
    ax.xaxis.grid(True, which="both", linestyle=":", linewidth=0.5, color="0.75", zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.13, -0.03,
              r"Same system, same $\varepsilon$; only the observable changes.",
              fontsize=8.5, color="0.45")

    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench4_fig1_Nadaptive_bar.pdf"), show)


# -----------------------------------------------------------------------------
# Plot 2 — N_max/N_min ratio vs eps (the headline result)
# -----------------------------------------------------------------------------

def plot2_ratio_vs_eps(df_ratio: pd.DataFrame, out_dir: str, show: bool):
    if df_ratio.empty:
        print("  Skipping Plot 2: bench4_ratio.csv is empty.")
        return

    df = df_ratio.sort_values("target_rel_sem", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    x = df["target_rel_sem"].values * 100.0
    y = df["ratio"].values

    ax.plot(x, y, marker="o", color="#4477AA", lw=1.8, markersize=7, zorder=4)

    y_max = y.max()
    for i, (xi, yi, row) in enumerate(zip(x, y, df.itertuples())):
        # Put the label below the marker for the two highest points so it
        # never collides with the title; above for the rest.
        below = yi > 0.75 * y_max
        xytext = (8, -22) if below else (8, 6)
        va = "top" if below else "bottom"
        ax.annotate(
            f"{get_label(row.obs_max)} / {get_label(row.obs_min)}\n{yi:.1f}x",
            (xi, yi), textcoords="offset points", xytext=xytext,
            fontsize=8.5, color="0.25", va=va)

    ax.set_xscale("log")
    ax.invert_xaxis()   # tighter eps to the right
    ax.set_xlabel(r"Target relative SEM  $\varepsilon$  [%]")
    ax.set_ylabel(r"$N_{\max} / N_{\min}$ across the 5 observables")
    ax.set_title(
        "Observable-dependent convergence persists at $d=4$\n"
        r"Ratio between highest- and lowest-variance observable, vs.\ target $\varepsilon$",
        fontsize=11, pad=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="0.78")
    ax.set_ylim(0, y_max * 1.25)

    fig.text(
        0.12, -0.02,
        r"As $\varepsilon$ tightens past the $N_{\min}=100$ floor, the ratio approaches"
        r" the asymptotic $(\sigma/|\mu|)^2$ ratio between observables.",
        fontsize=8.2, color="0.4")

    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench4_fig2_ratio_vs_eps.pdf"), show)


# -----------------------------------------------------------------------------
# Plot 3 — Variance / relative-std bar chart (physics link)
# -----------------------------------------------------------------------------

def plot3_variance(df_var: pd.DataFrame, out_dir: str, show: bool):
    if df_var.empty:
        print("  Skipping Plot 3: bench4_variance.csv is empty.")
        return

    obs_list = ordered(df_var["observable"])
    df_var = df_var.copy()
    df_var["obs_order"] = df_var["observable"].map({k: i for i, k in enumerate(obs_list)})
    df_var = df_var.sort_values("obs_order")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    y_pos = np.arange(len(df_var))
    colors = [get_color(o) for o in df_var["observable"]]
    labels = [get_label(o) for o in df_var["observable"]]

    ax = axes[0]
    ax.barh(y_pos, df_var["variance"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, height=0.55, zorder=3)
    for y, row in zip(y_pos, df_var.itertuples()):
        ax.text(row.variance * 1.06, y, f"{row.variance:.4f}",
                va="center", ha="left", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(right=df_var["variance"].max() * 4.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(r"Variance  $\sigma^2$")
    ax.set_title(r"Variance per observable", fontsize=11)
    ax.xaxis.grid(True, which="both", ls=":", lw=0.5, color="0.75", zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.barh(y_pos, df_var["sigma_over_absmean"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, height=0.55, zorder=3)
    for y, row in zip(y_pos, df_var.itertuples()):
        ax.text(row.sigma_over_absmean * 1.06, y, f"{row.sigma_over_absmean:.3f}",
                va="center", ha="left", fontsize=9)
    ax.set_xlim(right=df_var["sigma_over_absmean"].max() * 1.35)
    ax.set_xlabel(r"Relative std dev  $\sigma / |\mu|$")
    ax.set_title(r"$\sigma/|\mu|$ drives $N_{\rm optimal}$ at fixed $\varepsilon$", fontsize=11)
    ax.xaxis.grid(True, which="both", ls=":", lw=0.5, color="0.75", zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.03,
              r"$\langle\sigma_z^{(2)}\rangle$ sits near its steady-state floor "
              r"$\Rightarrow$ small $\sigma/|\mu|$ $\Rightarrow$ far fewer trajectories needed.",
              ha="center", fontsize=9.5, color="0.35")

    fig.suptitle(
        r"Observable statistics — two coupled dissipative qubits ($d=4$, $N_{\rm ref}=20{,}000$)",
        fontsize=11, y=1.02)
    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench4_fig3_variance.pdf"), show)


# -----------------------------------------------------------------------------
# Plot 4 — Fixed-N comparison panel (small N fails / large N oversamples)
# -----------------------------------------------------------------------------

def plot4_fixed_comparison(df_fixed: pd.DataFrame, out_dir: str, show: bool):
    if df_fixed.empty:
        print("  Skipping Plot 4: bench4_fixed.csv is empty.")
        return

    obs_list = ordered(df_fixed["observable"])
    roles = ["small", "large"]
    Ns = {r: int(df_fixed[df_fixed["fixed_role"] == r]["N_fixed"].iloc[0]) for r in roles}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=False)

    EPS_TARGET = 0.01

    for ax, role in zip(axes, roles):
        sub = df_fixed[df_fixed["fixed_role"] == role].set_index("observable").reindex(obs_list)
        x = np.arange(len(obs_list))
        colors_bar = ["#4dac26" if m else "#d6604d" for m in sub["target_met"]]

        ax.bar(x, sub["rel_sem"] * 100.0, color=colors_bar, width=0.55,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(EPS_TARGET * 100.0, color="0.2", linestyle="--", lw=1.3, zorder=4)
        ax.text(0.98, 0.93, r"target $\varepsilon=1\%$", ha="right", va="top",
                fontsize=8, color="0.25", transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))

        y_top = sub["rel_sem"].max() * 100.0
        for xi, row in zip(x, sub.itertuples()):
            ax.text(xi, row.rel_sem * 100.0 * 1.08, f"{row.rel_sem*100:.2f}%",
                    ha="center", va="bottom", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))

        ax.set_xticks(x)
        ax.set_xticklabels([get_label(o) for o in obs_list], fontsize=11)
        ax.set_yscale("log")
        ax.set_ylim(top=max(y_top, EPS_TARGET * 100.0) * 2.2)
        ax.set_title(f"Fixed $N={Ns[role]:,}$"
                     + ("\n(matches lowest-variance obs.)" if role == "small"
                        else "\n(matches highest-variance obs.)"),
                     fontsize=10)
        ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, color="0.8")
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"Achieved relative SEM  [\%]")

    legend_patches = [
        mpatches.Patch(color="#4dac26", label="Target met"),
        mpatches.Patch(color="#d6604d", label="Target missed"),
    ]
    axes[0].legend(handles=legend_patches, loc="center left", fontsize=8.5,
                   framealpha=0.9, edgecolor="0.8")

    waste_max = df_fixed[df_fixed.fixed_role == "large"]["waste_factor"].max()
    fig.suptitle(
        "No single fixed $N$ works for every observable, even at $d=4$\n"
        r"Left: $N$ tuned to $\langle\sigma_z^{(2)}\rangle$ starves the others.  "
        r"Right: $N$ tuned to $\langle\sigma_z^{(1)}\rangle$ oversamples "
        r"$\langle\sigma_z^{(2)}\rangle$ by " + f"{waste_max:.0f}" + r"$\times$.",
        fontsize=10.5, y=1.03)

    fig.tight_layout()
    save_or_show(fig, os.path.join(out_dir, "bench4_fig4_fixed_comparison.pdf"), show)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=".",
        help="Directory containing the four CSV files (default: .)")
    parser.add_argument("--out", default=None,
        help="Output directory for PDF files (default: same as --data)")
    parser.add_argument("--show", action="store_true",
        help="Display figures interactively instead of saving PDF files")
    args = parser.parse_args()

    data_dir = args.data
    out_dir = args.out if args.out else data_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.show:
        matplotlib.use("TkAgg")

    print("Loading CSV data ...")
    df_main  = load_csv(data_dir, "bench4_main.csv")
    df_var   = load_csv(data_dir, "bench4_variance.csv")
    df_ratio = load_csv(data_dir, "bench4_ratio.csv")
    df_fixed = load_csv(data_dir, "bench4_fixed.csv")

    for df in [df_main, df_var, df_ratio, df_fixed]:
        if not df.empty:
            df.columns = df.columns.str.strip()

    print("\nGenerating figures ...")
    plot1_Nadaptive_bar(df_main, out_dir, args.show)
    plot2_ratio_vs_eps(df_ratio, out_dir, args.show)
    plot3_variance(df_var, out_dir, args.show)
    plot4_fixed_comparison(df_fixed, out_dir, args.show)

    # -- Quick sanity report --------------------------------------------------
    print("\n-- Benchmark 4 summary ---------------------------------------")
    if not df_ratio.empty:
        tightest = df_ratio.sort_values("target_rel_sem").iloc[0]
        print(f"\n  At tightest target (eps={tightest['target_rel_sem']*100:.1f}%): "
              f"N_max/N_min = {tightest['ratio']:.1f}x "
              f"({tightest['obs_max']} vs {tightest['obs_min']})")
    if not df_main.empty:
        effs = df_main["efficiency"].dropna()
        n_pass = ((effs >= 0.7) & (effs <= 1.3)).sum()
        print(f"  Efficiency in [0.7, 1.3]: {n_pass} / {len(effs)} rows")
    print("\nDone.")


if __name__ == "__main__":
    main()
