"""
Benchmark IV: QuTiP target_tol vs LiQuID adaptive RelSEM stopping.

Unlike the original draft of this script, the (system, eps, N_qutip, outcome)
data is no longer copied by hand into a Python literal. It is read from:

  - a QuTiP results JSONL file, produced by qutip_benchmark.py /
    qutip_benchmark_driven.py / qutip_pg_check.py (see bench_common.py for
    the record schema), and
  - a LiQuID results JSONL file with records of the form
        {"system": ..., "observable": ..., "eps": ..., "N_liquid": ...}
    (see liquid_results.example.jsonl for the placeholder values taken from
    the manuscript's Tables 4/5, pending the actual LiQuID-generation
    scripts, which will produce this file directly once available).

Run via run_all.sh, or directly:
    python make_qutip_comparison.py \
        --qutip-results results/qutip_results.jsonl \
        --liquid-results results/liquid_results.jsonl \
        --out-prefix results/qutip_vs_liquid_comparison
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from bench_common import read_jsonl, label_for

EPS_TOL = 1e-9  # float-compare tolerance for matching eps values


def _key(system, observable, eps):
    return (system, observable, round(float(eps), 10))


def load_qutip_results(path):
    """One record per (system, observable, eps). When multiple records share a
    key (e.g. a main sweep and a smaller-ntraj_max diagnostic check for the
    same case), keep the one with the larger ntraj_max -- that is the more
    rigorous experiment and should not be silently clobbered by a cheaper
    diagnostic run just because it was appended later. Ties fall back to
    file order (latest wins)."""
    by_key = {}
    for rec in read_jsonl(path):
        key = _key(rec["system"], rec["observable"], rec["eps"])
        prev = by_key.get(key)
        if prev is None or rec.get("ntraj_max", 0) >= prev.get("ntraj_max", 0):
            by_key[key] = rec
    return by_key


def load_liquid_results(path):
    by_key = {}
    for rec in read_jsonl(path):
        by_key[_key(rec["system"], rec["observable"], rec["eps"])] = rec["N_liquid"]
    return by_key


def build_dataset(qutip_by_key, liquid_by_key):
    data = []
    missing = []
    for key, rec in qutip_by_key.items():
        system, observable, eps = key
        if key not in liquid_by_key:
            missing.append(key)
            continue
        data.append({
            "label": rec.get("label", label_for(system, observable)),
            "system": system, "observable": observable, "eps": eps,
            "N_qutip": rec["N"], "N_liquid": liquid_by_key[key],
            "outcome": rec["outcome"], "rel_err": rec.get("rel_err"),
            "ntraj_max": rec.get("ntraj_max"),
        })
    if missing:
        print("Skipping QuTiP records with no matching LiQuID data:", file=sys.stderr)
        for system, observable, eps in missing:
            print(f"  - {label_for(system, observable)} eps={eps:.1%}", file=sys.stderr)
    return data


STYLE = {
    "converged": dict(marker="o", color="#2E7D32", label="Both converge normally",
                       size=90, zorder=3),
    "false_converged": dict(marker="*", color="#C62828", label="QuTiP false-converged",
                             size=380, zorder=4),
    "not_converged": dict(marker="^", color="#616161",
                           label="QuTiP did not converge\n(capped at $N_{\\max}$)",
                           size=140, zorder=3),
}


def make_plot(data, out_prefix, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    all_n = [d["N_liquid"] for d in data] + [d["N_qutip"] for d in data]
    if all_n:
        # Consistent half-decade padding below/above the data range, regardless
        # of what decade the data itself sits in (the previous lo formula's
        # combination of floor(log10(.))/10 and max(1, .) changed behaviour
        # qualitatively depending on the magnitude of min(all_n) -- e.g. it
        # would silently clip small-N points like Nmin=100 out of view once
        # the eps grid is widened enough to push max(all_n) past ~1e5).
        log_lo = np.floor(np.log10(min(all_n))) - 0.3
        log_hi = np.ceil(np.log10(max(all_n))) + 0.1
        lo, hi = 10 ** log_lo, 10 ** log_hi
    else:
        lo, hi = 1, 1e5
    lims = [lo, hi]
    ax.plot(lims, lims, "--", color="black", linewidth=1.1, alpha=0.6,
            zorder=1, label="$N_{\\mathrm{QuTiP}} = N_{\\mathrm{LiQuID}}$")

    seen_labels = set()
    for d in data:
        s = STYLE[d["outcome"]]
        lbl = s["label"] if s["label"] not in seen_labels else None
        seen_labels.add(s["label"])
        ax.scatter(d["N_liquid"], d["N_qutip"], marker=s["marker"], s=s["size"],
                   facecolor=s["color"], edgecolor="black", linewidth=0.6,
                   zorder=s["zorder"], label=lbl)

    # Auto-generate tags for the interesting failure classes, driven entirely
    # by the loaded data. Earlier versions of this function placed full
    # multi-line text boxes with arrows at hardcoded data-coordinate offsets;
    # that worked for a small hand-picked test case but visibly collided once
    # given a real, denser sweep (3 false-convergence points + up to 6 capped
    # points overlapped badly). Compact tags anchored to each marker scale to
    # any number of points without new collision risk -- full numeric detail
    # for every tagged point belongs in Table 8, not duplicated on the plot.
    false_pts = [d for d in data if d["outcome"] == "false_converged"]
    not_conv_pts = [d for d in data if d["outcome"] == "not_converged"]

    false_pts_sorted = sorted(false_pts, key=lambda d: -(d["rel_err"] or 0))
    for i, d in enumerate(false_pts_sorted):
        ax.annotate(f"F{i+1}", xy=(d["N_liquid"], d["N_qutip"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color="#C62828")

    not_conv_sorted = sorted(not_conv_pts, key=lambda d: d["N_qutip"])
    for i, d in enumerate(not_conv_sorted):
        ax.annotate(f"C{i+1}", xy=(d["N_liquid"], d["N_qutip"]),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color="#616161")

    if false_pts_sorted or not_conv_sorted:
        note_parts = []
        if false_pts_sorted:
            note_parts.append(f"F1\u2013F{len(false_pts_sorted)}: false convergence")
        if not_conv_sorted:
            note_parts.append(f"C1\u2013C{len(not_conv_sorted)}: capped, did not converge")
        ax.text(0.02, 0.02, "  |  ".join(note_parts) + " (see Table 8 for values)",
                transform=ax.transAxes, fontsize=7.2, color="#333333",
                va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel(r"$N_{\mathrm{adapt}}$ (LiQuID, Welford RelSEM)")
    ax.set_ylabel(r"$N$ at stop (QuTiP, target$\_$tol jackknife)")
    ax.set_title("QuTiP target_tol vs LiQuID adaptive stopping" + title_suffix)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.2,
              framealpha=0.95, borderaxespad=0.)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=220)
    print(f"saved {out_prefix}.pdf and {out_prefix}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qutip-results", default="results/qutip_results.jsonl")
    p.add_argument("--liquid-results", default="results/liquid_results.jsonl")
    p.add_argument("--out-prefix", default="figs/qutip_vs_liquid_comparison")
    p.add_argument("--title-suffix", default="",
                    help="appended to the plot title, e.g. '\\n(draft — pending replication)'")
    args = p.parse_args()

    qutip_by_key = load_qutip_results(args.qutip_results)
    if not qutip_by_key:
        sys.exit(f"No QuTiP records found in {args.qutip_results}. "
                  f"Run qutip_benchmark.py first (see run_all.sh).")

    liquid_by_key = load_liquid_results(args.liquid_results)
    if not liquid_by_key:
        sys.exit(f"No LiQuID records found in {args.liquid_results}.")

    data = build_dataset(qutip_by_key, liquid_by_key)
    if not data:
        sys.exit("No (system, observable, eps) keys matched between the two result files.")

    make_plot(data, args.out_prefix, args.title_suffix)


if __name__ == "__main__":
    main()
