"""
Convert LiQuID's own benchmark output into the liquid_results.jsonl schema
used by make_qutip_comparison.py (see bench_common.py for the schema).

Only two of LiQuID's benchmark CSVs cover the same (system, observable, eps)
grid as the QuTiP side of this comparison:

  - bench_adaptive_stopping.csv (Table 4: adaptive stopping efficiency)
        -> pure decay, Eq. 11, observable sigma_z
        column used: N_adaptive
  - bench2_main.csv (Table 5: observable-dependent trajectory allocation)
        -> driven qubit, Eq. 2, observables Pe / sigma_z ("sz") / Pg
        column used: N_adaptive_med

(LiQuID's other benchmark CSVs -- bench1a_sem_scaling.csv, bench2_variance.csv,
bench3_*.csv -- back Tables 1-3 and 6 in the paper, which aren't part of this
particular QuTiP-vs-LiQuID N_adapt comparison, so they aren't consumed here.)

The physical systems in those two CSVs are verified to match bench_common.py's
`pure_decay`/sigma_z and `driven`/{Pe,sigma_z,Pg} definitions (same H, L,
Omega, gamma, T -- see the .cpp headers), so eps values line up directly with
what qutip_benchmark.py sweeps.

Usage:
    python liquid_csv_to_jsonl.py \
        --adaptive-stopping-csv path/to/bench_adaptive_stopping.csv \
        --variance-csv path/to/bench2_main.csv \
        --out results/liquid_results.jsonl

Or point at the LiQuID benchmarks/results/ directory directly:
    python liquid_csv_to_jsonl.py --liquid-results-dir /path/to/LiQuID/benchmarks/results
"""
import argparse
import csv
import json
import os

# bench2_main.csv's "observable" column uses LiQuID's short codes; map them
# to the same names bench_common.py uses (matches DISPLAY_LABEL keys there).
OBSERVABLE_MAP = {"Pe": "Pe", "sz": "sigma_z", "Pg": "Pg"}


def convert_adaptive_stopping(path):
    """bench_adaptive_stopping.csv -> pure_decay/sigma_z records."""
    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({
                "system": "pure_decay",
                "observable": "sigma_z",
                "eps": float(row["target_rel_sem"]),
                "N_liquid": int(float(row["N_adaptive"])),
                "N_optimal": int(float(row["N_optimal"])),
                "efficiency": float(row["efficiency"]),
                "achieved_rel_sem": float(row["achieved_rel_sem"]),
                "wall_s": float(row["wall_s"]),
                "source": os.path.basename(path),
            })
    return records


def convert_variance_dependence(path):
    """bench2_main.csv -> driven/{Pe,sigma_z,Pg} records."""
    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            obs = OBSERVABLE_MAP.get(row["observable"])
            if obs is None:
                continue  # e.g. sigma_x rows, which the QuTiP side doesn't cover
            records.append({
                "system": "driven",
                "observable": obs,
                "eps": float(row["target_rel_sem"]),
                "N_liquid": int(float(row["N_adaptive_med"])),
                "N_liquid_lo": int(float(row["N_adaptive_lo"])),
                "N_liquid_hi": int(float(row["N_adaptive_hi"])),
                "N_optimal": int(float(row["N_optimal"])),
                "efficiency": float(row["efficiency"]),
                "achieved_rel_sem": float(row["achieved_rel_sem"]),
                "wall_s": float(row["wall_s"]),
                "source": os.path.basename(path),
            })
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--liquid-results-dir", default=None,
                    help="LiQuID benchmarks/results/ dir; used to fill in "
                         "--adaptive-stopping-csv / --variance-csv defaults")
    p.add_argument("--adaptive-stopping-csv", default=None,
                    help="path to bench_adaptive_stopping.csv (Table 4)")
    p.add_argument("--variance-csv", default=None,
                    help="path to bench2_main.csv (Table 5)")
    p.add_argument("--out", default="results/liquid_results.jsonl")
    args = p.parse_args()

    adaptive_csv = args.adaptive_stopping_csv
    variance_csv = args.variance_csv
    if args.liquid_results_dir:
        adaptive_csv = adaptive_csv or os.path.join(args.liquid_results_dir,
                                                      "bench_adaptive_stopping.csv")
        variance_csv = variance_csv or os.path.join(args.liquid_results_dir,
                                                      "bench2_main.csv")

    if not adaptive_csv or not os.path.exists(adaptive_csv):
        raise SystemExit(f"Cannot find bench_adaptive_stopping.csv "
                          f"(looked at: {adaptive_csv}). Pass --adaptive-stopping-csv "
                          f"or --liquid-results-dir.")
    if not variance_csv or not os.path.exists(variance_csv):
        raise SystemExit(f"Cannot find bench2_main.csv "
                          f"(looked at: {variance_csv}). Pass --variance-csv "
                          f"or --liquid-results-dir.")

    records = convert_adaptive_stopping(adaptive_csv) + convert_variance_dependence(variance_csv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} LiQuID records to {args.out} "
          f"({adaptive_csv}, {variance_csv})")


if __name__ == "__main__":
    main()
