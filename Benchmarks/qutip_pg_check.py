"""
Diagnostic check on the driven-qubit Pg observable, which is the case that
tends to trip QuTiP's target_tol stopping (it either never satisfies the
tolerance and gets capped, or throws a RuntimeWarning from an ill-conditioned
target_ntraj computation when the estimated variance is near zero).

Unlike qutip_benchmark*.py this prints res.stats in full for inspection, and
defaults to a much smaller ntraj_max (5000) so the pathology shows up fast
rather than burning the full trajectory budget.
"""
import argparse
import warnings

from bench_common import run_case, print_row, append_jsonl

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps", type=float, nargs="+", default=[0.10],
                    help="target relative precision(s) to check")
    p.add_argument("--ntraj-max", type=int, default=5000,
                    help="trajectory cap (kept small by default to surface the "
                         "pathology quickly; use qutip_benchmark.py for the full sweep)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/qutip_results.jsonl",
                    help="JSONL file to append records to")
    p.add_argument("--no-write", action="store_true",
                    help="print only, do not write JSONL records")
    args = p.parse_args()

    records = []
    for eps in args.eps:
        rec, res = run_case("driven", "Pg", eps, args.ntraj_max,
                             seed=args.seed, return_result=True)
        print_row(rec)
        print(res.stats)
        records.append(rec)
        if not args.no_write:
            append_jsonl([rec], args.out)  # write immediately, see bench_common note

    if not args.no_write:
        print(f"\nWrote records to {args.out}")


if __name__ == "__main__":
    main()
