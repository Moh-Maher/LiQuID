"""
Fast standalone rerun of just the driven-qubit (Eq. 2) cases, without paying
for the pure-decay sweep in qutip_benchmark.py. Useful while iterating on the
driven system specifically (e.g. chasing the Pg pathology).

Writes the same JSONL record schema as qutip_benchmark.py (see bench_common.py)
so results from either script can be merged and plotted together.
"""
import argparse

from bench_common import run_case, print_row, append_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps", type=float, nargs="+", default=[0.10, 0.05],
                    help="target relative precisions to sweep")
    p.add_argument("--ntraj-max", type=int, default=200_000,
                    help="trajectory cap passed to mcsolve (default: 200000)")
    p.add_argument("--seed", type=int, default=0, help="np.random.seed before each run")
    p.add_argument("--observables", nargs="+", default=["Pe", "Pg", "sigma_z"],
                    choices=["Pe", "Pg", "sigma_z"])
    p.add_argument("--out", default="results/qutip_results.jsonl",
                    help="JSONL file to append records to")
    p.add_argument("--no-write", action="store_true",
                    help="print only, do not write JSONL records")
    args = p.parse_args()

    records = []
    for observable in args.observables:
        for eps in args.eps:
            rec = run_case("driven", observable, eps, args.ntraj_max, seed=args.seed)
            print_row(rec)
            records.append(rec)
            if not args.no_write:
                append_jsonl([rec], args.out)  # write immediately, see bench_common note

    if not args.no_write:
        print(f"\nWrote records to {args.out}")


if __name__ == "__main__":
    main()
