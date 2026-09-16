"""
Head-to-head benchmark: QuTiP 5 mcsolve(target_tol=...) vs LiQuID's
observable-relative-SEM adaptive stopping criterion.

Reproduces:
  - Eq. 11 system (pure spontaneous decay, sigma_z, exact solution known)
  - Eq. 2 system  (resonantly driven two-level atom, Pe/Pg/sigma_z)

For each target relative precision eps, we time how many trajectories
QuTiP's jackknife-resampling target_tol stopping needs and the wall time,
to compare directly against the paper's Table 4 / Table 6 (Welford RelSEM).

Note: QuTiP's target_tol semantics are (atol, rtol) on the expectation
value's *jackknife-estimated error*, not literally RelSEM, so we request
rtol=eps, atol=0 to match "relative precision target" as closely as
possible, and report both what it converges to and how many trajectories
it took.

Every run is written as a JSON record to --out (default results/qutip_results.jsonl)
so that make_qutip_comparison.py can plot directly from measured data instead of
numbers copied by hand. Records are appended one at a time, immediately after each
case finishes, rather than batched until the whole eps/observable sweep completes --
Pg-like cases can run for a long time near ntraj_max, and an interrupted sweep should
not lose already-completed results along with the unfinished one.
"""
import argparse

from bench_common import run_case, print_row, append_jsonl


def bench_pure_decay(eps_list, ntraj_max, seed, out_path, do_append):
    print(f"\n=== Pure decay (Eq. 11) ===")
    records = []
    for eps in eps_list:
        rec = run_case("pure_decay", "sigma_z", eps, ntraj_max, seed=seed)
        print_row(rec)
        records.append(rec)
        if do_append:
            append_jsonl([rec], out_path)  # write immediately; see docstring above
    return records


def bench_driven_qubit(eps_list, ntraj_max, seed, out_path, do_append):
    print(f"\n=== Driven qubit (Eq. 2) ===")
    records = []
    for observable in ("Pe", "Pg", "sigma_z"):
        print(f"\n-- observable: {observable} --")
        for eps in eps_list:
            rec = run_case("driven", observable, eps, ntraj_max, seed=seed)
            print_row(rec)
            records.append(rec)
            if do_append:
                append_jsonl([rec], out_path)  # write immediately; see docstring above
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps", type=float, nargs="+", default=[0.10, 0.05, 0.02, 0.01],
                    help="target relative precisions to sweep (default: paper smoke-test grid)")
    p.add_argument("--ntraj-max", type=int, default=200_000,
                    help="trajectory cap passed to mcsolve (default: 200000)")
    p.add_argument("--seed", type=int, default=0, help="np.random.seed before each run")
    p.add_argument("--out", default="results/qutip_results.jsonl",
                    help="JSONL file to write records to")
    p.add_argument("--no-write", action="store_true",
                    help="print only, do not write JSONL records")
    p.add_argument("--only", choices=["pure_decay", "driven"], default=None,
                    help="run only one of the two systems")
    args = p.parse_args()

    do_append = not args.no_write

    if args.only in (None, "pure_decay"):
        bench_pure_decay(args.eps, args.ntraj_max, args.seed, args.out, do_append)
    if args.only in (None, "driven"):
        bench_driven_qubit(args.eps, args.ntraj_max, args.seed, args.out, do_append)

    if do_append:
        print(f"\nWrote records to {args.out}")


if __name__ == "__main__":
    main()
