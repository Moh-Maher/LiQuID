"""
Shared definitions for the QuTiP-vs-LiQuID benchmark suite.

All three qutip_*.py driver scripts import from here so that the system
definitions (Eq. 11 pure decay, Eq. 2 driven qubit), the "ground truth"
values used to classify convergence, and the JSONL record schema stay in
exactly one place. make_qutip_comparison.py also imports from here (for
labels + the JSONL reader) so the plot never has its own copy of numbers
that could drift out of sync with what the benchmarks actually measured.

Record schema (one JSON object per line, written to a .jsonl file):
    {
      "label":      "Pure decay $\\sigma_z$",   # display label for legends/annotations
      "system":     "pure_decay" | "driven",
      "observable": "sigma_z" | "Pe" | "Pg",
      "eps":        0.10,                       # requested target_tol relative precision
      "N":          1178,                       # trajectories QuTiP actually used
      "ntraj_max":  200000,                     # cap passed to mcsolve
      "wall":       12.3,                       # wall-clock seconds
      "mu_hat":     0.63,                       # QuTiP's final expectation value estimate
      "exact":      0.6321,                     # reference value (analytic or mesolve)
      "abs_err":    0.0021,
      "rel_err":    0.0033,
      "outcome":    "converged" | "false_converged" | "not_converged",
      "seed":       0
    }

NOTE ON REPRODUCIBILITY (carried over from the original scripts): qt.mcsolve's
internal RNG is seeded here via np.random.seed() before each call, but QuTiP 5's
own trajectory RNG stream is not guaranteed to be pinned by that alone across
qutip versions. The exact (system, eps) pair that false-converges may not
reproduce run-to-run. Replicate (e.g. 3x, as done for LiQuID) before treating
any single run's classification as final.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import qutip as qt

# ---------------------------------------------------------------------------
# Display labels — the single source of truth for how (system, observable)
# pairs are named in print output, JSONL records, and plot legends.
# ---------------------------------------------------------------------------
DISPLAY_LABEL = {
    ("pure_decay", "sigma_z"): "Pure decay $\\sigma_z$",
    ("driven", "Pe"): "Driven $P_e$",
    ("driven", "Pg"): "Driven $P_g$",
    ("driven", "sigma_z"): "Driven $\\sigma_z$",
}


def label_for(system: str, observable: str) -> str:
    return DISPLAY_LABEL.get((system, observable), f"{system} {observable}")


# ---------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------
def _pure_decay_setup():
    """Eq. 11: H=0, L=sqrt(gamma) sigma_-, T=1."""
    gamma = 1.0
    H = 0 * qt.qeye(2)
    L = np.sqrt(gamma) * qt.sigmam()
    psi0 = qt.basis(2, 0)  # |up>
    tlist = [0, 1.0]
    ops = {"sigma_z": qt.sigmaz()}
    return H, psi0, tlist, [L], ops


def _driven_setup():
    """Eq. 2: H=Omega*sigma_x, L=sqrt(gamma) sigma_-, Omega=0.5, gamma=1, T=20."""
    Omega, gamma, T = 0.5, 1.0, 20.0
    H = Omega * qt.sigmax()
    L = np.sqrt(gamma) * qt.sigmam()
    psi0 = qt.basis(2, 0)
    tlist = [0, T]
    ops = {
        "Pe": qt.projection(2, 0, 0),
        "Pg": qt.projection(2, 1, 1),
        "sigma_z": qt.sigmaz(),
    }
    return H, psi0, tlist, [L], ops


SYSTEMS = {
    "pure_decay": _pure_decay_setup,
    "driven": _driven_setup,
}


def get_system(system: str, observable: str):
    """Return (H, psi0, tlist, c_ops, op) for one (system, observable) pair."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choices: {list(SYSTEMS)}")
    H, psi0, tlist, c_ops, ops = SYSTEMS[system]()
    if observable not in ops:
        raise ValueError(
            f"unknown observable {observable!r} for system {system!r}; "
            f"choices: {list(ops)}"
        )
    return H, psi0, tlist, c_ops, ops[observable]


# ---------------------------------------------------------------------------
# Ground truth for classifying convergence.
#
# pure_decay/sigma_z has a closed form. The driven system has no simple
# closed form for finite T, so we solve the (deterministic) master equation
# with mesolve, which is numerically exact up to ODE solver tolerance and
# carries no Monte-Carlo sampling noise -- a fair reference for judging
# whether mcsolve's target_tol run actually landed close to the truth.
# ---------------------------------------------------------------------------
_exact_cache: dict = {}


def compute_exact(system: str, observable: str) -> float:
    key = (system, observable)
    if key in _exact_cache:
        return _exact_cache[key]

    if system == "pure_decay" and observable == "sigma_z":
        gamma, T = 1.0, 1.0
        exact = 2 * np.exp(-gamma * T) - 1
    else:
        H, psi0, tlist, c_ops, op = get_system(system, observable)
        fine_tlist = np.linspace(tlist[0], tlist[-1], 2000)
        res = qt.mesolve(H, psi0, fine_tlist, c_ops=c_ops, e_ops=[op])
        exact = float(np.real(res.expect[0][-1]))

    _exact_cache[key] = exact
    return exact


# ---------------------------------------------------------------------------
# Running one (system, observable, eps) case and classifying the outcome
# ---------------------------------------------------------------------------
FALSE_CONVERGE_REL_ERR_FACTOR = 3.0  # rel_err beyond this multiple of eps -> suspect


@dataclass
class BenchRecord:
    label: str
    system: str
    observable: str
    eps: float
    N: int
    ntraj_max: int
    wall: float
    mu_hat: float
    exact: float
    abs_err: float
    rel_err: float
    outcome: str
    seed: Optional[int] = None


def classify(N: int, ntraj_max: int, rel_err: float, eps: float) -> str:
    if N >= ntraj_max:
        return "not_converged"
    if rel_err > FALSE_CONVERGE_REL_ERR_FACTOR * eps:
        return "false_converged"
    return "converged"


def run_case(system: str, observable: str, eps: float, ntraj_max: int,
             seed: Optional[int] = None, return_result: bool = False):
    H, psi0, tlist, c_ops, op = get_system(system, observable)
    exact = compute_exact(system, observable)

    if seed is not None:
        np.random.seed(seed)

    t0 = time.time()
    res = qt.mcsolve(
        H, psi0, tlist, c_ops=c_ops, e_ops=[op],
        ntraj=ntraj_max,
        target_tol=(0.0, eps),
        options={"map": "serial", "progress_bar": False},
    )
    wall = time.time() - t0

    mu_hat = float(np.real(res.expect[0][-1]))
    N = int(res.num_trajectories)
    abs_err = abs(mu_hat - exact)
    rel_err = abs_err / abs(exact) if exact != 0 else float("inf")
    outcome = classify(N, ntraj_max, rel_err, eps)

    record = BenchRecord(
        label=label_for(system, observable),
        system=system, observable=observable, eps=eps,
        N=N, ntraj_max=ntraj_max, wall=wall,
        mu_hat=mu_hat, exact=exact, abs_err=abs_err, rel_err=rel_err,
        outcome=outcome, seed=seed,
    )
    if return_result:
        return record, res
    return record


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------
def append_jsonl(records, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            d = asdict(r) if isinstance(r, BenchRecord) else r
            f.write(json.dumps(d) + "\n")


def read_jsonl(path: str) -> list:
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_row(r: "BenchRecord | dict") -> None:
    d = asdict(r) if isinstance(r, BenchRecord) else r
    flag = {"converged": "", "false_converged": "  <-- FALSE CONVERGENCE",
            "not_converged": "  <-- CAPPED, DID NOT CONVERGE"}[d["outcome"]]
    print(f"{d['label']:>22} eps={d['eps']:6.1%}  N={d['N']:>9}  "
          f"wall={d['wall']:7.2f}s  mu_hat={d['mu_hat']:10.5f}  "
          f"exact={d['exact']:10.5f}  rel_err={d['rel_err']:8.2%}{flag}")
