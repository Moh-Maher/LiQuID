# Benchmark 4 — Higher-Dimensional Variance Check (Checking Only)
## LiQuID — *Observable-Dependent Convergence in Monte Carlo Wave-Function Simulations*

This file documents the `bench4_*` files separately from `BENCHMARKS.md`
because, unlike Benchmarks I–III and the Table 8 QuTiP comparison, **none of
`bench4`'s output corresponds to any table or figure published in
`the_paper.pdf`.** It exists purely as an internal check and is not part of
the paper's reproduction pipeline.

---

## What this benchmark is for

`bench4_higher_dim_variance.cpp` extends Benchmark 2's argument (observable-
dependent convergence) from the single-qubit (d = 2) system used throughout
the paper to a d = 4 system: two coupled, dissipative qubits. The question it
answers is whether the observable-dependent variance pathology demonstrated
in the d = 2 benchmarks is an artefact of the small Hilbert space, or persists
at higher dimension.

This is exactly the extension flagged in §8.1.1 *"Many-body systems: local
vs. global observables"* of `the_paper.pdf`. The paper is explicit that this
many-body benchmark was **not included in the manuscript**: no independent
analytic or numerically-exact reference for N_opt exists at moderate-to-large
system size, so the kind of ground-truth validation used in §5–7 isn't
available here. The paper flags a spin-chain extension — validated against
exact diagonalization or DMRG-computed moments for small L — as a direction
for future work. `bench4` is a step in that direction, kept in the repository
for internal checking, not as evidence backing any published claim.

---

## Physical system

Two coupled, dissipative qubits (d = 4):

```
H  = Ω₁ σx⊗I + Ω₂ I⊗σx + J σz⊗σz         coherent drive + Ising coupling
L₁ = √γ₁ σ₋⊗I                             local decay, qubit 1
L₂ = √γ₂ I⊗σ₋                             local decay, qubit 2
ψ₀ = |e₁,e₂⟩ = (1,0,0,0)ᵀ                  both qubits excited
Ω₁ = 0.5, Ω₂ = 0.2, J = 0.25, γ₁ = 1.0, γ₂ = 0.9
T  = 12.0   (≳ 10 × the slowest decay time 1/γ₂ ≈ 1.11 → near steady state)
```

Basis ordering (row-major Kronecker product, local basis {|e⟩,|g⟩}):
index 0 = |e₁,e₂⟩, 1 = |e₁,g₂⟩, 2 = |g₁,e₂⟩, 3 = |g₁,g₂⟩.

Five observables with different statistical behaviour:

| Observable | Definition | Note |
|---|---|---|
| `Pe1` | Pe⊗I | marginal excited-state population, qubit 1 |
| `Pe2` | I⊗Pe | marginal excited-state population, qubit 2 |
| `Pee` | Pe⊗Pe | joint excited-state population (both qubits) |
| `sz1` | σz⊗I | ⟨σz⟩ of qubit 1 |
| `sz2` | I⊗σz | ⟨σz⟩ of qubit 2 — parameters chosen so it sits near its steady-state floor, deliberately the low-variance end of the spread (mirrors the role P_g played in Benchmark 2) |

---

## File map

| File | Role |
|---|---|
| `bench4_higher_dim_variance.cpp` | C++ binary: reference run, adaptive stopping sweep, N_max/N_min ratio, fixed-N comparison |
| `bench4_higher_dim_variance_plot.py` | Python plotter reading the CSVs below |
| `results/bench4_variance.csv` | σ², \|µ\|, σ/\|µ\| per observable (reference run) |
| `results/bench4_main.csv` | N_optimal / N_adaptive / efficiency per (observable, ε) |
| `results/bench4_ratio.csv` | N_max/N_min ratio across observables, per ε target |
| `results/bench4_fixed.csv` | Fixed-N comparison (small N vs. large N, all observables) |
| `results/bench4_N_state.csv` | Raw (obs_idx, eps_idx) → N_adaptive accumulation |
| `results/bench4_ref_state.csv` | Reference-run state |
| `figs/bench4_fig1_Nadaptive_bar.pdf` | Required N per observable at ε = 1% |
| `figs/bench4_fig2_ratio_vs_eps.pdf` | N_max/N_min ratio as ε tightens |
| `figs/bench4_fig3_variance.pdf` | σ² and σ/\|µ\| per observable |
| `figs/bench4_fig4_fixed_comparison.pdf` | Fixed-N failure/oversampling panel |

**None of the four figures or six CSVs above appear anywhere in
`the_paper.pdf`.** They are not referenced by `main.tex`, and their absence
has no effect on paper compilation.

---

## Benchmark structure (as implemented)

1. **Step 1 — Reference run.** Single N_ref-trajectory batch (fixed seed 42),
   all five observables, → mean, σ², σ/\|µ\| per observable
   (`bench4_variance.csv`, `bench4_ref_state.csv`).
2. **Step 2 — Adaptive stopping.** Per (observable, ε) at
   ε ∈ {10%, 5%, 2%, 1%, 0.5%}. Records N_optimal, N_adaptive,
   efficiency = N_optimal/N_adaptive, achieved rel_SEM
   (`bench4_main.csv`, `bench4_N_state.csv`).
3. **Step 3 — N_max/N_min ratio** across observables, at each ε and overall
   (`bench4_ratio.csv`).
4. **Step 4 — Fixed-N comparison.** One small N (favours the lowest-variance
   observable) vs. one large N, all observables (`bench4_fixed.csv`).

`stop_at_sem(eps)` stops when *all* registered observables satisfy
rel_SEM < eps; to drive stopping on a single observable, only that observable
is registered. Reference/fixed-N runs (min = max = N, target = 1e-15)
register all five observables at once, since the run length is fixed
regardless of any one observable's precision — this reuses one trajectory
batch for all five statistics instead of five separate fixed-N runs.

---

## Running it

```bash
cd benchmarks/
./bench4_higher_dim_variance
python3 bench4_higher_dim_variance_plot.py --data results/ --out figs/
```

or, via the same variance/pattern as the other benchmarks:

```bash
python3 bench4_higher_dim_variance_plot.py            # CSVs in cwd
python3 bench4_higher_dim_variance_plot.py --show
```

---

## Why this is kept separate from `BENCHMARKS.md`

`BENCHMARKS.md` documents reproduction of everything that actually appears in
`the_paper.pdf` — Tables 1–6 and Figures 1–7 from Benchmarks I–III, and
Table 8 from the QuTiP-vs-LiQuID comparison. `bench4` produces no table or
figure the paper uses, so keeping it in this separate file avoids implying,
in the main reproduction guide, that these outputs are needed to reproduce
any part of the published manuscript.
