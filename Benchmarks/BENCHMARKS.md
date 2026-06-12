# Benchmark Reproducibility Instructions
## LiQuID — *Observable-Dependent Convergence in Monte Carlo Wave-Function Simulations*
### Computer Physics Communications submission

Three quantitative benchmarks validate the adaptive SEM-based stopping criterion
implemented in LiQuID.  Each benchmark is self-contained: one C++ executable
produces the CSV data, one Python script reads those CSVs and writes the figures
that appear in the paper.

---
## Repository layout

After cloning (or extracting the CPC archive), the relevant directories are:

```
LiQuID/
├── include/liquid/          # Header-only library (all physics and statistics)
│   ├── core/                #   RNG, types, config
│   ├── linalg/              #   Dense and CSR sparse operators
│   ├── ode/                 #   Dormand–Prince RK45, RK4
│   ├── system/              #   OpenSystem, LindbladSet
│   ├── trajectory/          #   MCWFPropagator, TrajectoryState
│   ├── ensemble/            #   Welford/RunningStatistics, ConvergenceMonitor,
│   │                        #     EnsembleManager, AllocatorPolicy seam
│   └── simulation/          #   SimulationBuilder, ParameterSweepBuilder
├── src/core/rng.cpp         # One compiled translation unit (seed sequencing)
├── tests/
│   ├── unit/                # 101 unit tests
│   ├── validation/          # 40 validation tests (analytic comparisons)
│   └── benchmark/           # Legacy timing benchmarks (not paper figures)
├── benchmarks/              # ← PAPER REPRODUCTION LIVES HERE
│   ├── BENCHMARK_README.md  #   This file
│   ├── Makefile
│   ├── run_benchmarks.sh    #   One-command pipeline
│   ├── bench_convergence.cpp           # → Table 3
│   ├── bench1_adaptive_stopping.cpp    # → Table 4, Figure 1
│   ├── bench2_variance_dependence.cpp  # → Table 1, Table 5, Figures 2–3
│   ├── bench3_fixed_vs_adaptive.cpp    # → Tables 2 & 6, Figures 4–7
│   ├── bench1_adaptive_stopping_plot.py
│   ├── bench2_variance_dependence_plot.py
│   ├── bench3_fixed_vs_adaptive_plot.py
│   ├── bench_adaptive_stopping_plot.py # (auxiliary — efficiency scatter)
│   ├── results/             #   CSV data files (created at runtime)
│   └── figs/                #   PDF/PNG figures (created at runtime)
├── CMakeLists.txt
└── README.md
```

## File map

| Benchmark | C++ source | Python plotter | CSVs produced | Figures produced |
|-----------|-----------|----------------|---------------|-----------------|
| **B1 — SEM scaling** | `bench1a_sem_scaling.cpp` | `validate_sem_scaling.py`*(Validates statistical scaling and convergence in benchmark error data.)* | `bench1a_sem_scaling.csv` | — |
| **B1 — Adaptive stopping** | `bench1b_adaptive_stopping.cpp` | `bench1b_adaptive_stopping_plot.py` | `bench_adaptive_stopping.csv` | `bench_adaptive_stopping_fig1.pdf/png` |
| **B2 — Variance dependence** | `bench2_variance_dependence.cpp` | `bench2_variance_dependence_plot.py` | `bench2_variance.csv`, `bench2_main.csv` | `bench2_fig1_Nadaptive_bar.pdf`, `bench2_fig4_efficiency.pdf` |
| **B3 — Fixed-N vs adaptive** | `bench3_fixed_vs_adaptive.cpp` | `bench3_fixed_vs_adaptive_plot.py` | `bench3_adaptive.csv`, `bench3_fixed.csv`, `bench3_comparison.csv` | `fig1_bar_N_adaptive.pdf`, `fig2_waste_factor.pdf`, `fig3_relsem_heatmap.pdf`, `fig4_runtime.pdf` |


> **Note on `bench2_convergence.csv`.**  Step 3 of `bench2_variance_dependence.cpp`
> (the convergence-history loop that writes this file) is currently wrapped in
> `/* … */` comments.  The CSV is therefore **not produced** by default and
> `bench2_fig2_convergence.pdf` cannot be generated until that block is
> uncommented.

---

## Prerequisites

### C++ compiler

Any C++17-conforming compiler:

```bash
g++ --version      # GCC ≥ 9
clang++ --version  # Clang ≥ 9
```

LiQuID is header-only.  Add the `include/` directory to the compiler's include
path — no library to link against (OpenMP is optional).

### Python packages

```bash
pip install numpy pandas matplotlib
```

Tested with Python ≥ 3.9, NumPy ≥ 1.21, pandas ≥ 1.3, Matplotlib ≥ 3.5.

---


## Build

Set `LIQUID_ROOT` in `benchmarks/Makefile` to point to the `LiQuID/` directory,
then:

```bash
cd benchmarks/
make all
```

---

## Run everything at once

```bash
bash run_benchmarks.sh
```

Compiles all binaries, runs them (CSV data → `results/`), then runs all
plotting scripts (figures → `figs/`). Approximate total runtime: 55 min
(single core, Intel Core i7-4510U).

---

## Reproducing each paper figure and table

### Table 1 — Per-observable variance characterisation

**Binary:** `bench2_variance_dependence`
**CSV:** `results/bench2_variance.csv`

```bash
./bench2_variance_dependence
```

---

### Table 2 — Fixed-N failure modes

**Binary:** `bench3_fixed_vs_adaptive`
**CSV:** `results/bench3_comparison.csv`

```bash
./bench3_fixed_vs_adaptive
```

---

### Table 3 — Fixed-N convergence of ⟨σ_z(T=1)⟩

**Binary:** `bench1a_sem_scaling`
**CSV:** `results/bench1a_sem_scaling.csv`

```bash
./bench1a_sem_scaling
```

---

### Table 4 — Benchmark I: adaptive stopping efficiency

**Binary:** `bench1b_adaptive_stopping`
**CSV:** `results/bench_adaptive_stopping.csv`

```bash
./bench1b_adaptive_stopping
```

---

### Table 5 — Benchmark II: observable-dependent trajectory allocation

**Binary:** `bench2_variance_dependence`
**CSV:** `results/bench2_main.csv`

```bash
./bench2_variance_dependence
```

---

### Table 6 — Benchmark III: adaptive run reference

**Binary:** `bench3_fixed_vs_adaptive`
**CSV:** `results/bench3_adaptive.csv`

```bash
./bench3_fixed_vs_adaptive
```

---

### Figure 1 — Adaptive stopping: N_adaptive vs N_optimal and efficiency per target

**Source:** `bench_adaptive_stopping.csv`
**Output:** `figs/bench_adaptive_stopping_fig1.pdf`

```bash
python3 bench1b_adaptive_stopping_plot.py
```

---

### Figure 2 — Required adaptive trajectory counts per observable (bar chart)

**Source:** `bench2_main.csv`
**Output:** `figs/bench2_fig1_Nadaptive_bar.pdf`

```bash
python3 bench2_variance_dependence_plot.py
```

---

### Figure 3 — Adaptive stopping efficiency η vs N_optimal (scatter)

**Source:** `bench2_main.csv`
**Output:** `figs/bench2_fig4_efficiency.pdf`

```bash
python3 bench2_variance_dependence_plot.py --data results/ --out figs/
```

*(Figures 2 and 3 are both produced by the same script in one call.)*

---

### Figure 4 — Adaptive N vs fixed-N reference lines

**Source:** `bench3_adaptive.csv`
**Output:** `figs/fig1_bar_N_adaptive.pdf`

```bash
python3 bench3_fixed_vs_adaptive_plot.py --data results/ --out figs/
```

---

### Figure 5 — Waste factor per fixed-N choice

**Source:** `bench3_comparison.csv`
**Output:** `figs/fig2_waste_factor.pdf`

```bash
python3 bench3_fixed_vs_adaptive_plot.py --data results/ --out figs/
```

---

### Figure 6 — Achieved RelSEM heatmap

**Source:** `bench3_comparison.csv`, `bench3_adaptive.csv`
**Output:** `figs/fig3_relsem_heatmap.pdf`

```bash
python3 bench3_fixed_vs_adaptive_plot.py --data results/ --out figs/
```

---

### Figure 7 — Wall-clock runtime: adaptive vs fixed-N

**Source:** `bench3_fixed.csv`, `bench3_adaptive.csv`
**Output:** `figs/fig4_runtime.pdf`

```bash
python3 bench3_fixed_vs_adaptive_plot.py --data results/ --out figs/
```

### Numerical reproducibility notes

**Deterministic seeding.** Each benchmark derives per-trajectory seeds from
a user-supplied base seed using LiQuID's splittable xoshiro256\*\* generator
(`include/liquid/core/rng.hpp`). Trajectory n always produces the same quantum
jump sequence regardless of the total ensemble size, so re-running any
benchmark will produce bit-identical CSV values on the same platform and
compiler.

**Cross-platform variance.** Results are IEEE 754 double-precision throughout.
Minor differences in the last 1–2 decimal places of wall-clock times and
non-deterministic thread scheduling may arise across platforms, but all
quantitative claims in the paper (efficiency η, RelSEM achieved, waste
factors) are stable to the precision reported.

**Reference variance estimation.** Benchmarks I–III derive σ² and |µ| from a
N_ref = 10⁶ reference run embedded in the binary. The reference run uses a
fixed base seed (`seed = 42`) so the derived N_opt values are identical across
runs. If you change the seed, N_opt values will shift by O(1/√N_ref) ≈ 0.1%,
which is within the stopping variability already reported.

**Minimum-N guard.** `N_min = 100` is hard-coded in all three production
benchmarks. Changing it will alter the P_g entries at ε = 10% and 5% in
Table 5 (Benchmark II) where N_opt < N_min.

## Paper figure checklist

After a successful run the following files must exist for `main.tex` to compile
without missing-figure warnings:

```
bench_adaptive_stopping_fig1.pdf   ← bench1b + bench1_adaptive_stopping_plot.py
bench2_fig1_Nadaptive_bar.pdf      ← bench2 + bench2_variance_dependence_plot.py
bench2_fig4_efficiency.pdf         ← bench2 + bench2_variance_dependence_plot.py
fig1_bar_N_adaptive.pdf            ← bench3 + bench3_fixed_vs_adaptive_plot.py
fig2_waste_factor.pdf              ← bench3 + bench3_fixed_vs_adaptive_plot.py
fig3_relsem_heatmap.pdf            ← bench3 + bench3_fixed_vs_adaptive_plot.py
fig4_runtime.pdf                   ← bench3 + bench3_fixed_vs_adaptive_plot.py
```
---
