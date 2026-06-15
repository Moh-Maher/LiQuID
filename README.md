# LiQuID — Lindblad Quantum Integrated Dynamics

A high-performance C++17 framework for simulating open quantum systems using the
Monte Carlo Wave Function (MCWF) method, with adaptive trajectory management,
sparse operator support, and an optional machine-learning allocation policy.

---

## Table of Contents

1. [What LiQuID is](#what-liquid-is)
2. [Quick start](#quick-start)
3. [Building](#building)
4. [Core concepts](#core-concepts)
5. [API reference](#api-reference)
   - [Operators](#operators)
   - [SimulationBuilder](#simulationbuilder)
   - [Simulation factories](#simulation-factories)
   - [Observable definitions](#observable-definitions)
   - [Stopping criteria](#stopping-criteria)
   - [Parameter sweeps](#parameter-sweeps)
   - [ML policy](#ml-policy)
6. [Examples](#examples)
   - [Two-level decay](#example-1-two-level-decay)
   - [Driven qubit steady state](#example-2-driven-qubit-steady-state)
   - [Sparse cavity QED](#example-3-sparse-cavity-qed)
   - [Time-dependent Hamiltonian](#example-4-time-dependent-hamiltonian)
   - [Parameter sweep](#example-5-parameter-sweep)
   - [ML allocation policy](#example-6-ml-allocation-policy)
7. [Architecture overview](#architecture-overview)
8. [Performance guide](#performance-guide)
9. [Extending LiQuID](#extending-liquid)
10. [Test suite](#test-suite)
11. [Known limitations](#known-limitations)
12. [Citation](#citation)
---

## What LiQuID is

LiQuID solves the Lindblad master equation

```
dρ/dt = -i[H,ρ] + Σₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})
```

by stochastic unravelling into quantum trajectories (the MCWF method).
Each trajectory evolves a wavefunction under an effective non-Hermitian
Hamiltonian and undergoes random quantum jumps. Ensemble averages reproduce
the density matrix solution.

**LiQuID is not a general quantum computing toolkit.** Its identity is
intelligent MCWF simulation: adaptive solvers, convergence-based stopping,
sparse operator support for large Hilbert spaces, and an ML-ready allocation
policy seam.

---

## Quick start

```cpp
#include "liquid/liquid.hpp"
using namespace liquid;
using namespace liquid::ensemble;  // for EnsembleResult, ObservableDef, etc.

int main() {
    // Two-level atom decaying at rate gamma
    const double omega = 1.0, gamma = 1.0, T = 3.0;

    DenseOperator H(2);
    H(0,0) = Scalar{ omega/2, 0};
    H(1,1) = Scalar{-omega/2, 0};

    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0};  // sigma_minus

    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1, 0};
    sz(1,1) = Scalar{-1, 0};

    Simulation sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .seed(42)
        .dt(1e-3)
        .stop_at_sem(0.01)          // stop when relative SEM < 1%
        .min_trajectories(100)
        .max_trajectories(10000)
        .build();

    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};    // excited state

    auto result = sim.run(psi0, 0.0, T);

    printf("<sigma_z> = %.4f +/- %.4f  (N=%zu)\n",
        result.observables[0].mean,
        result.observables[0].sem,
        result.total_trajectories);
    // Expected: <sigma_z> = 2*exp(-3) - 1 = -0.900
}
```

Compile:
```bash
g++ -std=c++17 -O2 -Iinclude my_sim.cpp src/core/rng.cpp -o my_sim
```

---

## Building

### Requirements

- C++17 compiler (GCC ≥ 8, Clang ≥ 7, MSVC 2019+)
- CMake ≥ 3.16 (optional but recommended)
- OpenMP (optional, for parallel ensemble execution)

### CMake (recommended)

```bash
git clone https://github.com/yourorg/LiQuID
cd LiQuID
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build        # run all 141 tests
```

#### CMake options

| Option | Default | Description |
|---|---|---|
| `LIQUID_OPENMP` | ON | Enable OpenMP parallel ensemble |
| `LIQUID_TESTS` | ON | Build unit and validation tests |
| `LIQUID_EXAMPLES` | ON | Build example programs |
| `LIQUID_SANITIZE` | OFF | Enable AddressSanitizer + UBSan |

```bash
# Debug build with sanitizers
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug -DLIQUID_SANITIZE=ON
cmake --build build_debug -j$(nproc)
```

### Manual compilation

All headers are in `include/`. The only compiled source file is `src/core/rng.cpp`.

```bash
# Minimal: single simulation file
g++ -std=c++17 -O2 -Iinclude my_sim.cpp src/core/rng.cpp -o my_sim

# With OpenMP
g++ -std=c++17 -O2 -fopenmp -Iinclude my_sim.cpp src/core/rng.cpp -o my_sim

# Debug
g++ -std=c++17 -O0 -g -DLIQUID_DEBUG -Iinclude my_sim.cpp src/core/rng.cpp -o my_sim
```

---

## Core concepts

### The MCWF algorithm

LiQuID evolves each trajectory under the non-Hermitian effective Hamiltonian:

```
H_eff = H - (i/2) Σₖ Lₖ†Lₖ
```

The wavefunction norm decreases continuously. When it drops below a
uniform random threshold `r ∈ (0,1]`, a quantum jump fires on channel `k`
with probability `‖Lₖ|ψ⟩‖² / Σⱼ ‖Lⱼ|ψ⟩‖²`. After the jump the state
is renormalized and a fresh `r` is drawn.

Ensemble average over N trajectories gives:

```
⟨Ô⟩(t) = (1/N) Σᵢ ⟨ψᵢ(t)|Ô|ψᵢ(t)⟩ / ⟨ψᵢ(t)|ψᵢ(t)⟩
```

### Observables

Observable functions receive the (possibly unnormalized) MCWF wavefunction.
**Always normalize before computing expectation values:**

```cpp
// Correct
[op](const StateVector& psi) -> Real {
    const Real ns = psi.norm_sq();
    return ns > 1e-30
        ? expectation(op, psi).real() / ns
        : 0.0;
}
```

The convenience function `make_operator_observable(name, op)` handles this
automatically for dense operators.

### Scalar type

All quantum amplitudes use `liquid::Scalar = std::complex<double>`.
Real-valued quantities use `liquid::Real = double`.

---

## API reference

### Operators

#### DenseOperator

```cpp
DenseOperator A(Dim n);             // n×n zero matrix
A(i, j) = Scalar{real, imag};      // element assignment
A(i, j);                           // element access

// Matrix-vector product (allocating)
StateVector out = A.apply(psi);

// Matrix-vector product (non-allocating, hot-path interface)
A.apply_add(psi, Scalar{alpha,0}, out);  // out += alpha * A * psi

DenseOperator Adag = A.adjoint();   // Hermitian conjugate
DenseOperator B    = A.matmul(C);  // matrix product

// Convenience constructors
DenseOperator I  = make_identity(n);
DenseOperator Z  = make_zero_operator(n);
DenseOperator LdagL = adjoint_times(L);  // L†L

// Expectation value <psi|O|psi>
Scalar ev = expectation(O, psi);
```

#### SparseOperator (CSR format)

```cpp
// Construct from (row, col, value) triplets
std::vector<Triplet> trips = {
    {0, 0, Scalar{1.0, 0.0}},
    {1, 0, Scalar{0.0, 1.0}},
};
SparseOperator A(dim, std::move(trips));

// Duplicate (row,col) entries are automatically summed.
// Entries with |value| < 1e-300 are dropped.

A.size();           // matrix dimension
A.nnz();            // number of stored elements
A.sparsity();       // fraction of non-zeros
A(i, j);            // element access (for testing, not hot path)

// Same hot-path interface as DenseOperator
A.apply_add(psi, alpha, out);

// Algebraic operations (allocating, for precomputation)
SparseOperator Adag = A.adjoint();
SparseOperator B    = A.matmul(C);
A.add_scaled(B, alpha);  // A += alpha * B (rebuilds CSR — use at construction only)

// Expectation value
Scalar ev = sparse_expectation(A, psi);

// Conversions
SparseOperator S = to_sparse(dense_op, zero_tol=1e-14);
DenseOperator  D = sparse_op.to_dense();

// Convenience
SparseOperator I    = make_sparse_identity(n);
SparseOperator LdagL = sparse_adjoint_times(L);
```

#### StateVector

```cpp
StateVector psi(Dim n);          // zero-initialized
psi[i] = Scalar{real, imag};

psi.norm_sq();                   // ‖ψ‖²
psi.norm();                      // ‖ψ‖
psi.normalize();                 // ψ → ψ/‖ψ‖  (in-place)
psi.scale(alpha);                // ψ → α·ψ
psi.add_scaled(phi, alpha);      // ψ += α·φ
psi.set_zero();
psi.copy_from(other);

Scalar ip = StateVector::inner(phi, psi);  // ⟨φ|ψ⟩
```

---

### SimulationBuilder

The primary API for dense, time-independent systems.

```cpp
Simulation sim = SimulationBuilder{}
    // ── System ────────────────────────────────────────────────────
    .hamiltonian(H)                   // DenseOperator H
    .collapse_operator(L1)            // Add one Lindblad operator
    .collapse_operator(L2)            // Add more as needed

    // ── Observables ───────────────────────────────────────────────
    .observe("name", op)              // DenseOperator: computes <psi|op|psi>/<psi|psi>
    .observe("name", fn)              // Custom ObservableFn

    // ── ODE solver ────────────────────────────────────────────────
    .dt(1e-3)                         // Initial stepsize (DOPRI45 adapts from here)
    .tolerances(1e-8, 1e-6)          // atol, rtol for adaptive solver

    // ── Ensemble ──────────────────────────────────────────────────
    .seed(42)                         // Global RNG seed
    .num_threads(4)                   // OpenMP threads (requires -fopenmp)
    .diagnostics(DiagnosticLevel::Summary)

    // ── Stopping ──────────────────────────────────────────────────
    .stop_at_sem(0.01)                // Relative SEM target (default: 1%)
    .min_trajectories(100)            // Never stop before this
    .max_trajectories(100000)         // Always stop at this
    .max_wall_time(60.0)              // Wall-clock budget in seconds (0 = unlimited)

    // ── ML policy (optional) ──────────────────────────────────────
    .allocator_policy(make_ml_policy(seed))

    .build();                         // Validates config, returns Simulation
```

**Validation at `build()` time:**
- Hamiltonian must be set
- At least one collapse operator required
- At least one observable required
- All operators must have the same dimension
- `min_trajectories ≤ max_trajectories`

---

### Simulation factories

For sparse operators, time-dependent Hamiltonians, or custom stepper types,
use the static factory methods:

```cpp
// Generic factory: any system type × stepper type
auto sim = Simulation::make<SystemType, StepperType>(
    system,       // moved in
    observables,  // std::vector<ObservableDef>
    stopping,     // StoppingCriteria
    config,       // EnsembleConfig
    policy        // AllocatorPolicy (optional)
);

// Convenience: SparseOpenSystem + DormandPrince45
auto sim = Simulation::make_sparse_dopri45(
    sparse_system, observables, stopping, config);

// Convenience: DenseOpenSystem + DormandPrince45
auto sim = Simulation::make_dense_dopri45(
    dense_system, observables, stopping, config);
```

#### EnsembleConfig

```cpp
EnsembleConfig ec;
ec.global_seed            = 42;
ec.num_threads            = 1;
ec.diag_level             = DiagnosticLevel::None;
ec.propagator.dt_initial  = 1e-3;
ec.propagator.atol        = 1e-8;
ec.propagator.rtol        = 1e-6;
ec.propagator.dt_min      = 1e-12;
ec.propagator.dt_max      = 0.1;
```

---

### Observable definitions

```cpp
// From a dense operator (automatic normalization)
ObservableDef obs = make_operator_observable("name", dense_op);

// From a sparse operator (manual normalization required)
SparseOperator n_op = make_photon_number(N_fock);
ObservableDef obs = {
    "photon_number",
    [op = std::move(n_op)](const StateVector& psi) -> Real {
        const Real ns = psi.norm_sq();
        return ns > 1e-30 ? sparse_expectation(op, psi).real() / ns : 0.0;
    }
};

// Custom function (any calculation on the wavefunction)
ObservableDef obs = {
    "purity",
    [](const StateVector& psi) -> Real {
        // For MCWF: each trajectory is a pure state, so purity = 1 always.
        // Useful for checking normalization.
        const Real ns = psi.norm_sq();
        return ns > 1e-30 ? 1.0 : 0.0;
    }
};
```

---

### Stopping criteria

```cpp
StoppingCriteria sc;
sc.target_rel_sem    = 0.01;    // Stop when SEM/|mean| < 1% for all observables
sc.min_trajectories  = 100;     // Never stop before this (enforced by default)
sc.max_trajectories  = 100000;  // Hard ceiling
sc.max_wall_seconds  = 0.0;     // Wall-clock budget (0 = unlimited)
sc.enforce_min       = true;    // Enforce min_trajectories even after convergence
```

Stopping decisions in priority order:
1. External stop request (`sim.request_stop()` or signal)
2. `max_trajectories` reached
3. `max_wall_seconds` exceeded
4. `min_trajectories` not yet reached → continue
5. `target_rel_sem` satisfied → converged

---

### Parameter sweeps

```cpp
auto sweep = ParameterSweepBuilder{}
    // Explicit values
    .parameter("gamma", {0.1, 0.5, 1.0, 2.0, 5.0})

    // OR: linear range [lo, hi] with N points (inclusive)
    .parameter_range("gamma", 0.1, 5.0, 10)

    // OR: log-spaced range [lo, hi] with N points (inclusive)
    .parameter_logrange("g_over_kappa", 0.05, 5.0, 10)

    // Factory: called once per parameter value, returns a Simulation
    .simulation_factory([](double gamma) {
        return SimulationBuilder{}
            .hamiltonian(H)
            .collapse_operator(make_L(gamma))
            .observe("sz", sz)
            .max_trajectories(1000)
            .build();
    })

    .initial_state(psi0)
    .time_interval(0.0, T)
    .build();

SweepResult result = sweep.run();

// Output
result.print_table();              // formatted console table
result.save_csv("output.csv");     // CSV: param,obs_mean,obs_sem,...
result.save_json("output.json");   // structured JSON

// Access programmatically
for (const auto& pt : result.points) {
    printf("gamma=%.3f  <sz>=%.4f +/- %.4f\n",
        pt.param_value,
        pt.result.observables[0].mean,
        pt.result.observables[0].sem);
}
```

#### CSV format

```
gamma,sigma_z_mean,sigma_z_sem,sigma_z_rel_sem,N_trajectories,N_failed,mean_jumps_per_traj,total_wall_s,traj_wall_s
0.1,0.7408,0.0031,0.0042,1000,0,0.095,0.234,0.000234
0.5,0.2231,0.0044,0.0197,1000,0,0.394,0.198,0.000198
...
```

---

### ML policy

The ML policy trains a small feedforward neural network online as trajectories
complete. At Phase 6 it uses the same trajectory generation as the uniform
policy; the trained network provides a foundation for future seed-selection
and rare-event-biasing strategies.

```cpp
// Drop-in replacement for the uniform policy
AllocatorPolicy policy = make_ml_policy(
    42,     // global seed
    8,      // hidden layer size (default: 8)
    0.01    // learning rate (default: 0.01)
);

// Use in SimulationBuilder
Simulation sim = SimulationBuilder{}
    ...
    .allocator_policy(make_ml_policy(42))
    .build();

// Use directly with EnsembleManager
EnsembleManager<SystemType, StepperType> mgr(
    &sys, obs, stopping, config, make_ml_policy(42));
```

**Custom policy** — any callable with the right signature:

```cpp
// AllocatorPolicy = std::function<AllocationDecision(const EnsembleSummary&)>
AllocatorPolicy my_policy = [](const EnsembleSummary& s) -> AllocationDecision {
    static TrajId next_id = 0;
    static Seed   seed_counter = 99;
    return AllocationDecision{
        AllocAction::SpawnNew,
        seed_counter++ ^ (next_id * 0x9e3779b97f4a7c15ULL),
        next_id++
    };
};
```

---

## Examples

### Example 1: Two-level decay

```cpp
#include "liquid/liquid.hpp"
using namespace liquid;

int main() {
    // H = (ω/2)σ_z,  L = √γ σ_-
    // Exact: <σ_z(t)> = 2exp(-γt) - 1
    const double omega = 1.0, gamma = 1.0, T = 2.0;

    DenseOperator H(2), L(2), sz(2);
    H(0,0) = Scalar{ omega/2, 0}; H(1,1) = Scalar{-omega/2, 0};
    L(1,0) = Scalar{ std::sqrt(gamma), 0};
    sz(0,0) = Scalar{1,0}; sz(1,1) = Scalar{-1,0};

    Simulation sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .seed(42).dt(1e-3)
        .stop_at_sem(0.01).min_trajectories(100).max_trajectories(10000)
        .build();

    StateVector psi0(2); psi0[0] = Scalar{1,0};
    auto result = sim.run(psi0, 0.0, T);

    printf("<sigma_z> = %.4f +/- %.4f  (N=%zu, exact=%.4f)\n",
        result.observables[0].mean,
        result.observables[0].sem,
        result.total_trajectories,
        2.0*std::exp(-gamma*T)-1.0);
}
```

---

### Example 2: Driven qubit steady state

```cpp
#include "liquid/liquid.hpp"
using namespace liquid;

int main() {
    // Resonant drive in rotating frame:
    // H = Ω σ_x,  L = √γ σ_-
    // Steady state: ρ_ee = 4Ω²/(γ² + 8Ω²)
    const double Omega = 0.5, gamma = 1.0, T_ss = 20.0;

    DenseOperator H(2), L(2), rho_ee(2);
    H(0,1) = H(1,0) = Scalar{Omega, 0};
    L(1,0) = Scalar{std::sqrt(gamma), 0};
    rho_ee(0,0) = Scalar{1,0};

    Simulation sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("rho_ee", std::move(rho_ee))
        .seed(42).dt(5e-4)
        .stop_at_sem(0.02).min_trajectories(200).max_trajectories(5000)
        .build();

    StateVector psi0(2); psi0[0] = Scalar{1,0};
    auto result = sim.run(psi0, 0.0, T_ss);

    const double exact = 4*Omega*Omega / (gamma*gamma + 8*Omega*Omega);
    printf("rho_ee = %.4f +/- %.4f  (exact=%.4f)\n",
        result.observables[0].mean,
        result.observables[0].sem, exact);
}
```

---

### Example 3: Sparse cavity QED

```cpp
#include "liquid/liquid.hpp"
using namespace liquid;
using namespace liquid::ensemble;

int main() {
    // Jaynes-Cummings: H = g(a†σ_- + aσ_+) + ε(a + a†)
    // Basis: |n,s> → index 2n+s  (s=0:excited, s=1:ground)
    const int    N_fock = 15;
    const double g = 1.0, kappa = 1.0, gamma = 0.1, epsilon = 0.2;
    const Dim    dim = 2 * N_fock;

    auto idx = [](int n, int s){ return static_cast<Idx>(2*n+s); };

    // Build Hamiltonian (JC coupling + coherent drive)
    std::vector<Triplet> H_trips;
    for (int n = 0; n < N_fock; ++n) {
        if (n+1 < N_fock) {
            double v_jc = g * std::sqrt(n+1.0);
            H_trips.push_back({idx(n+1,1), idx(n,0), Scalar{v_jc, 0}});
            H_trips.push_back({idx(n,0), idx(n+1,1), Scalar{v_jc, 0}});
            double v_dr = epsilon * std::sqrt(n+1.0);
            for (int s : {0,1}) {
                H_trips.push_back({idx(n+1,s), idx(n,s), Scalar{v_dr,0}});
                H_trips.push_back({idx(n,s), idx(n+1,s), Scalar{v_dr,0}});
            }
        }
    }
    SparseOperator H(dim, std::move(H_trips));

    // Cavity decay L1 = √κ a
    std::vector<Triplet> L1t;
    for (int n=1; n<N_fock; ++n) {
        double v = std::sqrt(kappa*n);
        for (int s : {0,1})
            L1t.push_back({idx(n-1,s), idx(n,s), Scalar{v,0}});
    }
    // Atomic decay L2 = √γ σ_-
    std::vector<Triplet> L2t;
    for (int n=0; n<N_fock; ++n)
        L2t.push_back({idx(n,1), idx(n,0), Scalar{std::sqrt(gamma),0}});

    std::vector<SparseOperator> ops;
    ops.emplace_back(dim, std::move(L1t));
    ops.emplace_back(dim, std::move(L2t));

    SparseOpenSystem sys(std::move(H), LindbladSet<SparseTag>(std::move(ops)));

    // Photon-number observable
    std::vector<Triplet> n_trips;
    for (int n=0; n<N_fock; ++n)
        for (int s : {0,1})
            n_trips.push_back({idx(n,s), idx(n,s), Scalar{(double)n,0}});
    SparseOperator n_op(dim, std::move(n_trips));

    std::vector<ObservableDef> obs;
    obs.push_back({"photon_number",
        [op = std::move(n_op)](const StateVector& psi) -> Real {
            const Real ns = psi.norm_sq();
            return ns > 1e-30
                ? sparse_expectation(op, psi).real() / ns : 0.0;
        }});

    StoppingCriteria sc;
    sc.min_trajectories = 300; sc.max_trajectories = 3000;
    sc.target_rel_sem   = 0.05;

    EnsembleConfig ec;
    ec.global_seed = 42; ec.diag_level = DiagnosticLevel::None;
    ec.propagator.dt_initial = 5e-3;

    auto sim = Simulation::make_sparse_dopri45(
        std::move(sys), std::move(obs), sc, ec);

    StateVector psi0(dim); psi0[idx(0,1)] = Scalar{1,0};
    auto result = sim.run(psi0, 0.0, 30.0);

    printf("<n>_ss = %.4f +/- %.4f  (N=%zu)\n",
        result.observables[0].mean,
        result.observables[0].sem,
        result.total_trajectories);
}
```

---

### Example 4: Time-dependent Hamiltonian

```cpp
#include "liquid/liquid.hpp"
using namespace liquid;
using namespace liquid::ensemble;

int main() {
    // H(t) = H_static + f(t) * H_drive
    // f(t) = sin(ω_d * t)  (modulated drive)
    const double omega = 1.0, gamma = 0.5, Omega = 0.3, omega_d = 1.0;

    DenseOperator H_static(2);   // zero (resonant frame)
    DenseOperator H_drive(2);
    H_drive(0,1) = H_drive(1,0) = Scalar{Omega, 0};

    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0};

    std::vector<DriveTerm<DenseTag>> drives;
    drives.push_back({
        [omega_d](Real t){ return std::sin(omega_d * t); },
        std::move(H_drive)
    });

    std::vector<DenseOperator> ops; ops.push_back(std::move(L));
    LindbladSet<DenseTag> lb(std::move(ops));
    DenseTDOpenSystem sys(std::move(H_static), std::move(drives),
                          std::move(lb));

    // Register observable
    DenseOperator rho_ee(2); rho_ee(0,0) = Scalar{1,0};
    std::vector<ObservableDef> obs;
    obs.push_back(make_operator_observable("rho_ee", rho_ee));

    StoppingCriteria sc;
    sc.min_trajectories = 200; sc.max_trajectories = 2000;
    sc.target_rel_sem = 0.05;

    EnsembleConfig ec;
    ec.global_seed = 42; ec.diag_level = DiagnosticLevel::None;
    ec.propagator.dt_initial = 1e-3;

    // Time-dependent systems use Simulation::make directly
    auto sim = Simulation::make<DenseTDOpenSystem, liquid::ode::RK4Stepper>(
        std::move(sys), std::move(obs), sc, ec);

    StateVector psi0(2); psi0[0] = Scalar{1,0};
    auto result = sim.run(psi0, 0.0, 20.0);

    printf("rho_ee(ss) = %.4f +/- %.4f  (N=%zu)\n",
        result.observables[0].mean,
        result.observables[0].sem,
        result.total_trajectories);
}
```

---

### Example 5: Parameter sweep

```cpp
#include "liquid/liquid.hpp"
using namespace liquid;

int main() {
    StateVector psi0(2); psi0[0] = Scalar{1,0};

    auto sweep = ParameterSweepBuilder{}
        .parameter_logrange("gamma", 0.1, 10.0, 8)
        .simulation_factory([](double gamma) {
            DenseOperator H(2), L(2), sz(2);
            H(0,0) = Scalar{0.5,0}; H(1,1) = Scalar{-0.5,0};
            L(1,0) = Scalar{std::sqrt(gamma),0};
            sz(0,0) = Scalar{1,0}; sz(1,1) = Scalar{-1,0};
            return SimulationBuilder{}
                .hamiltonian(std::move(H))
                .collapse_operator(std::move(L))
                .observe("sz", std::move(sz))
                .seed(42).dt(1e-3)
                .stop_at_sem(0.02)
                .min_trajectories(100).max_trajectories(5000)
                .build();
        })
        .initial_state(psi0)
        .time_interval(0.0, 2.0)
        .build();

    SweepResult result = sweep.run();
    result.print_table();
    result.save_csv("gamma_sweep.csv");
    result.save_json("gamma_sweep.json");
}
```

---

### Example 6: ML allocation policy

```cpp
#include "liquid/liquid.hpp"
using namespace liquid;
using namespace liquid::ensemble;
using namespace liquid::ensemble::ml;

int main() {
    DenseOperator H(2), L(2), sz(2);
    H(0,0)=Scalar{0.5,0}; H(1,1)=Scalar{-0.5,0};
    L(1,0)=Scalar{1.0,0};
    sz(0,0)=Scalar{1,0}; sz(1,1)=Scalar{-1,0};

    // ML policy trains online as trajectories complete
    AllocatorPolicy ml = make_ml_policy(
        42,     // seed
        8,      // hidden neurons per layer
        0.01    // learning rate
    );

    Simulation sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .seed(42).dt(1e-3)
        .stop_at_sem(0.01)
        .min_trajectories(100).max_trajectories(10000)
        .allocator_policy(std::move(ml))
        .build();

    StateVector psi0(2); psi0[0] = Scalar{1,0};
    auto result = sim.run(psi0, 0.0, 1.0);

    printf("<sz> = %.4f +/- %.4f  (N=%zu)\n",
        result.observables[0].mean,
        result.observables[0].sem,
        result.total_trajectories);
}
```

---

## Architecture overview

```
Layer 5  Application    SimulationBuilder, Simulation, ParameterSweep, output
Layer 4  Ensemble       RunningStatistics, ObservableAccumulator,
                        ConvergenceMonitor, AdaptiveAllocator (ML seam),
                        EnsembleManager (OpenMP batches)
Layer 3  Trajectory     TrajectoryState (CoreState + DiagnosticRecord),
                        MCWFPropagator (event-driven, resumable)
Layer 2  Physics        LindbladSet (dense+sparse, precomputed Γ),
                        OpenSystem (TI+TD × dense+sparse, precomputed H_eff)
Layer 1  Numerics       StateVector, DenseOperator, SparseOperator (CSR),
                        RK4Stepper, DormandPrince45 (FSAL, PI controller)
Layer 0  Foundation     types, RNGState (xoshiro256**), PropagatorConfig
```

**Dependency rule**: each layer depends only on layers below it. The ML
component attaches at the `AllocatorPolicy` seam in Layer 4 and reads
diagnostic data from Layers 3–4. It never touches Layers 0–2.

---

## Performance guide

### Solver choice

| Situation | Recommended |
|---|---|
| Reference / debugging | `RK4Stepper` (fixed step, predictable) |
| Production (default) | `DormandPrince45` (adaptive, 40–60× fewer RHS evals) |
| Very stiff systems | `DormandPrince45` with tight tolerances |

DOPRI45 uses FSAL (First Same As Last): the last stage of step n is reused
as the first stage of step n+1, giving 6 RHS evaluations per accepted step
instead of 7.

### Operator format

| Hilbert space | Recommended |
|---|---|
| N ≤ 20 | `DenseOperator` |
| N > 20, sparse H | `SparseOperator` |
| Large sparse (N > 100) | `SparseOperator` + `DormandPrince45` |

The JC model with N_fock=15 has dim=30. The Hamiltonian has O(N_fock) = 15
non-zeros out of 900 elements (1.7% sparsity). For N_fock=100 (dim=200):
0.5% sparsity → ~100× fewer multiplications per matvec vs dense.

### Parallelism

```cpp
EnsembleConfig ec;
ec.num_threads = 4;  // or: std::thread::hardware_concurrency()
```

Each thread owns its own `MCWFPropagator` (thread-local scratch space).
Threads accumulate into per-thread `ObservableAccumulator` instances.
Merge happens once per batch under a critical section — lock contention
is negligible for trajectory times > 1ms.

OpenMP speedup is meaningful when each trajectory takes ≥ 1ms (large
Hilbert spaces or long simulation times). For 2-level systems with T~1,
trajectories complete in ~10µs and parallelism overhead dominates.

### Convergence-based stopping

Always prefer `stop_at_sem` over a fixed `max_trajectories`. The framework
runs exactly as many trajectories as needed for the target accuracy:

```cpp
.stop_at_sem(0.01)          // 1% relative SEM
.min_trajectories(100)       // statistical floor
.max_trajectories(100000)    // safety ceiling
```

For a 2-level problem at T=1 with γ=1: σ²(σ_z) ≈ 0.5.
Target SEM = 1% of |mean| ≈ 0.01 × 0.27 ≈ 0.003.
Required N ≈ σ²/SEM² ≈ 0.5/9×10⁻⁶ ≈ 55,000.

---

## Extending LiQuID

### Custom stepper

Any class matching the stepper concept works:

```cpp
struct MyCustomStepper {
    // Required interface:
    template<typename RHSFn>
    liquid::ode::StepResult try_step(
        Real t, const StateVector& psi_in, StateVector& psi_out,
        Real dt_suggested, Real t_max, RHSFn&& rhs_fn);

    void reset(Real dt_initial) noexcept;
};

// Use it:
auto sim = Simulation::make<DenseOpenSystem, MyCustomStepper>(
    sys, obs, sc, ec);
```

### Custom allocation policy

```cpp
// AllocatorPolicy = std::function<AllocationDecision(const EnsembleSummary&)>
// EnsembleSummary contains: trajectories_completed, current_rel_sem,
//                           wall_time_elapsed, completed_summaries
// AllocationDecision contains: action, new_seed, new_traj_id

AllocatorPolicy rare_event_policy = [counter = TrajId{0}]
    (const EnsembleSummary& s) mutable -> AllocationDecision {
    // Example: alternate between two seed families
    const TrajId id = counter++;
    const Seed seed = (id % 2 == 0)
        ? 0xdeadbeefULL ^ id
        : 0xcafebabeULL ^ id;
    return AllocationDecision{AllocAction::SpawnNew, seed, id};
};
```

### New system types

Implement `apply_Heff(Real t, const StateVector&, StateVector&)`,
`jump_probabilities(...)`, `apply_jump(...)`, `hilbert_dim()`,
`num_channels()`. Then use `Simulation::make<YourSystem, StepperType>`.

---

## Test suite

```bash
cmake -B build && cmake --build build
ctest --test-dir build --output-on-failure
```

| Suite | Tests | Validates |
|---|---|---|
| test_rng | 8 | xoshiro256** uniformity, reproducibility, serialization |
| test_linalg | 16 | StateVector, DenseOperator, inner products |
| test_opensystem | 8 | H_eff precomputation, apply_Heff, jump operators |
| test_trajectory_state | 10 | Invariants, factory, diagnostics |
| test_running_statistics | 13 | Welford algorithm, parallel merge |
| test_ensemble | 11 | ObservableAccumulator, ConvergenceMonitor, SimulationBuilder |
| test_dopri45 | 5 | FSAL, PI controller, physics correctness |
| test_sparse | 13 | CSR format, matvec, adjoint, SparseOpenSystem |
| test_ml_policy | 15 | Feature extraction, MLP training, policy interface |
| test_parameter_sweep | 11 | Builder, range generation, CSV/JSON output |
| val_two_level_decay | 5 | KS test on jump times, ensemble vs analytic |
| val_driven_two_level | 3 | Rabi oscillations, steady-state formula |
| val_ensemble_convergence | 4 | SEM ∝ 1/√N, convergence stopping |
| val_cavity_qed | 4 | JC excitation conservation, vacuum decay |
| val_smart_stopping | 5 | Wall-clock budget, SEM scaling |
| val_ml_policy | 6 | Online training, physics correctness |
| val_parameter_sweep | 4 | Sweep physics, CSV round-trip |
| **Total** | **141** | |

---

## Known limitations

**No Eigen dependency**: the linear algebra backend is hand-implemented.
For large dense systems (N > 200), linking Eigen and replacing
`DenseOperator`'s `apply_add` with `Eigen::MatrixXcd::operator*` will give
BLAS-level performance. The interface is frozen — the swap is one file.

**No GPU support**: the architecture supports future GPU backends via a
custom stepper and system type. The `EnsembleManager` threading model maps
to CUDA streams. Not implemented.

**No MPI**: `RunningStatistics::merge()` supports parallel reduction.
Distributing the work queue across MPI ranks requires adding an MPI
communicator at the `EnsembleManager` level. The data structures are
serializable; the integration point is defined.

**No Python bindings**: the `AllocatorPolicy` seam accepts any callable,
including a Python function via pybind11. Bindings are not included.

**Time-dependent H in SimulationBuilder**: `SimulationBuilder` only
supports time-independent dense Hamiltonians. For time-dependent systems
use `Simulation::make<DenseTDOpenSystem, StepperType>` directly.

**Jump location accuracy**: Phase 1 locates jump times at the end of the
step where the norm crossed the threshold (no bisection). This introduces
a systematic error of O(dt) in jump times. For most observables measured at
`t_final` this is negligible. For jump-time statistics at high accuracy,
replace `JumpDetectorType::Threshold` with `LinearInterpBisect` (stubbed,
Phase 3 roadmap item).

## Citation

If you use LiQuID in your research, please cite both the software and the dataset.

### Dataset

Mohammed, Mohammed Maher Abdelrahim (2026), "LiQuID — Lindblad Quantum Integrated Dynamics", Mendeley Data, V1, doi:10.17632/zm9s954bnn.1
