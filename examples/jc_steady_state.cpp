// examples/jc_steady_state.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Example: Driven-dissipative Jaynes-Cummings — steady-state photon number
//
// Scientific question:
//   How does the steady-state intracavity photon number <a†a>_ss depend
//   on the atom-cavity coupling g in a coherently driven cavity?
//
// Physical setup (rotating frame at drive frequency ω_d = ω):
//   H = Δ_c a†a + Δ_a/2 σ_z + g(a†σ_- + aσ_+) + ε(a + a†)
//   L1 = √κ a        (cavity photon loss)
//   L2 = √γ σ_-      (atomic decay)
//
//   Resonance: Δ_c = Δ_a = 0  (drive on resonance)
//   Drive amplitude: ε = 0.2κ  (weak driving)
//   Sweep: g/κ ∈ [0.05, 5.0] (log-spaced, 10 points)
//
// Expected physics:
//   - At g=0 (empty cavity): <n>_ss = ε²/κ² = 0.04 photons (linear response)
//   - At g >> κ: strong coupling splits the resonance → photon blockade
//     reduces <n>_ss dramatically
//   - The crossover g ~ κ marks the onset of cavity QED effects
//
// Build:
//   g++ -std=c++17 -O2 -DNDEBUG -Iinclude \
//       examples/jc_steady_state.cpp src/core/rng.cpp -o jc_steady_state
//
// Runtime: ~60-120s
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/liquid.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace liquid;
using namespace liquid::ode;
using namespace liquid::ensemble;

// ── Physical parameters ───────────────────────────────────────────────────────

static constexpr int    N_FOCK  = 12;    // Fock space truncation
static constexpr double KAPPA   = 1.0;   // cavity decay (sets time scale)
static constexpr double GAMMA_A = 0.1;   // atomic decay
static constexpr double EPSILON = 0.2;   // drive amplitude (weak: ε << κ)
static constexpr double T_SS    = 30.0;  // run time >> 1/κ

// ── Basis: |n, spin> → index 2n+s  (s=0:excited, s=1:ground) ────────────────

static Idx jc_idx(int n, int s) { return static_cast<Idx>(2*n + s); }
static Dim jc_dim() { return static_cast<Dim>(2 * N_FOCK); }

// ── Operator factories ────────────────────────────────────────────────────────

// H = g(a†σ_- + aσ_+) + ε(a + a†)
// At resonance Δ_c = Δ_a = 0: no diagonal terms from free evolution.
static SparseOperator make_H(double g) {
    const Dim dim = jc_dim();
    std::vector<Triplet> trips;
    trips.reserve(4 * N_FOCK);

    for (int n = 0; n < N_FOCK; ++n) {
        // JC coupling: g·a†σ_- : |n,e> → |n+1,g>  (create photon, lower atom)
        if (n + 1 < N_FOCK) {
            const double v_jc = g * std::sqrt(static_cast<double>(n + 1));
            trips.push_back({jc_idx(n+1,1), jc_idx(n,0), Scalar{v_jc, 0}});
            trips.push_back({jc_idx(n,0), jc_idx(n+1,1), Scalar{v_jc, 0}});
        }

        // Drive: ε·a†  :  |n,s> → |n+1,s>  with amplitude ε·sqrt(n+1)
        if (n + 1 < N_FOCK) {
            const double v_dr = EPSILON * std::sqrt(static_cast<double>(n + 1));
            trips.push_back({jc_idx(n+1,0), jc_idx(n,0), Scalar{v_dr, 0}});
            trips.push_back({jc_idx(n+1,1), jc_idx(n,1), Scalar{v_dr, 0}});
            // h.c.: ε·a
            trips.push_back({jc_idx(n,0), jc_idx(n+1,0), Scalar{v_dr, 0}});
            trips.push_back({jc_idx(n,1), jc_idx(n+1,1), Scalar{v_dr, 0}});
        }
    }
    return SparseOperator(dim, std::move(trips));
}

static SparseOperator make_L_cavity() {
    const Dim dim = jc_dim();
    std::vector<Triplet> trips;
    for (int n = 1; n < N_FOCK; ++n) {
        const double v = std::sqrt(KAPPA * static_cast<double>(n));
        trips.push_back({jc_idx(n-1,0), jc_idx(n,0), Scalar{v, 0}});
        trips.push_back({jc_idx(n-1,1), jc_idx(n,1), Scalar{v, 0}});
    }
    return SparseOperator(dim, std::move(trips));
}

static SparseOperator make_L_atomic() {
    const Dim dim = jc_dim();
    std::vector<Triplet> trips;
    const double v = std::sqrt(GAMMA_A);
    for (int n = 0; n < N_FOCK; ++n)
        trips.push_back({jc_idx(n,1), jc_idx(n,0), Scalar{v, 0}});
    return SparseOperator(dim, std::move(trips));
}

// ── Observables ───────────────────────────────────────────────────────────────

static ObservableDef photon_obs() {
    const Dim dim = jc_dim();
    std::vector<Triplet> trips;
    for (int n = 0; n < N_FOCK; ++n) {
        trips.push_back({jc_idx(n,0), jc_idx(n,0), Scalar{(double)n, 0}});
        trips.push_back({jc_idx(n,1), jc_idx(n,1), Scalar{(double)n, 0}});
    }
    SparseOperator n_op(dim, std::move(trips));

    return {"photon_number",
            [op = std::move(n_op)](const StateVector& psi) -> Real {
                const Real ns = psi.norm_sq();
                return ns > 1e-30
                    ? sparse_expectation(op, psi).real() / ns
                    : 0.0;
            }};
}

static ObservableDef excited_obs() {
    const Dim dim = jc_dim();
    std::vector<Triplet> trips;
    for (int n = 0; n < N_FOCK; ++n)
        trips.push_back({jc_idx(n,0), jc_idx(n,0), Scalar{1.0, 0}});
    SparseOperator pe_op(dim, std::move(trips));

    return {"excited_pop",
            [op = std::move(pe_op)](const StateVector& psi) -> Real {
                const Real ns = psi.norm_sq();
                return ns > 1e-30
                    ? sparse_expectation(op, psi).real() / ns
                    : 0.0;
            }};
}

// ── Simulation factory ────────────────────────────────────────────────────────

static Simulation make_sim(double g_over_kappa) {
    const double g = g_over_kappa * KAPPA;

    SparseOperator H  = make_H(g);
    SparseOperator L1 = make_L_cavity();
    SparseOperator L2 = make_L_atomic();

    std::vector<SparseOperator> ops;
    ops.push_back(std::move(L1));
    ops.push_back(std::move(L2));

    SparseOpenSystem sys(std::move(H),
                        LindbladSet<SparseTag>(std::move(ops)));

    std::vector<ObservableDef> obs;
    obs.push_back(photon_obs());
    obs.push_back(excited_obs());

    StoppingCriteria sc;
    sc.min_trajectories = 300;
    sc.max_trajectories = 3000;
    sc.target_rel_sem   = 0.05;  // 5% relative SEM

    EnsembleConfig ec;
    ec.global_seed           = 42;
    ec.diag_level            = DiagnosticLevel::None;
    ec.propagator.dt_initial = 5e-3;

    return Simulation::make_sparse_dopri45(
        std::move(sys), std::move(obs), sc, ec);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf(
        "╔═══════════════════════════════════════════════════════════════╗\n"
        "║  LiQuID: Driven-Dissipative Jaynes-Cummings Steady State      ║\n"
        "╠═══════════════════════════════════════════════════════════════╣\n"
        "║  N_fock=%-2d  κ=%.1f  γ=%.2f  ε=%.2f  T=%.0f                       ║\n"
        "║  Expected: <n>_ss → ε²/κ²=%.3f at g=0 (linear cavity)        ║\n"
        "╚═══════════════════════════════════════════════════════════════╝\n\n",
        N_FOCK, KAPPA, GAMMA_A, EPSILON, T_SS,
        EPSILON*EPSILON/(KAPPA*KAPPA));

    // Initial state: vacuum + ground atom |n=0, g>
    // (correct: the steady state is reached regardless of initial state)
    StateVector psi0(jc_dim());
    psi0[jc_idx(0, 1)] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter_logrange("g_over_kappa", 0.05, 5.0, 10)
        .simulation_factory(make_sim)
        .initial_state(psi0)
        .time_interval(0.0, T_SS)
        .build();

    std::printf("Running sweep: 10 log-spaced values of g/κ ∈ [0.05, 5.0]\n");
    std::printf("(Expect non-zero photon numbers — drive keeps system active)\n\n");

    SweepResult result = sweep.run();

    result.print_table();

    std::printf("Physics check:\n");
    std::printf("  Linear cavity (g=0): <n>_ss = ε²/κ² = %.4f\n",
        EPSILON*EPSILON);
    std::printf("  First point (g/κ=%.3f): <n>_ss = %.4f ± %.4f\n",
        result.points.front().param_value,
        result.points.front().result.observables[0].mean,
        result.points.front().result.observables[0].sem);
    std::printf("  Last point  (g/κ=%.2f): <n>_ss = %.4f ± %.4f\n",
        result.points.back().param_value,
        result.points.back().result.observables[0].mean,
        result.points.back().result.observables[0].sem);
    std::printf("  Photon blockade: expect <n> < %.4f for large g\n\n",
        EPSILON*EPSILON);

    result.save_csv("jc_steady_state.csv");
    result.save_json("jc_steady_state.json");
    std::printf("Saved: jc_steady_state.csv, jc_steady_state.json\n");

    return 0;
}
