// benchmarks/bench_adaptive_stopping.cpp
// -------------------------------------------------------------------------------
// Task — Validate the adaptive SEM-based stopping criterion.
//
// Question answered:
//   Does LiQuID stop at approximately the theoretically optimal number of
//   trajectories, N_optimal = (σ / (ε·|μ|))²?
//
// Physical system:
//   H = 0  (zero Hamiltonian — pure spontaneous decay)
//   L = √γ σ₋,   γ = 1
//   Observable: ⟨σ_z(T=1)⟩
//   Initial state: |↑⟩  (psi0 = [1, 0]ᵀ)
//
// Analytic reference:  ⟨σ_z(T)⟩ = 2 e^{−γT} − 1  = 2e⁻¹ − 1 ≈ −0.2642
//
// Protocol:
//   1. Reference run: N_ref = 1,000,000 trajectories.
//      Estimate population variance  σ² = SEM² × N_ref.
//      Store  mean_ref, variance_ref.
//
//   2. For each relative precision target ε ∈ {10%, 5%, 2%, 1%, 0.5%, 0.2%, 0.1%}:
//        N_optimal = ceil( σ² / (ε · |μ|)² )
//        Run LiQuID with .stop_at_sem(ε)  (3 replicas, take median N).
//        Compute  efficiency = N_optimal / N_adaptive.
//
//   3. Write bench_adaptive_stopping.csv
//   4. Write bench_adaptive_stopping_plot.py  (produces the figure)
//
// Success criterion: efficiency ∈ [0.9, 1.1] for all targets.
// -------------------------------------------------------------------------------

#include "liquid/liquid.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace liquid;
using namespace liquid::ensemble;

// -- helpers ---------------------------------------------------------

// Build a simulation with H=0, L=σ₋, observable σ_z.
// stop_at_sem sets the relative-SEM target; setting it impossibly small
// (1e-15) combined with min_N = max_N = N forces a fixed-N run.
static Simulation make_decay_sim(double rel_sem_target,
                                  std::size_t min_N,
                                  std::size_t max_N,
                                  Seed        seed)
{
    // H = 0
    DenseOperator H(2);   // default-constructed: all zeros

    // L = σ₋  (γ = 1, already absorbed)
    DenseOperator L(2);
    L(1, 0) = Scalar{1.0, 0.0};

    // σ_z = diag(+1, −1)
    DenseOperator sz(2);
    sz(0, 0) = Scalar{ 1.0, 0.0};
    sz(1, 1) = Scalar{-1.0, 0.0};

    return SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sz", std::move(sz))
        .seed(seed)
        .dt(1e-3)
        .stop_at_sem(rel_sem_target)
        .min_trajectories(min_N)
        .max_trajectories(max_N)
        .build();
}

// Median of a sorted length-3 array (already sorted by caller).
template<typename T>
static inline T median3(T a[3]) { return a[1]; }

// -- main -----------------------------------------------------------------------------

int main()
{
    constexpr double      T     = 1.0;
    constexpr std::size_t N_REF = 50'000;

    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};   // |↑⟩
    psi0[1] = Scalar{0.0, 0.0};

    // -- Step 1: Reference run ─────────────────────────────────────────────
    std::printf("|=========================================================|\n");
    std::printf("|  Task  : Adaptive stopping validation                  |\n");
    std::printf("|  System: two-level decay, H=0, γ=1, T=1                 |\n");
    std::printf("|=========================================================|\n\n");
				  
    std::printf("Step 1: Reference run  N_ref = %zu\n", N_REF);
    std::printf("  (stop_at_sem = 1e-15, min = max = N_ref → fixed-N run)\n");

    auto t_ref0 = std::chrono::steady_clock::now();
    auto ref_sim = make_decay_sim(
        1e-15,          // impossible target → always runs to max_N
        N_REF, N_REF,   // min == max → fixed N
        42ULL
    );
    auto ref_result = ref_sim.run(psi0, 0.0, T);
    double t_ref = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ref0).count();

    const double mean_ref = ref_result.observables[0].mean;
    const double sem_ref  = ref_result.observables[0].sem;
    // Population variance estimate: σ² = SEM² × N
    const double var_ref  = sem_ref * sem_ref * static_cast<double>(N_REF);
    const double std_ref  = std::sqrt(var_ref);

    // Analytic check
    const double mean_analytic = 2.0 * std::exp(-1.0) - 1.0;
    const double analytic_err  = std::abs(mean_ref - mean_analytic)
                                  / std::abs(mean_analytic) * 100.0;

    std::printf("\n  ⟨σ_z(T=1)⟩  = %+.8f   (analytic: %+.8f,  error: %.3f%%)\n",
        mean_ref, mean_analytic, analytic_err);
    std::printf("  SEM_ref      = %.3e\n", sem_ref);
    std::printf("  σ (pop. std) = %.6f\n", std_ref);
    std::printf("  Var (pop.)   = %.6f\n", var_ref);
    std::printf("  Wall time    = %.1f s\n\n", t_ref);

    // -- Step 2: Adaptive runs ---------------------------------------------
    const double targets[] = {0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    const int    n_targets  = static_cast<int>(sizeof targets / sizeof targets[0]);

    std::printf("Step 2: Adaptive stopping  (3 replicas per target, median reported)\n\n");

    // Table header
    std::printf("  %-10s  %-12s  %-12s  %-10s  %-14s  %-9s\n",
        "ε (rel)", "N_optimal", "N_adaptive", "Efficiency",
        "achieved ε", "wall (s)");
    const std::string sep(74, '─');
    std::printf("  %s\n", sep.c_str());

    // CSV
    FILE* csv = std::fopen("bench_adaptive_stopping.csv", "w");
    if (!csv) {
        std::fprintf(stderr, "ERROR: cannot open bench_adaptive_stopping.csv\n");
        return 1;
    }
    std::fprintf(csv,
        "target_rel_sem,N_optimal,N_adaptive,efficiency,"
        "achieved_rel_sem,wall_s,mean,variance_ref\n");

    // Arrays for plot script
    std::vector<std::size_t> v_Nopt, v_Nadapt;
    std::vector<double>      v_target, v_eff, v_sem_achiev;

    for (int ti = 0; ti < n_targets; ++ti) {
        const double eps = targets[ti];

        // N_optimal = ceil( σ² / (ε · |μ|)² )
        const double denom   = eps * std::abs(mean_ref);
        const std::size_t N_opt = static_cast<std::size_t>(
            std::ceil(var_ref / (denom * denom)));

        // 3 replicas
        std::size_t ns[3];
        double walls[3], sems_ach[3];

        for (int rep = 0; rep < 3; ++rep) {
            auto t0  = std::chrono::steady_clock::now();
            auto sim = make_decay_sim(
                eps,
                /*min_N=*/50,
                /*max_N=*/5'000'000,
                static_cast<Seed>(1000 + ti * 100 + rep * 7)
            );
            auto result = sim.run(psi0, 0.0, T);
            walls[rep]   = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            ns[rep]      = result.total_trajectories;
            sems_ach[rep]= result.observables[0].rel_sem;
        }

        std::sort(ns,       ns + 3);
        std::sort(walls,    walls + 3);
        std::sort(sems_ach, sems_ach + 3);

        const std::size_t N_adapt    = median3(ns);
        const double      wall_med   = median3(walls);
        const double      sem_achiev = median3(sems_ach);
        const double      efficiency = (N_adapt > 0)
            ? static_cast<double>(N_opt) / static_cast<double>(N_adapt)
            : 0.0;

        // Mark rows that pass the success criterion
        const char* flag = (efficiency >= 0.9 && efficiency <= 1.1) ? "✓" : "✗";

        std::printf("  %-10.3f  %-12zu  %-12zu  %-10.3f  %-14.4f  %-9.2f  %s\n",
            eps, N_opt, N_adapt, efficiency, sem_achiev, wall_med, flag);

        std::fprintf(csv,
            "%.6f,%zu,%zu,%.6f,%.6f,%.4f,%.8f,%.8f\n",
            eps, N_opt, N_adapt, efficiency, sem_achiev,
            wall_med, mean_ref, var_ref);

        v_Nopt.push_back(N_opt);
        v_Nadapt.push_back(N_adapt);
        v_target.push_back(eps);
        v_eff.push_back(efficiency);
        v_sem_achiev.push_back(sem_achiev);
    }

    std::fclose(csv);
    std::printf("  %s\n", sep.c_str());
    std::printf("  ✓ = efficiency ∈ [0.9, 1.1]\n\n");

    // -- Step 3: Summary statistics --------------------------------------
    std::printf("Step 3: Summary\n");
    std::printf("  Reference mean  : %+.8f\n", mean_ref);
    std::printf("  Analytic mean   : %+.8f\n", mean_analytic);
    std::printf("  Population σ    : %.6f\n",  std_ref);
    std::printf("  Population σ²   : %.6f\n",  var_ref);

    int n_pass = 0;
    for (double e : v_eff)
        if (e >= 0.9 && e <= 1.1) ++n_pass;
    std::printf("  Targets passing [0.9, 1.1]: %d / %d\n\n", n_pass, n_targets);
    std::printf("Saved: bench_adaptive_stopping.csv\n\n");
    std::printf("Interpretation:\n");
    std::printf("  Efficiency ≈ 1.0  → stopping fired at statistically optimal N.\n");
    std::printf("  Efficiency < 0.9  → LiQuID ran too many trajectories (conservative).\n");
    std::printf("  Efficiency > 1.1  → LiQuID stopped early (aggressive).\n");

    return 0;
}
