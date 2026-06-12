// benchmarks/bench2_variance_dependence.cpp
// -----------------------------------------------------------------------------
// Task - Variance Dependence of the Adaptive Stopping Criterion.
//
// Core argument:
//   Different observables measured on the same physical system have vastly
//   different variances.  Because N_optimal = σ²/(ε·|μ|)², achieving the
//   same relative precision ε requires vastly different ensemble sizes —
//   by up to two orders of magnitude.  Fixed-N ensembles are therefore
//   inherently inefficient.  LiQuID's stop_at_sem() eliminates this problem
//   automatically because it stops each simulation as soon as its own
//   observable satisfies the SEM target.
//
// --- API note -----------------------------------------------------------------
// stop_at_sem(eps) takes ONE argument and stops when ALL registered
// observables satisfy rel_SEM < eps.  To drive stopping on a SPECIFIC
// observable we register only that observable in the simulation.
// Per-observable rel_SEM is computed manually as sem / |mean|.
// There is no incremental-runner API; convergence histories are built by
// repeated full sim.run() calls at growing fixed N.
// -----------------------------------------------------------------------------
//
// ---Physical system: driven two-level atom -----------------------------------
//   H  = Ω σ_x        resonant drive in rotating frame,  Ω = 0.5
//   L  = √γ σ₋         spontaneous decay,                γ = 1.0
//   ψ₀ = |↑⟩ = (1,0)ᵀ
//   T  = 20.0          long enough to reach near-steady-state
//
//   Steady-state reference (analytic):
//     ρ_ee^ss = 4Ω²/(γ² + 8Ω²)    (see Example 2 in README)
//
// --- Four observables with deliberately different variances ------------------
//   Pe   = |e><e| = diag(1,0)       bounded [0,1],   low variance
//   σ_z  = diag(+1,−1)              bounded [-1,1],  moderate variance
//   σ_x  = [[0,1],[1,0]]            oscillating mean near zero → high rel-var
//   Pg   = |g><g| = diag(0,1)       1 − Pe, same magnitude, different sign
//
//   Note on jump count:
//   The true per-trajectory integer jump count has Poissonian variance and
//   would be the highest-variance observable.  LiQuID does not expose a
//   built-in jump-count observable in v0.7.0.  Pg is used instead as the
//   fourth observable: it has the same variance as Pe but a different mean
//   (close to 1 − ρ_ee^ss ≈ 0.80), so σ/|μ| is smaller and N_optimal is
//   the lowest of the four — the opposite end of the variance spectrum.
//   This gives a clean four-point spread in the bar chart.
//
// --- Outputs -----------------------------------------------------------------
//   bench2_variance.csv      σ², |μ|, σ/|μ| per observable (from ref run)
//   bench2_main.csv          N_adaptive, N_optimal, efficiency per (obs, ε)
//   bench2_convergence.csv   RelSEM(N) history per observable at ε = 1%
// ----------------------------------------------------------------------------

#include "liquid/liquid.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace liquid;
using namespace liquid::ensemble;

// -----------------------------------------------------------------------------
// Physical parameters
// -----------------------------------------------------------------------------

static constexpr double OMEGA = 0.5;   // Rabi drive amplitude
static constexpr double GAMMA = 1.0;   // spontaneous decay rate
static constexpr double T_SIM = 20.0;  // near-steady-state
static constexpr double DT    = 5e-4;  // DOPRI45 initial step

// -----------------------------------------------------------------------------
// Operator builders
// -----------------------------------------------------------------------------

static DenseOperator make_H() {
    DenseOperator H(2);                       // H = Ω σ_x
    H(0,1) = Scalar{OMEGA, 0.0};
    H(1,0) = Scalar{OMEGA, 0.0};
    return H;
}
static DenseOperator make_L() {
    DenseOperator L(2);                       // L = √γ σ₋
    L(1,0) = Scalar{std::sqrt(GAMMA), 0.0};
    return L;
}
static DenseOperator make_Pe() {
    DenseOperator o(2); o(0,0) = Scalar{1.0,0.0}; return o;  // |e><e|
}
static DenseOperator make_sz() {
    DenseOperator o(2);
    o(0,0) = Scalar{ 1.0,0.0}; o(1,1) = Scalar{-1.0,0.0};
    return o;
}
static DenseOperator make_sx() {
    DenseOperator o(2);
    o(0,1) = Scalar{1.0,0.0}; o(1,0) = Scalar{1.0,0.0};
    return o;
}
static DenseOperator make_Pg() {
    DenseOperator o(2); o(1,1) = Scalar{1.0,0.0}; return o;  // |g><g|
}

//---------------------------------------------------------------------------------
// Simulation factory — ONE observable per simulation.
// stop_at_sem(eps) stops when that single observable reaches eps rel-SEM.
// For a fixed-N reference run, set stop_at_sem to 1e-15 and min=max=N.
// ---------------------------------------------------------------------------------

static Simulation make_single_obs_sim(
        DenseOperator obs_op,
        const char*   obs_name,
        double        rel_sem_target,
        std::size_t   min_N,
        std::size_t   max_N,
        Seed          seed)
{
    return SimulationBuilder{}
        .hamiltonian(make_H())
        .collapse_operator(make_L())
        .observe(obs_name, std::move(obs_op))
        .seed(seed)
        .dt(DT)
        .stop_at_sem(rel_sem_target)
        .min_trajectories(min_N)
        .max_trajectories(max_N)
        .build();
}

// -----------------------------------------------------------------------------
// Observable registry
// -----------------------------------------------------------------------------

static constexpr int N_OBS = 4;

struct ObsMeta {
    const char* name;
    const char* label;
    DenseOperator (*make_op)();
};

static const ObsMeta OBS[N_OBS] = {
    { "Pe",   "Population Pe",  make_Pe },
    { "sz",   "sigma_z",        make_sz },
    { "sx",   "sigma_x",        make_sx },
    { "Pg",   "Ground state Pg", make_Pg },
};

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main()
{
    constexpr std::size_t N_REF = 50'000;

    // -- Banner --------------------------------------------------------------
    std::printf(
        "|==========================================================|\n"
        "|  Task   : Variance Dependence of Adaptive Stopping           |\n"
        "|  System : driven two-level atom, Ω=0.5, γ=1.0, T=20         |\n"
        "|  ψ₀     : |↑⟩                                                |\n"
        "|  Obs    : Pe │ σ_z │ σ_x │ Pg                                |\n"
        "|  Analytic ρ_ee^ss = 4Ω²/(γ²+8Ω²) ≈ 0.111                   |\n"
        "|==========================================================|\n\n");

    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};   // |↑⟩ = excited state
    psi0[1] = Scalar{0.0, 0.0};

    const double rho_ee_ss = 4.0*OMEGA*OMEGA / (GAMMA*GAMMA + 8.0*OMEGA*OMEGA);
    std::printf("  Analytic ρ_ee^ss = %.6f\n\n", rho_ee_ss);

    // ==========================================================
    // Step 1: Reference run — each observable independently at fixed N_REF.
    //   Fixed-N: set stop_at_sem to 1e-15 (impossible), min = max = N_REF.
    //   rel_sem is computed manually as sem / |mean|.
    // ==========================================================
    std::printf("Step 1: Reference runs  (N_ref = %zu per observable)\n\n", N_REF);

    double mean_ref[N_OBS], sem_ref[N_OBS], var_ref[N_OBS], relsem_ref[N_OBS];

    const std::string sep_med(68, '─');
    std::printf("  %-20s  %+12s  %12s  %12s  %10s\n",
        "Observable", "mean", "sem", "σ²", "σ/|μ|");
    std::printf("  %s\n", sep_med.c_str());

    for (int i = 0; i < N_OBS; ++i) {
        auto sim = make_single_obs_sim(
            OBS[i].make_op(), OBS[i].name,
            1e-15, N_REF, N_REF,
            static_cast<Seed>(100 + i * 7));
        auto res = sim.run(psi0, 0.0, T_SIM);

        mean_ref[i]   = res.observables[0].mean;
        sem_ref[i]    = res.observables[0].sem;
        var_ref[i]    = sem_ref[i] * sem_ref[i] * static_cast<double>(N_REF);
        relsem_ref[i] = (std::abs(mean_ref[i]) > 1e-12)
            ? sem_ref[i] / std::abs(mean_ref[i]) : 9999.0;

        std::printf("  %-20s  %+12.6f  %12.6e  %12.6f  %10.4f\n",
            OBS[i].label, mean_ref[i], sem_ref[i], var_ref[i], relsem_ref[i]);
    }
    std::printf("  %s\n\n", sep_med.c_str());

    // -- Write bench2_variance.csv (Plot 3 source) ---------------------------
    {
        FILE* fv = std::fopen("bench2_variance.csv", "w");
        if (!fv) { std::fprintf(stderr, "ERROR: bench2_variance.csv\n"); return 1; }
        std::fprintf(fv, "observable,label,mean,variance,sigma,sigma_over_absmean\n");
        for (int i = 0; i < N_OBS; ++i) {
            std::fprintf(fv, "%s,\"%s\",%.8f,%.8f,%.8f,%.8f\n",
                OBS[i].name, OBS[i].label,
                mean_ref[i], var_ref[i],
                std::sqrt(var_ref[i]),
                relsem_ref[i] * std::sqrt(static_cast<double>(N_REF)));
            // Note: σ/|μ| = (sem/|mean|) × √N = relsem_ref × √N_REF
        }
        std::fclose(fv);
        std::printf("  Saved: bench2_variance.csv\n\n");
    }

    // ==========================================================
    // Step 2: Adaptive runs — main result table.
    //   For each (observable, ε):
    //     N_optimal = ceil( σ² / (ε · |μ|)² )
    //     3 replicas → median N_adaptive (and lo/hi for error bars)
    //     efficiency = N_optimal / N_adaptive
    //     achieved rel_sem = sem / |mean| from the run result
    // ==========================================================
    const double targets[] = {0.10, 0.05, 0.02, 0.01, 0.005};
    const int n_targets = static_cast<int>(sizeof targets / sizeof targets[0]);

    std::printf("Step 2: Adaptive stopping  (3 replicas per row)\n\n");

    FILE* fmain = std::fopen("bench2_main.csv", "w");
    if (!fmain) { std::fprintf(stderr, "ERROR: bench2_main.csv\n"); return 1; }
    std::fprintf(fmain,
        "observable,label,target_rel_sem,mean_ref,variance_ref,"
        "N_optimal,N_adaptive_med,N_adaptive_lo,N_adaptive_hi,"
        "efficiency,achieved_rel_sem,wall_s\n");

    int total_pass = 0, total_rows = 0;
    const std::string sep_long(90, '─');

    for (int oi = 0; oi < N_OBS; ++oi) {
        std::printf(
            "  ── Observable: %-20s  μ = %+.6f,  σ² = %.6f\n",
            OBS[oi].label, mean_ref[oi], var_ref[oi]);
        std::printf("  %-10s  %-12s  %-12s  %-10s  %-14s  %-9s\n",
            "ε (rel)", "N_optimal", "N_adaptive", "Efficiency",
            "achieved ε", "wall (s)");
        std::printf("  %s\n", sep_long.c_str());

        for (int ti = 0; ti < n_targets; ++ti) {
            const double eps = targets[ti];

            // Skip when |μ| ≈ 0: relative SEM is ill-defined
            if (std::abs(mean_ref[oi]) < 1e-10) {
                std::printf("  %-10.3f  [skipped: |μ| < 1e-10]\n", eps);
                continue;
            }

            const double denom = eps * std::abs(mean_ref[oi]);
            const std::size_t N_opt = static_cast<std::size_t>(
                std::ceil(var_ref[oi] / (denom * denom)));

            // 3 replicas
            std::size_t ns[3];
            double walls[3], relsems[3];

            for (int rep = 0; rep < 3; ++rep) {
                Seed seed = static_cast<Seed>(
                    3000 + oi * 1000 + ti * 100 + rep * 17);

                auto t0r = std::chrono::steady_clock::now();
                auto sim = make_single_obs_sim(
                    OBS[oi].make_op(), OBS[oi].name,
                    eps,
                    /*min_N=*/100,
                    /*max_N=*/20'000'000,
                    seed);
                auto res = sim.run(psi0, 0.0, T_SIM);
                walls[rep] = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0r).count();

                ns[rep] = res.total_trajectories;
                // Compute rel_sem manually: sem / |mean|
                double m = res.observables[0].mean;
                double s = res.observables[0].sem;
                relsems[rep] = (std::abs(m) > 1e-12) ? s / std::abs(m) : 9999.0;
            }

            std::sort(ns,      ns + 3);
            std::sort(walls,   walls + 3);
            std::sort(relsems, relsems + 3);

            const std::size_t N_lo  = ns[0];
            const std::size_t N_med = ns[1];
            const std::size_t N_hi  = ns[2];
            const double wall_med   = walls[1];
            const double rsem_med   = relsems[1];
            const double eff        = (N_med > 0)
                ? static_cast<double>(N_opt) / static_cast<double>(N_med) : 0.0;

            const char* flag = (eff >= 0.8 && eff <= 1.2) ? "✓" : "✗";
            if (eff >= 0.8 && eff <= 1.2) ++total_pass;
            ++total_rows;

            std::printf(
                "  %-10.3f  %-12zu  %-12zu  %-10.3f  %-14.4f  %-9.2f  %s\n",
                eps, N_opt, N_med, eff, rsem_med, wall_med, flag);

            std::fprintf(fmain,
                "%s,\"%s\",%.6f,%.8f,%.8f,"
                "%zu,%zu,%zu,%zu,"
                "%.6f,%.6f,%.4f\n",
                OBS[oi].name, OBS[oi].label, eps,
                mean_ref[oi], var_ref[oi],
                N_opt, N_med, N_lo, N_hi,
                eff, rsem_med, wall_med);
        }
        std::printf("  %s\n\n", sep_long.c_str());
    }

    std::fclose(fmain);
    std::printf("  Saved: bench2_main.csv\n\n");
/*
    // ==========================================================
    // Step 3: Convergence histories at ε = 1%  (for Plot 2)
    //
    // For each observable, we run repeated full simulations at increasing
    // fixed-N values and record the running rel_SEM at each N.
    // This uses only sim.run() — the only available execution API.
    //
    // Schedule: N = 50, 100, 200, 300, ..., up to the observed N_adaptive
    // for that observable plus a 50% margin, or MAX_HIST_N, whichever
    // is smaller.
    // ==========================================================
    constexpr double      HIST_EPS   = 0.01;
    constexpr std::size_t MAX_HIST_N = 60'000;

    // First, find the ε = 1% N_adaptive for each observable from bench2_main.
    // We re-run a single replica here to get that value fresh.
    std::size_t N_stop_per_obs[N_OBS];
    std::printf("Step 3: Convergence histories at ε = 1%%\n\n");
    std::printf("  Detecting stopping N per observable (1 replica each)…\n");

    for (int oi = 0; oi < N_OBS; ++oi) {
        if (std::abs(mean_ref[oi]) < 1e-10) {
            N_stop_per_obs[oi] = MAX_HIST_N;
            continue;
        }
        auto sim = make_single_obs_sim(
            OBS[oi].make_op(), OBS[oi].name,
            HIST_EPS, 100, MAX_HIST_N,
            static_cast<Seed>(8000 + oi * 41));
        auto res = sim.run(psi0, 0.0, T_SIM);
        N_stop_per_obs[oi] = res.total_trajectories;
        std::printf("    %-20s  → N_stop = %zu\n",
            OBS[oi].label, N_stop_per_obs[oi]);
    }

    // Build a dense-enough N schedule for smooth curves.
    // We use ~40 points spaced so they cover 0 to max(N_stop)*1.3.
    const std::size_t N_max_all = *std::max_element(
        N_stop_per_obs, N_stop_per_obs + N_OBS);
    const std::size_t N_hist_max = std::min(
        static_cast<std::size_t>(N_max_all * 1.3 + 100), MAX_HIST_N);

    // Build schedule: small linear steps early, coarser later
    std::vector<std::size_t> N_schedule;
    {
        // 20 points in [100, N_hist_max/4], then 20 points up to N_hist_max
        const std::size_t lo = 100, mid = N_hist_max / 4;
        const int pts_lo = 20, pts_hi = 20;
        for (int k = 0; k <= pts_lo; ++k)
            N_schedule.push_back(lo + k * (mid - lo) / pts_lo);
        for (int k = 1; k <= pts_hi; ++k)
            N_schedule.push_back(mid + k * (N_hist_max - mid) / pts_hi);
        // Deduplicate and sort
        std::sort(N_schedule.begin(), N_schedule.end());
        N_schedule.erase(std::unique(N_schedule.begin(), N_schedule.end()),
                         N_schedule.end());
    }

    std::printf("\n  Running convergence histories (%zu N-points, max N=%zu)…\n\n",
        N_schedule.size(), N_hist_max);

    FILE* fhist = std::fopen("bench2_convergence.csv", "w");
    if (!fhist) { std::fprintf(stderr, "ERROR: bench2_convergence.csv\n"); return 1; }
    std::fprintf(fhist, "observable,label,N,rel_sem\n");

    for (int oi = 0; oi < N_OBS; ++oi) {
        std::printf("  Observable: %-20s", OBS[oi].label);
        std::fflush(stdout);

        int pts_written = 0;
        for (std::size_t N : N_schedule) {
            // Fixed-N run: min = max = N, impossible SEM target
            auto sim = make_single_obs_sim(
                OBS[oi].make_op(), OBS[oi].name,
                1e-15, N, N,
                static_cast<Seed>(9000 + oi * 53 + N % 997));
            auto res = sim.run(psi0, 0.0, T_SIM);

            double m = res.observables[0].mean;
            double s = res.observables[0].sem;
            double rs = (std::abs(m) > 1e-12) ? s / std::abs(m) : 9999.0;

            std::fprintf(fhist, "%s,\"%s\",%zu,%.8f\n",
                OBS[oi].name, OBS[oi].label, N, rs);
            ++pts_written;
        }
        std::printf("  %d points written\n", pts_written);
    }

    std::fclose(fhist);
    std::printf("\n  Saved: bench2_convergence.csv\n\n");
*/
    // ==========================================================
    // Step 4: Summary
    // ==========================================================
    std::printf("%s\n", std::string(66, '═').c_str());
    std::printf("Summary\n\n");

    std::printf("  Variance hierarchy (σ²):\n");
    for (int i = 0; i < N_OBS; ++i)
        std::printf("    %-22s  σ² = %10.6f\n", OBS[i].label, var_ref[i]);

    std::printf("\n  N_stop at ε = 1%%:\n");
    //for (int i = 0; i < N_OBS; ++i)
     //   std::printf("    %-22s  N  = %zu\n", OBS[i].label, N_stop_per_obs[i]);

    std::printf("\n  Efficiency ∈ [0.8, 1.2]: %d / %d rows\n\n",
        total_pass, total_rows);

    std::printf("  Output files:\n");
    std::printf("    bench2_variance.csv     → Plot 3 (variance bar chart)\n");
    std::printf("    bench2_main.csv         → Plot 1 (N_adaptive bar), Plot 4\n");
    //std::printf("    bench2_convergence.csv  → Plot 2 (convergence histories)\n\n");

    return 0;
}
