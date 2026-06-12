// benchmarks/bench3_fixed_vs_adaptive.cpp
// -----------------------------------------------
// Task : — Fixed-N vs Adaptive-N: Practical Efficiency of Adaptive Stopping.
//
// Scientific question:
//   What practical benefit does adaptive stopping (stop_at_sem) provide
//   compared to the traditional fixed-N workflow?
//
// This is distinct from Benchmark 2, which asks:
//   "Do different observables require different trajectory counts?"
//   (Answer: yes, by up to two orders of magnitude.)
//
// Here we ask:
//   "Given that a user naively picks a fixed N, how much does that hurt them?"
//
//   Three failure modes emerge in the table:
//     1. WASTE     — fixed N >> N_adaptive:  trajectories are wasted.
//     2. CORRECT   — fixed N ≈ N_adaptive:  the user got lucky.
//     3. INSUFFICIENT — fixed N << N_adaptive:  the user never met the target.
//
// -- Physical system: driven two-level atom --------------------------------
//   H  = Ω σ_x        resonant drive in rotating frame,  Ω = 0.5
//   L  = √γ σ₋         spontaneous decay,                γ = 1.0
//   ψ₀ = |↑⟩ = (1,0)ᵀ
//   T  = 20.0          long enough to reach near-steady-state
//
//   Steady-state reference (analytic):
//     ρ_ee^ss = 4Ω²/(γ² + 8Ω²) ≈ 0.111
//
// -- Observables (sx dropped: |μ| ≈ 0 makes rel-SEM ill-defined) ---------
//   Pe   = |e><e|   bounded [0,1],   low variance
//   Pg   = |g><g|   bounded [0,1],   low variance (1−Pe, but ~5× smaller N_opt)
//   sz   = σ_z      bounded [-1,1],  moderate variance, largest N_adaptive
//
// -- Adaptive stopping -----------------------------------------------
//   stop_at_sem(0.01)  with  min=100, max=10,000,000
//   3 replicas → median N_adaptive used as the reference point.
//
// -- Fixed-N grid -----------------------------------------------
//   N ∈ { 1000, 5000, 10000, 50000 }
//   Implemented as: stop_at_sem(1e-15), min=max=N  (impossible target → fixed N)
//   1 replica per (observable, N) — enough for the paper table.
//
// -- Key derived quantities -----------------------------------------------
//   waste_factor  = N_fixed / N_adaptive       (> 1 → over-sampled)
//   coverage      = N_fixed / N_adaptive       (< 1 → under-sampled, same formula)
//   target_met    = (achieved_rel_sem <= 1.01 × EPS_TARGET)
//
// -- Output files -----------------------------------------------
//   bench3_adaptive.csv     — adaptive run results (3 replicas per observable)
//   bench3_fixed.csv        — fixed-N run results  (4 N-values × 3 observables)
//   bench3_comparison.csv   — merged table with waste_factor and target_met
//
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

// ----------------------------------------------------------------------------
// Physical parameters
// ----------------------------------------------------------------------------

static constexpr double OMEGA     = 0.5;   // Rabi drive amplitude
static constexpr double GAMMA     = 1.0;   // spontaneous decay rate
static constexpr double T_SIM     = 20.0;  // simulation end time (near-ss)
static constexpr double DT        = 5e-4;  // DOPRI45 initial step
static constexpr double EPS_TARGET = 0.01; // 1% relative SEM target

// ----------------------------------------------------------------------------
// Operator builders
// ----------------------------------------------------------------------------

static DenseOperator make_H() {
    DenseOperator H(2);
    H(0,1) = Scalar{OMEGA, 0.0};
    H(1,0) = Scalar{OMEGA, 0.0};
    return H;
}
static DenseOperator make_L() {
    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(GAMMA), 0.0};
    return L;
}
static DenseOperator make_Pe() {
    DenseOperator o(2); o(0,0) = Scalar{1.0, 0.0}; return o;
}
static DenseOperator make_Pg() {
    DenseOperator o(2); o(1,1) = Scalar{1.0, 0.0}; return o;
}
static DenseOperator make_sz() {
    DenseOperator o(2);
    o(0,0) = Scalar{ 1.0, 0.0};
    o(1,1) = Scalar{-1.0, 0.0};
    return o;
}

// ----------------------------------------------------------------------------
// Simulation factory
//
//   Adaptive run:  rel_sem_target = EPS_TARGET,  min=100, max=10_000_000
//   Fixed-N run:   rel_sem_target = 1e-15,       min=max=N
// ----------------------------------------------------------------------------

static Simulation make_sim(
        DenseOperator  obs_op,
        const char*    obs_name,
        double         rel_sem_target,
        std::size_t    min_N,
        std::size_t    max_N,
        Seed           seed)
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

// ----------------------------------------------------------------------------
// Observable registry  (3 observables; sx excluded)
// ----------------------------------------------------------------------------

static constexpr int N_OBS = 3;

struct ObsMeta {
    const char* name;          // short key for CSV
    const char* label;         // human-readable
    DenseOperator (*make_op)();
};

static const ObsMeta OBS[N_OBS] = {
    { "Pe",  "Population Pe",   make_Pe },
    { "Pg",  "Ground state Pg", make_Pg },
    { "sz",  "sigma_z",         make_sz },
};

// ----------------------------------------------------------------------------
// Fixed-N grid
// ----------------------------------------------------------------------------

static const std::size_t FIXED_N[] = { 1000, 5000, 10000, 50000 };
static constexpr int N_FIXED = static_cast<int>(sizeof FIXED_N / sizeof FIXED_N[0]);

// ----------------------------------------------------------------------------
// Helper: compute rel_sem safely
// ----------------------------------------------------------------------------

static inline double rel_sem(double mean, double sem) {
    return (std::abs(mean) > 1e-12) ? sem / std::abs(mean) : 9999.0;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

int main()
{
    // ── Banner -------------------------------------------------------------
    std::printf(
        "|=========================================================================|\n"
        "|  Benchmark 3 — Fixed-N vs Adaptive-N Stopping                    |\n"
        "|  System  : driven two-level atom, Ω=0.5, γ=1.0, T=20            |\n"
        "|  ψ₀      : |↑⟩  (excited state)                                  |\n"
        "|  Obs     : Pe │ Pg │ σ_z                                          |\n"
        "|  ε_target: 1%% relative SEM                                       |\n"
        "|  Analytic ρ_ee^ss = 4Ω²/(γ²+8Ω²) ≈ 0.111                        |\n"
        "|=========================================================================|\n\n");

    // Initial state |↑⟩
    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};
    psi0[1] = Scalar{0.0, 0.0};

    const double rho_ee_ss = 4.0*OMEGA*OMEGA / (GAMMA*GAMMA + 8.0*OMEGA*OMEGA);
    std::printf("  Analytic ρ_ee^ss = %.6f\n\n", rho_ee_ss);

    // =========================================================================
    // Part 1: Adaptive runs
    //   3 replicas per observable.  Median N_adaptive is the reference.
    //   Seeds chosen to be well-separated from bench2 (base 10000).
    // =========================================================================
    std::printf("----------------------------------------------------------------------------\n");
    std::printf("Part 1: Adaptive stopping  (ε = %.0f%%, 3 replicas)\n\n",
                EPS_TARGET * 100.0);

    FILE* fadapt = std::fopen("bench3_adaptive.csv", "w");
    if (!fadapt) { std::fprintf(stderr, "ERROR: cannot open bench3_adaptive.csv\n"); return 1; }
    std::fprintf(fadapt,
        "observable,label,replica,"
        "N_adaptive,runtime_s,mean,sem,rel_sem,target_met\n");

    // Store medians for the comparison table
    std::size_t N_adapt_med[N_OBS]  = {};
    double      rt_adapt_med[N_OBS] = {};
    double      mean_adapt[N_OBS]   = {};
    double      sem_adapt[N_OBS]    = {};
    double      rs_adapt[N_OBS]     = {};

    const std::string sep68(68, '─');

    for (int oi = 0; oi < N_OBS; ++oi) {
        std::printf("  Observable: %s  (%s)\n", OBS[oi].name, OBS[oi].label);
        std::printf("  %-8s  %-12s  %-12s  %-10s  %-10s  %s\n",
                    "Replica", "N_adaptive", "runtime (s)", "mean", "rel_sem", "target_met");
        std::printf("  %s\n", sep68.c_str());

        std::size_t ns[3];
        double      rts[3], means[3], sems[3], rss[3];
        int         mets[3];

        for (int rep = 0; rep < 3; ++rep) {
            Seed seed = static_cast<Seed>(10000 + oi * 1000 + rep * 37);

            auto t0 = std::chrono::steady_clock::now();
            auto sim = make_sim(
                OBS[oi].make_op(), OBS[oi].name,
                EPS_TARGET,
                /*min_N=*/100,
                /*max_N=*/10'000'000,
                seed);
            auto res = sim.run(psi0, 0.0, T_SIM);
            double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            ns[rep]   = res.total_trajectories;
            rts[rep]  = wall;
            means[rep]= res.observables[0].mean;
            sems[rep] = res.observables[0].sem;
            rss[rep]  = rel_sem(means[rep], sems[rep]);
            mets[rep] = (rss[rep] <= EPS_TARGET * 1.01) ? 1 : 0;

            std::printf("  %-8d  %-12zu  %-12.4f  %-10.6f  %-10.6f  %s\n",
                        rep + 1, ns[rep], rts[rep],
                        means[rep], rss[rep],
                        mets[rep] ? "YES" : "NO");

            std::fprintf(fadapt,
                "%s,\"%s\",%d,%zu,%.6f,%.8f,%.8e,%.8f,%d\n",
                OBS[oi].name, OBS[oi].label, rep + 1,
                ns[rep], rts[rep], means[rep], sems[rep], rss[rep], mets[rep]);
        }

        // Median of 3 replicas (sort each array separately)
        std::size_t ns_s[3];   std::copy(ns,   ns+3,   ns_s);
        double      rts_s[3];  std::copy(rts,  rts+3,  rts_s);
        double      means_s[3];std::copy(means,means+3,means_s);
        double      sems_s[3]; std::copy(sems, sems+3, sems_s);
        double      rss_s[3];  std::copy(rss,  rss+3,  rss_s);
        std::sort(ns_s,    ns_s+3);
        std::sort(rts_s,   rts_s+3);
        std::sort(means_s, means_s+3);
        std::sort(sems_s,  sems_s+3);
        std::sort(rss_s,   rss_s+3);

        N_adapt_med[oi]  = ns_s[1];
        rt_adapt_med[oi] = rts_s[1];
        mean_adapt[oi]   = means_s[1];
        sem_adapt[oi]    = sems_s[1];
        rs_adapt[oi]     = rss_s[1];

        std::printf("  %s\n", sep68.c_str());
        std::printf("  Median N_adaptive = %zu,  rel_sem = %.4f%%\n\n",
                    N_adapt_med[oi], rs_adapt[oi] * 100.0);
    }

    std::fclose(fadapt);
    std::printf("  Saved: bench3_adaptive.csv\n\n");

    // =========================================================================
    // Part 2: Fixed-N runs
    //   One replica per (observable, N).
    //   Seeds: 20000 + oi * 1000 + fi * 13.
    //   Implemented as impossible SEM target with min = max = N.
    // =========================================================================
    std::printf("----------------------------------------------------------------------------\n");
    std::printf("Part 2: Fixed-N runs  (N ∈ {1k, 5k, 10k, 50k})\n\n");

    // Fixed-N results stored for comparison table
    struct FixedResult {
        std::size_t N;
        double      runtime_s;
        double      mean_val;
        double      sem_val;
        double      rel_sem_val;
        int         target_met;
    };
    // [obs][fixed_idx]
    FixedResult fixed_res[N_OBS][N_FIXED];

    FILE* ffixed = std::fopen("bench3_fixed.csv", "w");
    if (!ffixed) { std::fprintf(stderr, "ERROR: cannot open bench3_fixed.csv\n"); return 1; }
    std::fprintf(ffixed,
        "observable,label,N_fixed,runtime_s,mean,sem,rel_sem,target_met\n");

    for (int oi = 0; oi < N_OBS; ++oi) {
        std::printf("  Observable: %s  (%s)\n", OBS[oi].name, OBS[oi].label);
        std::printf("  %-10s  %-12s  %-10s  %-10s  %s\n",
                    "N_fixed", "runtime (s)", "mean", "rel_sem", "target_met");
        std::printf("  %s\n", sep68.c_str());

        for (int fi = 0; fi < N_FIXED; ++fi) {
            const std::size_t N = FIXED_N[fi];
            Seed seed = static_cast<Seed>(20000 + oi * 1000 + fi * 13);

            auto t0 = std::chrono::steady_clock::now();
            auto sim = make_sim(
                OBS[oi].make_op(), OBS[oi].name,
                1e-15,       // impossible → effectively fixed-N
                N, N,        // min = max = N
                seed);
            auto res = sim.run(psi0, 0.0, T_SIM);
            double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            const double m  = res.observables[0].mean;
            const double s  = res.observables[0].sem;
            const double rs = rel_sem(m, s);
            const int    met = (rs <= EPS_TARGET * 1.01) ? 1 : 0;

            fixed_res[oi][fi] = { N, wall, m, s, rs, met };

            std::printf("  %-10zu  %-12.4f  %-10.6f  %-10.6f  %s\n",
                        N, wall, m, rs, met ? "YES" : "NO");

            std::fprintf(ffixed,
                "%s,\"%s\",%zu,%.6f,%.8f,%.8e,%.8f,%d\n",
                OBS[oi].name, OBS[oi].label,
                N, wall, m, s, rs, met);
        }
        std::printf("\n");
    }

    std::fclose(ffixed);
    std::printf("  Saved: bench3_fixed.csv\n\n");

    // =========================================================================
    // Part 3: Merged comparison table + bench3_comparison.csv
    //
    //   For every (observable, N_fixed) pair compute:
    //     waste_factor = N_fixed / N_adaptive_med
    //       > 1.05  → OVER-SAMPLED  (wasteful)
    //       [0.95, 1.05] → MATCHED   (lucky coincidence)
    //       < 0.95  → UNDER-SAMPLED (precision target NOT met)
    // =========================================================================
    std::printf("----------------------------------------------------------------------------\n");
    std::printf("Part 3: Comparison table  (Fixed-N vs Adaptive-N)\n\n");

    FILE* fcmp = std::fopen("bench3_comparison.csv", "w");
    if (!fcmp) { std::fprintf(stderr, "ERROR: cannot open bench3_comparison.csv\n"); return 1; }
    std::fprintf(fcmp,
        "observable,label,"
        "N_adaptive_med,adaptive_rel_sem,adaptive_runtime_s,"
        "N_fixed,fixed_rel_sem,fixed_runtime_s,fixed_target_met,"
        "waste_factor,verdict\n");

    // -- Console header --------------------------------------------------------------
    const std::string sep_wide(92, '═');
    std::printf("  %s\n", sep_wide.c_str());
    std::printf("  %-6s  %-8s  %-12s  %-8s  %-7s  %-7s  %s\n",
                "Obs", "N_fixed", "N_adaptive", "ratio", "fixed_ε%", "adapt_ε%", "Verdict");
    std::printf("  %s\n", sep_wide.c_str());

    // Count outcomes across all cells
    int n_waste = 0, n_matched = 0, n_under = 0;

    for (int oi = 0; oi < N_OBS; ++oi) {
        for (int fi = 0; fi < N_FIXED; ++fi) {
            const std::size_t N_fix = fixed_res[oi][fi].N;
            const std::size_t N_ad  = N_adapt_med[oi];
            const double waste      = static_cast<double>(N_fix)
                                    / static_cast<double>(N_ad);

            const char* verdict;
            if      (waste > 1.05)  { verdict = "OVER-SAMPLED";   ++n_waste;   }
            else if (waste < 0.95)  { verdict = "UNDER-SAMPLED";  ++n_under;   }
            else                    { verdict = "MATCHED";         ++n_matched; }

            std::printf("  %-6s  %-8zu  %-12zu  %-8.2f  %-7.3f  %-7.3f  %s\n",
                        OBS[oi].name,
                        N_fix,
                        N_ad,
                        waste,
                        fixed_res[oi][fi].rel_sem_val * 100.0,
                        rs_adapt[oi] * 100.0,
                        verdict);

            std::fprintf(fcmp,
                "%s,\"%s\",%zu,%.8f,%.6f,%zu,%.8f,%.6f,%d,%.6f,%s\n",
                OBS[oi].name, OBS[oi].label,
                N_ad, rs_adapt[oi], rt_adapt_med[oi],
                N_fix, fixed_res[oi][fi].rel_sem_val,
                fixed_res[oi][fi].runtime_s,
                fixed_res[oi][fi].target_met,
                waste, verdict);
        }
        if (oi < N_OBS - 1)
            std::printf("  %s\n", std::string(92, '─').c_str());
    }

    std::printf("  %s\n\n", sep_wide.c_str());
    std::fclose(fcmp);
    std::printf("  Saved: bench3_comparison.csv\n\n");

    // =========================================================================
    // Part 4: Paper-quality summary
    // =========================================================================
    std::printf("----------------------------------------------------------------------------\n");
    std::printf("Summary for paper  (ε_target = 1%%)\n\n");

    // -- Adaptive N table -------------------------------------------------------
    std::printf("  Adaptive stopping (median of 3 replicas):\n\n");
    std::printf("    %-22s  %10s  %10s  %10s\n",
                "Observable", "N_adaptive", "rel_SEM (%)", "runtime (s)");
    std::printf("    %s\n", std::string(56, '─').c_str());
    for (int oi = 0; oi < N_OBS; ++oi)
        std::printf("    %-22s  %10zu  %10.4f  %10.4f\n",
                    OBS[oi].label, N_adapt_med[oi],
                    rs_adapt[oi] * 100.0, rt_adapt_med[oi]);
    std::printf("\n");

    // -- Waste/coverage at fixed N = 10000 -----------------------------------
    // Find the index for N = 10000
    int idx10k = -1;
    for (int fi = 0; fi < N_FIXED; ++fi)
        if (FIXED_N[fi] == 10000) { idx10k = fi; break; }

    if (idx10k >= 0) {
        std::printf("  Fixed-N = 10,000 comparison:\n\n");
        std::printf("    %-22s  %10s  %10s  %12s  %s\n",
                    "Observable", "N_adaptive", "N_fixed", "ratio", "Verdict");
        std::printf("    %s\n", std::string(70, '─').c_str());
        for (int oi = 0; oi < N_OBS; ++oi) {
            const double w = static_cast<double>(FIXED_N[idx10k])
                           / static_cast<double>(N_adapt_med[oi]);
            const char* tag = (w > 1.05) ? "WASTE" : (w < 0.95) ? "INSUFFICIENT" : "MATCHED";
            std::printf("    %-22s  %10zu  %10zu  %12.2f  %s\n",
                        OBS[oi].label, N_adapt_med[oi],
                        FIXED_N[idx10k], w, tag);
        }
        std::printf("\n");
    }

    // -- ASCII bar chart (for visual inspection at runtime) ------------------
    // x-axis: trajectories, bars scaled to 50 chars, reference line at N=10000
    std::printf("  ASCII bar chart  (reference line = N_fixed 10,000)\n\n");

    const std::size_t N_REF_LINE = 10000;
    // Find the maximum N_adaptive for scaling
    std::size_t N_max_bar = N_REF_LINE;
    for (int oi = 0; oi < N_OBS; ++oi)
        if (N_adapt_med[oi] > N_max_bar) N_max_bar = N_adapt_med[oi];
    N_max_bar = static_cast<std::size_t>(N_max_bar * 1.1);  // 10% margin

    const int BAR_WIDTH = 50;
    for (int oi = 0; oi < N_OBS; ++oi) {
        const int bar_len = static_cast<int>(
            std::round(static_cast<double>(N_adapt_med[oi])
                     / static_cast<double>(N_max_bar) * BAR_WIDTH));
        std::printf("    %-6s  |", OBS[oi].name);
        for (int k = 0; k < bar_len; ++k) std::printf("=");
        std::printf("  %zu\n", N_adapt_med[oi]);
    }
    // Reference line
    const int ref_pos = static_cast<int>(
        std::round(static_cast<double>(N_REF_LINE)
                 / static_cast<double>(N_max_bar) * BAR_WIDTH));
    std::printf("    %-6s  |", "");
    for (int k = 0; k < ref_pos; ++k) std::printf(" ");
    std::printf("^\n");
    std::printf("    %-6s   ", "");
    for (int k = 0; k < ref_pos; ++k) std::printf(" ");
    std::printf("Fixed N = 10,000\n\n");

    // -- Outcome tally ----------------------------------------------------------------------------
    std::printf("  Outcome tally across all %d (observable, N_fixed) pairs:\n\n",
                N_OBS * N_FIXED);
    std::printf("    OVER-SAMPLED  (waste)       : %d\n", n_waste);
    std::printf("    MATCHED       (within ±5%%)  : %d\n", n_matched);
    std::printf("    UNDER-SAMPLED (insufficient): %d\n\n", n_under);

    //-- Output file list ----------------------------------------------------------------------------
    std::printf("  Output files:\n");
    std::printf("    bench3_adaptive.csv    → N_adaptive per observable (bar chart source)\n");
    std::printf("    bench3_fixed.csv       → fixed-N precision and runtime\n");
    std::printf("    bench3_comparison.csv  → merged table with waste_factor\n\n");

    return 0;
}
