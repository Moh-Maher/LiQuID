// benchmarks/bench4_higher_dim_variance.cpp
// -----------------------------------------------------------------------------
// Task - Higher-Dimensional Benchmark: Observable-Dependent Convergence
//        Is Not Specific to the Two-Level System.
//
// Core argument (extends Benchmark 2 to d = 4):
//   Benchmark 2 showed that, for a driven two-level atom, different
//   observables have vastly different variances and therefore require very
//   different ensemble sizes N to reach the same relative precision ε
//   (N_optimal = σ²/(ε·|μ|)²).  A referee could reasonably ask whether this
//   is an artefact of the two-level Hilbert space.  This benchmark answers
//   that question by repeating the same measurement on a d = 4 open quantum
//   system: two coupled, dissipative qubits.
//
// -- Physical system: two coupled dissipative qubits (d = 4) -----------------
//   H  = Ω₁ σx⊗I + Ω₂ I⊗σx + J σz⊗σz         coherent drive + Ising coupling
//   L₁ = √γ₁ σ₋⊗I                             local decay, qubit 1
//   L₂ = √γ₂ I⊗σ₋                             local decay, qubit 2
//   ψ₀ = |e₁,e₂⟩ = (1,0,0,0)ᵀ                  both qubits excited
//   Ω₁ = 0.5, Ω₂ = 0.2, J = 0.25, γ₁ = 1.0, γ₂ = 0.9
//   T  = 12.0   (≳ 10 × the slowest decay time 1/γ₂ ≈ 1.11 → near steady state)
//
//   Basis ordering (row-major Kronecker product, local basis {|e⟩,|g⟩}):
//     index 0 = |e₁,e₂⟩, 1 = |e₁,g₂⟩, 2 = |g₁,e₂⟩, 3 = |g₁,g₂⟩
//
// -- Five observables with different statistical behaviour --------------------
//   Pe1  = Pe⊗I           marginal excited-state population, qubit 1
//   Pe2  = I⊗Pe           marginal excited-state population, qubit 2
//   Pee  = Pe⊗Pe           joint excited-state population (both qubits)
//   sz1  = σz⊗I            ⟨σz⟩ of qubit 1
//   sz2  = I⊗σz            ⟨σz⟩ of qubit 2 (parameters chosen so qubit 2 sits
//                           close to its steady-state floor → deliberately
//                           the LOW-variance end of the spread, mirroring the
//                           role Pg played in Benchmark 2)
//
// -- API note (same as Benchmark 2) -------------------------------------------
//   stop_at_sem(eps) stops when ALL registered observables satisfy
//   rel_SEM < eps.  To drive stopping on a SPECIFIC observable we register
//   only that observable.  For reference/fixed-N runs (min = max = N,
//   target = 1e-15) we register all five observables at once, since the
//   run length is fixed and independent of any one observable's precision;
//   this reuses the same trajectory batch for all five statistics and is
//   far cheaper than five separate fixed-N runs.
//
// -- Benchmark structure -------------------------------------------------------
//   Step 1  Reference run (single N_ref-trajectory batch, all 5 observables)
//           → mean, σ², σ/|μ| per observable.
//   Step 2  Adaptive stopping per (observable, ε) at
//           ε ∈ {10%, 5%, 2%, 1%, 0.5%}.  Records N_optimal, N_adaptive,
//           efficiency = N_optimal/N_adaptive, achieved rel_SEM.
//   Step 3  N_max/N_min ratio between observables, at each ε and overall.
//   Step 4  Fixed-N comparison: one small N (favours the lowest-variance
//           observable, fails the highest-variance one) and one large N
//           (meets every target but heavily oversamples the low-variance
//           observable).
//
// -- Output files ---------------------------------------------------------------
//   bench4_variance.csv   σ², |μ|, σ/|μ| per observable (reference run)
//   bench4_main.csv       N_optimal, N_adaptive, efficiency, achieved ε
//   bench4_ratio.csv      N_max/N_min across observables, per ε and overall
//   bench4_fixed.csv      fixed-N comparison (small N vs large N)
// -----------------------------------------------------------------------------

#include "liquid/liquid.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace liquid;
using namespace liquid::ensemble;

// -----------------------------------------------------------------------------
// Physical parameters
// -----------------------------------------------------------------------------

static constexpr double OMEGA1 = 0.5;    // drive amplitude, qubit 1
static constexpr double OMEGA2 = 0.2;    // drive amplitude, qubit 2
static constexpr double J_ZZ   = 0.25;   // Ising coupling strength
static constexpr double GAMMA1 = 1.0;    // spontaneous decay rate, qubit 1
static constexpr double GAMMA2 = 0.9;    // spontaneous decay rate, qubit 2
static constexpr double T_SIM  = 12.0;   // near-steady-state
static constexpr double DT     = 5e-4;   // DOPRI45 initial step

// -----------------------------------------------------------------------------
// Linear-algebra helper: Kronecker (tensor) product of two dense operators.
// -----------------------------------------------------------------------------

static DenseOperator kron(const DenseOperator& A, const DenseOperator& B) {
    const Dim da = A.size(), db = B.size();
    DenseOperator C(da * db);
    for (Idx i1 = 0; i1 < da; ++i1)
        for (Idx j1 = 0; j1 < da; ++j1) {
            const Scalar a = A(i1, j1);
            for (Idx i2 = 0; i2 < db; ++i2)
                for (Idx j2 = 0; j2 < db; ++j2)
                    C(i1 * db + i2, j1 * db + j2) = a * B(i2, j2);
        }
    return C;
}

// -----------------------------------------------------------------------------
// Local (single-qubit) operator builders, basis {|e>, |g>}
// -----------------------------------------------------------------------------

static DenseOperator local_I() {
    DenseOperator o(2); o(0,0) = Scalar{1.0,0.0}; o(1,1) = Scalar{1.0,0.0};
    return o;
}
static DenseOperator local_sx() {
    DenseOperator o(2); o(0,1) = Scalar{1.0,0.0}; o(1,0) = Scalar{1.0,0.0};
    return o;
}
static DenseOperator local_sz() {
    DenseOperator o(2); o(0,0) = Scalar{1.0,0.0}; o(1,1) = Scalar{-1.0,0.0};
    return o;
}
static DenseOperator local_sm() {                      // σ₋: |e> -> |g>
    DenseOperator o(2); o(1,0) = Scalar{1.0,0.0};
    return o;
}
static DenseOperator local_Pe() {
    DenseOperator o(2); o(0,0) = Scalar{1.0,0.0};
    return o;
}

// -----------------------------------------------------------------------------
// Composite (d = 4) system operators — built once, reused everywhere.
// -----------------------------------------------------------------------------

struct System4 {
    DenseOperator H, L1, L2;
    DenseOperator Pe1, Pe2, Pee, sz1, sz2;
};

static System4 build_system() {
    const DenseOperator I2 = local_I();
    const DenseOperator sx = local_sx();
    const DenseOperator sz = local_sz();
    const DenseOperator sm = local_sm();
    const DenseOperator Pe = local_Pe();

    const DenseOperator Hsx1 = kron(sx, I2);
    const DenseOperator Hsx2 = kron(I2, sx);
    const DenseOperator Hzz  = kron(sz, sz);

    DenseOperator H(4);
    for (Idx i = 0; i < 4; ++i)
        for (Idx j = 0; j < 4; ++j)
            H(i,j) = Scalar{OMEGA1,0.0} * Hsx1(i,j)
                   + Scalar{OMEGA2,0.0} * Hsx2(i,j)
                   + Scalar{J_ZZ,  0.0} * Hzz(i,j);

    DenseOperator L1 = kron(sm, I2);
    DenseOperator L2 = kron(I2, sm);
    for (Idx i = 0; i < 4; ++i)
        for (Idx j = 0; j < 4; ++j) {
            L1(i,j) = L1(i,j) * Scalar{std::sqrt(GAMMA1), 0.0};
            L2(i,j) = L2(i,j) * Scalar{std::sqrt(GAMMA2), 0.0};
        }

    System4 s;
    s.H = std::move(H);
    s.L1 = std::move(L1);
    s.L2 = std::move(L2);
    s.Pe1 = kron(Pe, I2);
    s.Pe2 = kron(I2, Pe);
    s.Pee = kron(Pe, Pe);
    s.sz1 = kron(sz, I2);
    s.sz2 = kron(I2, sz);
    return s;
}

// -----------------------------------------------------------------------------
// Observable registry
// -----------------------------------------------------------------------------

static constexpr int N_OBS = 5;

struct ObsMeta {
    const char* name;
    const char* label;
};

static const ObsMeta OBS[N_OBS] = {
    { "Pe1", "P_e1 (qubit 1 excited pop.)"   },
    { "Pe2", "P_e2 (qubit 2 excited pop.)"   },
    { "Pee", "P_ee (joint excited pop.)"     },
    { "sz1", "<sigma_z^(1)>"                 },
    { "sz2", "<sigma_z^(2)>"                 },
};

static DenseOperator obs_operator(const System4& sys, int idx) {
    switch (idx) {
        case 0: return sys.Pe1;
        case 1: return sys.Pe2;
        case 2: return sys.Pee;
        case 3: return sys.sz1;
        case 4: return sys.sz2;
    }
    throw std::runtime_error("bad observable index");
}

// -----------------------------------------------------------------------------
// Simulation factories
// -----------------------------------------------------------------------------

// Single-observable simulation: stop_at_sem(eps) drives on THIS observable.
static Simulation make_single_obs_sim(
        const System4& sys, int obs_idx,
        double rel_sem_target, std::size_t min_N, std::size_t max_N, Seed seed)
{
    return SimulationBuilder{}
        .hamiltonian(sys.H)
        .collapse_operator(sys.L1)
        .collapse_operator(sys.L2)
        .observe(OBS[obs_idx].name, obs_operator(sys, obs_idx))
        .seed(seed)
        .dt(DT)
        .stop_at_sem(rel_sem_target)
        .min_trajectories(min_N)
        .max_trajectories(max_N)
        .build();
}

// All-observable simulation for fixed-N runs (reference and fixed-N grid):
// registers all five observables so one trajectory batch yields every
// statistic at once.  min = max = N and an impossible SEM target makes the
// run length exactly N regardless of any one observable's precision.
static Simulation make_all_obs_fixed_sim(
        const System4& sys, std::size_t N, Seed seed)
{
    SimulationBuilder b;
    b.hamiltonian(sys.H).collapse_operator(sys.L1).collapse_operator(sys.L2);
    for (int i = 0; i < N_OBS; ++i)
        b.observe(OBS[i].name, obs_operator(sys, i));
    return b.seed(seed).dt(DT).stop_at_sem(1e-15)
            .min_trajectories(N).max_trajectories(N).build();
}

static inline double rel_sem(double mean, double sem) {
    return (std::abs(mean) > 1e-12) ? sem / std::abs(mean) : 9999.0;
}

// -----------------------------------------------------------------------------
// Tiny state-file helpers (plain "key,value,..." lines, no quoting needed —
// used only to pass numbers between the staged invocations below; this
// benchmark is split into stages because each individual run must complete
// within the sandbox's per-command wall-clock budget).
// -----------------------------------------------------------------------------

static void write_ref_state(const double mean_ref[N_OBS], const double var_ref[N_OBS]) {
    FILE* f = std::fopen("bench4_ref_state.csv", "w");
    if (!f) { std::fprintf(stderr, "ERROR: cannot write bench4_ref_state.csv\n"); std::exit(1); }
    for (int i = 0; i < N_OBS; ++i)
        std::fprintf(f, "%s,%.10e,%.10e\n", OBS[i].name, mean_ref[i], var_ref[i]);
    std::fclose(f);
}

static void read_ref_state(double mean_ref[N_OBS], double var_ref[N_OBS]) {
    FILE* f = std::fopen("bench4_ref_state.csv", "r");
    if (!f) {
        std::fprintf(stderr, "ERROR: bench4_ref_state.csv not found. Run stage 'ref' first.\n");
        std::exit(1);
    }
    char name[64]; double m, v;
    while (std::fscanf(f, "%63[^,],%lf,%lf\n", name, &m, &v) == 3) {
        for (int i = 0; i < N_OBS; ++i)
            if (std::string(name) == OBS[i].name) { mean_ref[i] = m; var_ref[i] = v; }
    }
    std::fclose(f);
}

// bench4_N_state.csv accumulates one row per (obs_idx, eps_idx) -> N_adaptive
static void append_N_state(int obs_idx, int eps_idx, std::size_t N_ad) {
    FILE* f = std::fopen("bench4_N_state.csv", "a");
    if (!f) { std::fprintf(stderr, "ERROR: cannot append bench4_N_state.csv\n"); std::exit(1); }
    std::fprintf(f, "%d,%d,%zu\n", obs_idx, eps_idx, N_ad);
    std::fclose(f);
}

static bool read_N_state(std::size_t N_adapt[N_OBS][5]) {
    FILE* f = std::fopen("bench4_N_state.csv", "r");
    if (!f) return false;
    int oi, ti; unsigned long long n;
    bool seen[N_OBS][5] = {};
    while (std::fscanf(f, "%d,%d,%llu\n", &oi, &ti, &n) == 3) {
        if (oi >= 0 && oi < N_OBS && ti >= 0 && ti < 5) {
            N_adapt[oi][ti] = static_cast<std::size_t>(n);
            seen[oi][ti] = true;
        }
    }
    std::fclose(f);
    for (int i = 0; i < N_OBS; ++i)
        for (int j = 0; j < 5; ++j)
            if (!seen[i][j]) return false;
    return true;
}

// -----------------------------------------------------------------------------
// Stage: ref — Step 1 only.
// -----------------------------------------------------------------------------

static int stage_ref(const System4& sys, const StateVector& psi0)
{
    constexpr std::size_t N_REF = 20'000;
    std::printf("Step 1: Reference run  (N_ref = %zu, all 5 observables)\n\n", N_REF);

    double mean_ref[N_OBS], sem_ref[N_OBS], var_ref[N_OBS], relsd_ref[N_OBS];
    double wall_ref = 0.0;
    {
        auto t0 = std::chrono::steady_clock::now();
        auto sim = make_all_obs_fixed_sim(sys, N_REF, /*seed=*/42);
        auto res = sim.run(psi0, 0.0, T_SIM);
        wall_ref = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        for (int i = 0; i < N_OBS; ++i) {
            mean_ref[i] = res.observables[i].mean;
            sem_ref[i]  = res.observables[i].sem;
            var_ref[i]  = sem_ref[i] * sem_ref[i] * static_cast<double>(N_REF);
            relsd_ref[i] = (std::abs(mean_ref[i]) > 1e-12)
                ? std::sqrt(var_ref[i]) / std::abs(mean_ref[i]) : 9999.0;
        }
    }

    const std::string sep_med(68, '-');
    std::printf("  %-28s  %+12s  %12s  %12s  %10s\n",
        "Observable", "mean", "sigma^2", "sigma", "sig/|mu|");
    std::printf("  %s\n", sep_med.c_str());
    for (int i = 0; i < N_OBS; ++i)
        std::printf("  %-28s  %+12.6f  %12.6f  %12.6f  %10.4f\n",
            OBS[i].label, mean_ref[i], var_ref[i], std::sqrt(var_ref[i]), relsd_ref[i]);
    std::printf("  %s\n", sep_med.c_str());
    std::printf("  (reference run wall time: %.1f s)\n\n", wall_ref);

    FILE* fv = std::fopen("bench4_variance.csv", "w");
    if (!fv) { std::fprintf(stderr, "ERROR: bench4_variance.csv\n"); return 1; }
    std::fprintf(fv, "observable,label,mean,variance,sigma,sigma_over_absmean\n");
    for (int i = 0; i < N_OBS; ++i)
        std::fprintf(fv, "%s,\"%s\",%.8f,%.8f,%.8f,%.8f\n",
            OBS[i].name, OBS[i].label, mean_ref[i], var_ref[i],
            std::sqrt(var_ref[i]), relsd_ref[i]);
    std::fclose(fv);
    std::printf("  Saved: bench4_variance.csv\n");

    write_ref_state(mean_ref, var_ref);
    std::printf("  Saved: bench4_ref_state.csv  (checkpoint for later stages)\n\n");
    return 0;
}

// -----------------------------------------------------------------------------
// Stage: sweep <obs_idx> — Step 2 for ONE observable, all 5 eps targets.
// Appends to bench4_main.csv and bench4_N_state.csv.
// -----------------------------------------------------------------------------

static int stage_sweep(const System4& sys, const StateVector& psi0, int oi)
{
    if (oi < 0 || oi >= N_OBS) {
        std::fprintf(stderr, "ERROR: obs_idx must be in [0,%d]\n", N_OBS - 1);
        return 1;
    }
    double mean_ref[N_OBS], var_ref[N_OBS];
    read_ref_state(mean_ref, var_ref);

    const double targets[] = {0.10, 0.05, 0.02, 0.01, 0.005};
    const int n_targets = static_cast<int>(sizeof targets / sizeof targets[0]);

    std::printf("Step 2: Adaptive stopping -- observable %d/%d: %s\n\n",
        oi + 1, N_OBS, OBS[oi].label);
    std::printf("  mu = %+.6f, sigma^2 = %.6f\n", mean_ref[oi], var_ref[oi]);
    std::printf("  %-10s  %-12s  %-12s  %-10s  %-14s  %-9s\n",
        "eps (rel)", "N_optimal", "N_adaptive", "Efficiency", "achieved eps", "wall (s)");
    const std::string sep_long(88, '-');
    std::printf("  %s\n", sep_long.c_str());

    // Write header only when this is the first observable's sweep.
    const bool write_header = (oi == 0);
    FILE* fmain = std::fopen("bench4_main.csv", write_header ? "w" : "a");
    if (!fmain) { std::fprintf(stderr, "ERROR: bench4_main.csv\n"); return 1; }
    if (write_header)
        std::fprintf(fmain,
            "observable,label,target_rel_sem,mean_ref,variance_ref,"
            "N_optimal,N_adaptive,efficiency,achieved_rel_sem,wall_s\n");

    for (int ti = 0; ti < n_targets; ++ti) {
        const double eps = targets[ti];
        const double denom = eps * std::abs(mean_ref[oi]);
        const std::size_t N_opt = static_cast<std::size_t>(
            std::ceil(var_ref[oi] / (denom * denom)));

        Seed seed = static_cast<Seed>(40000 + oi * 1000 + ti * 37);

        auto t0 = std::chrono::steady_clock::now();
        auto sim = make_single_obs_sim(
            sys, oi, eps, /*min_N=*/100, /*max_N=*/500'000, seed);
        auto res = sim.run(psi0, 0.0, T_SIM);
        double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        const std::size_t N_ad = res.total_trajectories;
        const double m = res.observables[0].mean;
        const double s = res.observables[0].sem;
        const double rs = rel_sem(m, s);
        const double eff = (N_ad > 0)
            ? static_cast<double>(N_opt) / static_cast<double>(N_ad) : 0.0;

        std::printf("  %-10.3f  %-12zu  %-12zu  %-10.3f  %-14.4f  %-9.2f\n",
            eps, N_opt, N_ad, eff, rs, wall);

        std::fprintf(fmain, "%s,\"%s\",%.6f,%.8f,%.8f,%zu,%zu,%.6f,%.6f,%.4f\n",
            OBS[oi].name, OBS[oi].label, eps, mean_ref[oi], var_ref[oi],
            N_opt, N_ad, eff, rs, wall);
        std::fflush(fmain);

        append_N_state(oi, ti, N_ad);
    }
    std::fclose(fmain);
    std::printf("  %s\n\n  Appended to bench4_main.csv and bench4_N_state.csv\n\n",
        sep_long.c_str());
    return 0;
}

// -----------------------------------------------------------------------------
// Stage: finalize — Step 3 (ratios) + Step 4 (fixed-N comparison) + summary.
// -----------------------------------------------------------------------------

static int stage_finalize(const System4& sys, const StateVector& psi0)
{
    double mean_ref[N_OBS], var_ref[N_OBS];
    read_ref_state(mean_ref, var_ref);

    std::size_t N_adapt[N_OBS][5];
    if (!read_N_state(N_adapt)) {
        std::fprintf(stderr,
            "ERROR: bench4_N_state.csv incomplete. Run all 5 'sweep' stages first.\n");
        return 1;
    }

    const double targets[] = {0.10, 0.05, 0.02, 0.01, 0.005};
    const int n_targets = static_cast<int>(sizeof targets / sizeof targets[0]);
    const std::string sep_long(88, '-');

    // ---- Step 3: N_max/N_min ratio -----------------------------------------
    std::printf("Step 3: N_max / N_min ratio across observables\n\n");

    FILE* frat = std::fopen("bench4_ratio.csv", "w");
    if (!frat) { std::fprintf(stderr, "ERROR: bench4_ratio.csv\n"); return 1; }
    std::fprintf(frat, "target_rel_sem,obs_min,N_min,obs_max,N_max,ratio\n");

    std::printf("  %-10s  %-10s  %-12s  %-10s  %-12s  %-8s\n",
        "eps", "obs(min N)", "N_min", "obs(max N)", "N_max", "ratio");
    std::printf("  %s\n", sep_long.c_str());

    std::size_t overall_min = SIZE_MAX, overall_max = 0;
    int overall_min_obs = -1, overall_min_eps = -1;
    int overall_max_obs = -1, overall_max_eps = -1;

    for (int ti = 0; ti < n_targets; ++ti) {
        int i_min = 0, i_max = 0;
        for (int oi = 1; oi < N_OBS; ++oi) {
            if (N_adapt[oi][ti] < N_adapt[i_min][ti]) i_min = oi;
            if (N_adapt[oi][ti] > N_adapt[i_max][ti]) i_max = oi;
        }
        const std::size_t nmin = N_adapt[i_min][ti];
        const std::size_t nmax = N_adapt[i_max][ti];
        const double ratio = (nmin > 0)
            ? static_cast<double>(nmax) / static_cast<double>(nmin) : 0.0;

        std::printf("  %-10.3f  %-10s  %-12zu  %-10s  %-12zu  %-8.2fx\n",
            targets[ti], OBS[i_min].name, nmin, OBS[i_max].name, nmax, ratio);

        std::fprintf(frat, "%.6f,%s,%zu,%s,%zu,%.4f\n",
            targets[ti], OBS[i_min].name, nmin, OBS[i_max].name, nmax, ratio);

        if (nmin < overall_min) { overall_min = nmin; overall_min_obs = i_min; overall_min_eps = ti; }
        if (nmax > overall_max) { overall_max = nmax; overall_max_obs = i_max; overall_max_eps = ti; }
    }
    std::printf("  %s\n\n", sep_long.c_str());
    std::fclose(frat);
    std::printf("  Saved: bench4_ratio.csv\n\n");

    // ---- Step 4: fixed-N comparison ----------------------------------------
    constexpr double EPS_FIXED = 0.01;
    int ti_1pct = 0;
    for (int ti = 0; ti < n_targets; ++ti)
        if (std::abs(targets[ti] - EPS_FIXED) < 1e-9) { ti_1pct = ti; break; }

    int i_lowvar = 0, i_highvar = 0;
    for (int oi = 1; oi < N_OBS; ++oi) {
        if (N_adapt[oi][ti_1pct] < N_adapt[i_lowvar][ti_1pct])  i_lowvar  = oi;
        if (N_adapt[oi][ti_1pct] > N_adapt[i_highvar][ti_1pct]) i_highvar = oi;
    }
    const std::size_t N_small = N_adapt[i_lowvar][ti_1pct];
    const std::size_t N_large = N_adapt[i_highvar][ti_1pct];

    std::printf("Step 4: Fixed-N comparison at eps_target = %.0f%%\n\n", EPS_FIXED*100.0);
    std::printf("  N_small = %zu  (matches %s's adaptive N -- %s will UNDER-sample)\n",
        N_small, OBS[i_lowvar].label, OBS[i_highvar].label);
    std::printf("  N_large = %zu  (matches %s's adaptive N -- %s will be OVER-sampled)\n\n",
        N_large, OBS[i_highvar].label, OBS[i_lowvar].label);

    FILE* ffix = std::fopen("bench4_fixed.csv", "w");
    if (!ffix) { std::fprintf(stderr, "ERROR: bench4_fixed.csv\n"); return 1; }
    std::fprintf(ffix,
        "observable,label,N_fixed,fixed_role,mean,sem,rel_sem,target_met,waste_factor\n");

    const std::pair<std::size_t,const char*> fixed_grid[2] = {
        { N_small, "small" },
        { N_large, "large" },
    };

    std::printf("  %-28s  %-10s  %-10s  %-10s  %-8s  %-8s\n",
        "Observable", "N_fixed", "role", "rel_sem", "met?", "waste");
    std::printf("  %s\n", sep_long.c_str());

    for (const auto& grid : fixed_grid) {
        const std::size_t N = grid.first;
        const char* role = grid.second;

        auto sim = make_all_obs_fixed_sim(sys, N, /*seed=*/50000 + static_cast<Seed>(N % 9973));
        auto res = sim.run(psi0, 0.0, T_SIM);

        for (int oi = 0; oi < N_OBS; ++oi) {
            const double m  = res.observables[oi].mean;
            const double s  = res.observables[oi].sem;
            const double rs = rel_sem(m, s);
            const int met    = (rs <= EPS_FIXED * 1.01) ? 1 : 0;
            const double waste = static_cast<double>(N)
                / static_cast<double>(N_adapt[oi][ti_1pct]);

            std::printf("  %-28s  %-10zu  %-10s  %-10.4f  %-8s  %-7.2fx\n",
                OBS[oi].label, N, role, rs, met ? "yes" : "no", waste);

            std::fprintf(ffix, "%s,\"%s\",%zu,%s,%.8f,%.8e,%.6f,%d,%.4f\n",
                OBS[oi].name, OBS[oi].label, N, role, m, s, rs, met, waste);
        }
        std::printf("  %s\n", sep_long.c_str());
    }
    std::fclose(ffix);
    std::printf("\n  Saved: bench4_fixed.csv\n\n");

    // ---- Summary ------------------------------------------------------------
    double relsd_ref[N_OBS];
    for (int i = 0; i < N_OBS; ++i)
        relsd_ref[i] = (std::abs(mean_ref[i]) > 1e-12)
            ? std::sqrt(var_ref[i]) / std::abs(mean_ref[i]) : 9999.0;

    std::printf("%s\n", std::string(66, '=').c_str());
    std::printf("Summary\n\n");
    std::printf("  Relative-std hierarchy (sigma/|mu|), reference run:\n");
    for (int i = 0; i < N_OBS; ++i)
        std::printf("    %-30s  sig/|mu| = %8.4f\n", OBS[i].label, relsd_ref[i]);

    std::printf("\n  Overall N_max/N_min across all (observable, eps) cells:\n");
    if (overall_min_obs >= 0 && overall_max_obs >= 0) {
        const double overall_ratio = (overall_min > 0)
            ? static_cast<double>(overall_max) / static_cast<double>(overall_min) : 0.0;
        std::printf("    N_min = %zu  (%s @ eps=%.3f)\n",
            overall_min, OBS[overall_min_obs].label, targets[overall_min_eps]);
        std::printf("    N_max = %zu  (%s @ eps=%.3f)\n",
            overall_max, OBS[overall_max_obs].label, targets[overall_max_eps]);
        std::printf("    Overall ratio: %.1fx\n", overall_ratio);
    }

    std::printf("\n  Output files:\n");
    std::printf("    bench4_variance.csv  -> Fig. variance/relative-std bar chart\n");
    std::printf("    bench4_main.csv      -> Fig. N_adaptive bar chart, efficiency scatter\n");
    std::printf("    bench4_ratio.csv     -> Fig. N_max/N_min ratio vs eps\n");
    std::printf("    bench4_fixed.csv     -> Fig. fixed-N failure/oversampling panel\n\n");

    return 0;
}

// -----------------------------------------------------------------------------
// main — dispatches to a stage. The benchmark is staged (rather than one
// monolithic run) purely so each invocation completes within a bounded
// wall-clock budget; the physics, seeds, and outputs are identical to a
// single-shot run.
//
//   ./bench4_higher_dim_variance ref
//   ./bench4_higher_dim_variance sweep 0    (repeat for obs_idx = 0..4)
//   ./bench4_higher_dim_variance finalize
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered: progress visible live

    const System4 sys = build_system();
    StateVector psi0(4);
    psi0[0] = Scalar{1.0, 0.0};   // |e1,e2>

    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s ref | sweep <0-4> | finalize\n", argv[0]);
        return 1;
    }
    const std::string stage = argv[1];

    std::printf(
        "|================================================================|\n"
        "|  Benchmark 4 -- Higher-Dimensional (d=4) Variance Dependence     |\n"
        "|  System : two coupled dissipative qubits                        |\n"
        "|  H = O1 sx1 + O2 sx2 + J sz1 sz2 ;  L1,L2 = local decay          |\n"
        "|  O1=%.2f  O2=%.2f  J=%.2f  g1=%.2f  g2=%.2f  T=%.1f              |\n"
        "|  psi0 : |e1,e2>                                                  |\n"
        "|  Obs   : Pe1 | Pe2 | Pee | sz1 | sz2                             |\n"
        "|================================================================|\n\n",
        OMEGA1, OMEGA2, J_ZZ, GAMMA1, GAMMA2, T_SIM);


    int rc = 0;
    if (stage == "ref") {
        rc = stage_ref(sys, psi0);
    } else if (stage == "sweep") {
        if (argc < 3) {
            std::fprintf(stderr, "Usage: %s sweep <obs_idx 0-4>\n", argv[0]);
            return 1;
        }
        rc = stage_sweep(sys, psi0, std::atoi(argv[2]));
    } else if (stage == "finalize") {
        rc = stage_finalize(sys, psi0);
    } else {
        std::fprintf(stderr,
            "Unknown stage '%s'. Usage: %s ref | sweep <0-4> | finalize\n",
            stage.c_str(), argv[0]);
        rc = 1;
    }
    return rc;
}
