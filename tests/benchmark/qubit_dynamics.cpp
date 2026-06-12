// examples/qubit_dynamics.cpp
// ─────────────────────────────────────────────────────────────────────────────
// LiQuID reproduction of the QuTiP tutorial:
//   "Qubit dynamics" — tutorials-v5/time-evolution/003_qubit-dynamics.ipynb
//   https://qutip.org/qutip-tutorials/
//
// Physical system:
//   H   = ω · (cos(θ)·σ_z + sin(θ)·σ_x)
//   L1  = √(γ₁·(n_th+1)) · σ_-    (spontaneous emission)
//   L2  = √(γ₁·n_th)     · σ_+    (thermal excitation)
//   L3  = √γ₂             · σ_z   (pure dephasing)
//
// Parameters (identical to QuTiP tutorial):
//   ω = 2π, θ = 0.2π, γ₁ = 0.5, γ₂ = 0.2, n_th = 0.5
//   Initial state: |0⟩ = excited state = basis(2,0)
//   Times: t ∈ [0, 10], 200 measurement points
//
// Observables: ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩
//
// Comparison:
//   - Analytic reference (Bloch equation solution for this system)
//   - LiQuID MCWF with convergence-based stopping
//   - Direct CSV output for plotting against QuTiP mcsolve results
//
// Build:
//   g++ -std=c++17 -O2 -DNDEBUG -Iinclude \
//       examples/qubit_dynamics.cpp src/core/rng.cpp -o qubit_dynamics
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/liquid.hpp"
#include <functional>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace liquid;
using namespace liquid::ensemble;

// ── Physical parameters (exactly as in QuTiP tutorial) ───────────────────────

static constexpr double OMEGA  = 2.0 * M_PI;  // qubit frequency
static constexpr double THETA  = 0.2 * M_PI;  // tilt from z-axis
static constexpr double GAMMA1 = 0.5;          // relaxation rate
static constexpr double GAMMA2 = 0.2;          // dephasing rate
static constexpr double N_TH   = 0.5;          // thermal occupation

// Time grid: 200 points in [0, 10] matching QuTiP linspace(0, 10, 200)
static constexpr int    N_TIMES = 200;
static constexpr double T_MAX   = 10.0;

// ── Pauli matrices ────────────────────────────────────────────────────────────
//   σ_z = [[1,0],[0,-1]]   σ_x = [[0,1],[1,0]]   σ_y = [[0,-i],[i,0]]
//   σ_+ = [[0,1],[0,0]]    σ_- = [[0,0],[1,0]]

static DenseOperator make_sigma_z() {
    DenseOperator sz(2);
    sz(0,0)=Scalar{ 1,0}; sz(1,1)=Scalar{-1,0};
    return sz;
}
static DenseOperator make_sigma_x() {
    DenseOperator sx(2);
    sx(0,1)=Scalar{1,0}; sx(1,0)=Scalar{1,0};
    return sx;
}
static DenseOperator make_sigma_y() {
    DenseOperator sy(2);
    sy(0,1)=Scalar{0,-1}; sy(1,0)=Scalar{0,1};
    return sy;
}
static DenseOperator make_sigma_plus() {
    DenseOperator sp(2);
    sp(0,1)=Scalar{1,0};
    return sp;
}
static DenseOperator make_sigma_minus() {
    DenseOperator sm(2);
    sm(1,0)=Scalar{1,0};
    return sm;
}

// ── System construction ───────────────────────────────────────────────────────

static DenseOpenSystem make_qubit_system() {
    // H = ω·(cos(θ)·σ_z + sin(θ)·σ_x)
    DenseOperator H(2);
    const double cos_t = std::cos(THETA);
    const double sin_t = std::sin(THETA);
    H(0,0) = Scalar{OMEGA * cos_t,  0};
    H(1,1) = Scalar{-OMEGA * cos_t, 0};
    H(0,1) = Scalar{OMEGA * sin_t,  0};
    H(1,0) = Scalar{OMEGA * sin_t,  0};

    // Collapse operators
    std::vector<DenseOperator> cops;

    // L1 = √(γ₁·(n_th+1)) · σ_-   (emission)
    const double rate1_em = GAMMA1 * (N_TH + 1.0);
    if (rate1_em > 0.0) {
        DenseOperator L1(2);
        L1(1,0) = Scalar{std::sqrt(rate1_em), 0};
        cops.push_back(std::move(L1));
    }

    // L2 = √(γ₁·n_th) · σ_+   (absorption from thermal bath)
    const double rate1_abs = GAMMA1 * N_TH;
    if (rate1_abs > 0.0) {
        DenseOperator L2(2);
        L2(0,1) = Scalar{std::sqrt(rate1_abs), 0};
        cops.push_back(std::move(L2));
    }

    // L3 = √γ₂ · σ_z   (pure dephasing)
    if (GAMMA2 > 0.0) {
        DenseOperator L3(2);
        L3(0,0) = Scalar{std::sqrt(GAMMA2),  0};
        L3(1,1) = Scalar{-std::sqrt(GAMMA2), 0};
        cops.push_back(std::move(L3));
    }

    return DenseOpenSystem(std::move(H),
                           LindbladSet<DenseTag>(std::move(cops)));
}

// ── Run LiQuID at a specific time point ──────────────────────────────────────
//
// LiQuID runs trajectories from t=0 to t=T, measuring at t=T.
// To get the time series ⟨σ(t)⟩ for all t, we run N_TIMES independent
// ensembles, one per measurement time. This matches what QuTiP's mcsolve
// does when it returns expectation values at each time in the tlist.
//
// Each ensemble uses convergence-based stopping (target 1% rel. SEM).

struct TimePoint {
    double t;
    double sx, sx_sem;
    double sy, sy_sem;
    double sz, sz_sem;
};

static TimePoint run_at_time(std::function<DenseOpenSystem()> sys_factory,
                              const StateVector& psi0,
                              double t_measure,
                              unsigned int seed_offset)
{
    DenseOperator Sx = make_sigma_x();
    DenseOperator Sy = make_sigma_y();
    DenseOperator Sz = make_sigma_z();

    std::vector<ObservableDef> obs;
    obs.push_back(make_operator_observable("sx", Sx));
    obs.push_back(make_operator_observable("sy", Sy));
    obs.push_back(make_operator_observable("sz", Sz));

    StoppingCriteria sc;
    sc.min_trajectories = 200;
    sc.max_trajectories = 5000;
    sc.target_rel_sem   = 0.02;   // 2% relative SEM

    EnsembleConfig ec;
    ec.global_seed           = static_cast<Seed>(42 + seed_offset);
    ec.diag_level            = DiagnosticLevel::None;
    ec.propagator.dt_initial = 1e-3;

    auto sim = Simulation::make_dense_dopri45(
        sys_factory(),  // construct fresh system for this run
        std::move(obs), sc, ec);

    auto result = sim.run(psi0, 0.0, t_measure);

    TimePoint tp;
    tp.t      = t_measure;
    tp.sx     = result.observables[0].mean;
    tp.sx_sem = result.observables[0].sem;
    tp.sy     = result.observables[1].mean;
    tp.sy_sem = result.observables[1].sem;
    tp.sz     = result.observables[2].mean;
    tp.sz_sem = result.observables[2].sem;
    return tp;
}

// ── Exact density matrix reference (Lindblad master equation) ───────────────
//
// We integrate the full 2×2 density matrix directly under the Lindblad
// master equation using RK4. This is the same as QuTiP mesolve.
//
// ρ = [[ρ_ee, ρ_eg], [ρ_ge, ρ_gg]]  (ρ_ge = conj(ρ_eg))
// dρ/dt = -i[H,ρ] + Σₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})
//
// State vector for integration: [ρ_ee, Re(ρ_eg), Im(ρ_eg)]
// (ρ_gg = 1 - ρ_ee, ρ_ge = conj(ρ_eg))

struct DMState {
    double rho_ee;       // excited population
    double rho_eg_re;    // Re(ρ_eg) = coherence real part
    double rho_eg_im;    // Im(ρ_eg) = coherence imaginary part

    // Expectation values: <σ_x>=2Re(ρ_eg), <σ_y>=2Im(ρ_eg), <σ_z>=ρ_ee-ρ_gg
    double sx() const { return 2.0 * rho_eg_re; }
    double sy() const { return -2.0 * rho_eg_im; }  // note: σ_y has -i
    double sz() const { return 2.0 * rho_ee - 1.0; }
};

// Note on σ_y convention:
// σ_y = [[0,-i],[i,0]], so <σ_y> = -2*Im(ρ_eg) + 2i*Re(ρ_eg)... let's be careful.
// <σ_y> = Tr(σ_y ρ) = σ_y[0,1]*ρ[1,0] + σ_y[1,0]*ρ[0,1]
//        = (-i)*ρ_ge + (i)*ρ_eg = i*(ρ_eg - ρ_ge) = i*(ρ_eg - conj(ρ_eg))
//        = i*2i*Im(ρ_eg) = -2*Im(ρ_eg)
// So <σ_y> = -2*Im(ρ_eg). The sign above is correct.

static DMState dm_rhs(const DMState& s) {
    // H = ω(cos(θ)σ_z + sin(θ)σ_x)
    // H = [[ω cos θ, ω sin θ],[ω sin θ, -ω cos θ]]
    const double H_ee = OMEGA * std::cos(THETA);   // H[0,0]
    const double H_gg = -H_ee;                      // H[1,1]
    const double H_eg = OMEGA * std::sin(THETA);   // H[0,1] = H[1,0] (real)

    const double rho_gg = 1.0 - s.rho_ee;
    // ρ_eg = rho_eg_re + i*rho_eg_im
    // ρ_ge = rho_eg_re - i*rho_eg_im

    // -i[H,ρ] contribution to dρ/dt:
    // [H,ρ]_ee = H_eg*ρ_ge - ρ_eg*H_ge = H_eg*(ρ_ge - ρ_eg) = H_eg*(-2i*Im(ρ_eg))
    // d(ρ_ee)/dt|_unitary = -i * H_eg * (-2i*Im(ρ_eg)) = -2 * H_eg * Im(ρ_eg)
    double drho_ee = -2.0 * H_eg * s.rho_eg_im;

    // [H,ρ]_eg = H_ee*ρ_eg - ρ_eg*H_gg + H_eg*(ρ_gg - ρ_ee)
    //          = (H_ee - H_gg)*ρ_eg + H_eg*(ρ_gg - ρ_ee)
    // d(ρ_eg)/dt|_unitary = -i * [(H_ee - H_gg)*ρ_eg + H_eg*(ρ_gg - ρ_ee)]
    const double H_diff = H_ee - H_gg;   // = 2ω cos θ
    const double sz_val = s.rho_ee - rho_gg;
    // -i * (H_diff * (re+i*im) + H_eg * (-sz_val))
    // = -i * H_diff * re + H_diff * im - (-i * H_eg * sz_val)
    // = (-H_diff*im - H_eg*sz_val) + i*(H_diff*re) ... let me be careful:
    // -i*(A_re + i*A_im) = A_im - i*A_re
    // A = H_diff*(rho_eg_re + i*rho_eg_im) + H_eg*(-sz_val)
    // A_re = H_diff*rho_eg_re - H_eg*sz_val
    // A_im = H_diff*rho_eg_im
    // -i*A: Re part = A_im = H_diff*rho_eg_im
    //       Im part = -A_re = -H_diff*rho_eg_re + H_eg*sz_val
    double drho_eg_re =  H_diff * s.rho_eg_im;
    double drho_eg_im = -H_diff * s.rho_eg_re + H_eg * sz_val;

    // Lindblad dissipators:
    // L1 = sqrt(γ₁(n+1)) σ_-:  D[L1]ρ_ee = -γ₁(n+1)*ρ_ee
    //                           D[L1]ρ_eg = -γ₁(n+1)/2 * ρ_eg
    const double r1_em = GAMMA1 * (N_TH + 1.0);
    drho_ee    -= r1_em * s.rho_ee;
    drho_eg_re -= 0.5 * r1_em * s.rho_eg_re;
    drho_eg_im -= 0.5 * r1_em * s.rho_eg_im;

    // L2 = sqrt(γ₁ n_th) σ_+: D[L2]ρ_ee = +γ₁n*ρ_gg
    //                          D[L2]ρ_eg = -γ₁n/2 * ρ_eg
    const double r1_abs = GAMMA1 * N_TH;
    drho_ee    += r1_abs * rho_gg;
    drho_eg_re -= 0.5 * r1_abs * s.rho_eg_re;
    drho_eg_im -= 0.5 * r1_abs * s.rho_eg_im;

    // L3 = sqrt(γ₂) σ_z: D[L3]ρ_ee = 0 (σ_z is diagonal)
    //                     D[L3]ρ_eg = -2γ₂ * ρ_eg
    drho_eg_re -= 2.0 * GAMMA2 * s.rho_eg_re;
    drho_eg_im -= 2.0 * GAMMA2 * s.rho_eg_im;

    return {drho_ee, drho_eg_re, drho_eg_im};
}

static DMState dm_rk4_step(double dt, DMState s) {
    auto add = [](const DMState& a, const DMState& b, double h) -> DMState {
        return {a.rho_ee    + h*b.rho_ee,
                a.rho_eg_re + h*b.rho_eg_re,
                a.rho_eg_im + h*b.rho_eg_im};
    };
    DMState k1 = dm_rhs(s);
    DMState k2 = dm_rhs(add(s, k1, dt/2));
    DMState k3 = dm_rhs(add(s, k2, dt/2));
    DMState k4 = dm_rhs(add(s, k3, dt));
    return {s.rho_ee    + dt/6*(k1.rho_ee    + 2*k2.rho_ee    + 2*k3.rho_ee    + k4.rho_ee),
            s.rho_eg_re + dt/6*(k1.rho_eg_re + 2*k2.rho_eg_re + 2*k3.rho_eg_re + k4.rho_eg_re),
            s.rho_eg_im + dt/6*(k1.rho_eg_im + 2*k2.rho_eg_im + 2*k3.rho_eg_im + k4.rho_eg_im)};
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf(
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  LiQuID vs QuTiP: Qubit Dynamics (tutorial 003)             ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  H = ω(cos(θ)σ_z + sin(θ)σ_x)  ω=2π θ=0.2π               ║\n"
        "║  L1=√(γ₁(n+1))σ_-  L2=√(γ₁n)σ_+  L3=√γ₂σ_z              ║\n"
        "║  γ₁=0.5  γ₂=0.2  n_th=0.5  |ψ₀⟩=|0⟩                      ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n\n");

    // Initial state: |0⟩ = excited (matches QuTiP basis(2,0))
    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};

    // Time grid: 200 points matching QuTiP linspace(0, 10, 200)
    std::vector<double> times(N_TIMES);
    for (int k = 0; k < N_TIMES; ++k)
        times[k] = T_MAX * k / (N_TIMES - 1);

    // Exact density matrix reference (Lindblad master equation, RK4 integration)
    // Initial state |0⟩: ρ_ee=1, ρ_eg=0
    std::vector<DMState> analytic(N_TIMES);
    {
        DMState s{1.0, 0.0, 0.0};   // ρ_ee=1, ρ_eg=0
        analytic[0] = s;
        const double dt_dm = 1e-4;  // Fine timestep to resolve ω=2π oscillations
        double t_now = 0.0;
        int k = 1;
        while (k < N_TIMES) {
            s = dm_rk4_step(dt_dm, s);
            t_now += dt_dm;
            if (t_now >= times[k] - dt_dm/2) {
                analytic[k] = s;
                ++k;
            }
        }
    }

    // Run LiQuID at each time point
    // We run a subset of 20 time points for speed, matching key QuTiP outputs
    // Full 200-point run would take ~200× longer — use 20 evenly spaced
    const int N_SAMPLE = 20;
    std::vector<TimePoint> liquid_results;
    liquid_results.reserve(N_SAMPLE);

    std::printf("Running LiQuID MCWF at %d time points...\n", N_SAMPLE);
    std::printf("(target 2%% relative SEM, up to 5000 trajectories per point)\n\n");

    for (int k = 0; k < N_SAMPLE; ++k) {
        const int time_idx = (k == N_SAMPLE-1)
            ? N_TIMES - 1
            : k * (N_TIMES / N_SAMPLE);
        const double t = times[time_idx];

        if (t < 1e-12) {
            // t=0: return initial state exactly
            liquid_results.push_back({t, 0.0,0.0, 0.0,0.0, 1.0,0.0});
            continue;
        }

        std::printf("  t = %5.2f ...  ", t); std::fflush(stdout);
        auto tp = run_at_time(make_qubit_system, psi0, t, static_cast<unsigned>(k * 1000));
        liquid_results.push_back(tp);
        std::printf("<sz>=%.4f ± %.4f\n", tp.sz, tp.sz_sem);
    }

    // ── Print comparison table ────────────────────────────────────────────────
    std::printf("\n");
    std::printf("%-8s  %-12s %-12s  %-12s %-12s  %-12s %-12s\n",
        "t",
        "<sz> LiQ", "Bloch ref",
        "<sx> LiQ", "Bloch ref",
        "<sy> LiQ", "Bloch ref");
    std::printf("%s\n", std::string(90, '-').c_str());

    double max_err_sz = 0.0, max_err_sx = 0.0, max_err_sy = 0.0;
    for (std::size_t k = 0; k < liquid_results.size(); ++k) {
        const auto& lp = liquid_results[k];
        // Find analytic value at this t
        int a_idx = static_cast<int>(std::round(lp.t / T_MAX * (N_TIMES-1)));
        a_idx = std::max(0, std::min(a_idx, N_TIMES-1));
        const DMState& ds = analytic[a_idx];

        const double err_sz = std::abs(lp.sz - ds.sz());
        const double err_sx = std::abs(lp.sx - ds.sx());
        const double err_sy = std::abs(lp.sy - ds.sy());
        if (err_sz > max_err_sz) max_err_sz = err_sz;
        if (err_sx > max_err_sx) max_err_sx = err_sx;
        if (err_sy > max_err_sy) max_err_sy = err_sy;

        std::printf("%8.3f  %+9.4f    %+9.4f    %+9.4f    %+9.4f    %+9.4f    %+9.4f\n",
            lp.t,
            lp.sz, ds.sz(),
            lp.sx, ds.sx(),
            lp.sy, ds.sy());
    }

    std::printf("\nMax |LiQuID - Bloch|: σ_z=%.4f  σ_x=%.4f  σ_y=%.4f\n\n",
        max_err_sz, max_err_sx, max_err_sy);

    // ── Save CSV for plotting ─────────────────────────────────────────────────
    // File 1: LiQuID results (sparse time grid, with SEM)
    {
        FILE* f = std::fopen("qubit_dynamics_liquid.csv", "w");
        std::fprintf(f, "t,sx,sx_sem,sy,sy_sem,sz,sz_sem\n");
        for (const auto& lp : liquid_results)
            std::fprintf(f, "%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                lp.t, lp.sx, lp.sx_sem, lp.sy, lp.sy_sem, lp.sz, lp.sz_sem);
        std::fclose(f);
    }

    // File 2: Bloch equation reference (full 200-point grid, for comparison)
    {
        FILE* f = std::fopen("qubit_dynamics_bloch.csv", "w");
        std::fprintf(f, "t,sx,sy,sz\n");
        for (int k = 0; k < N_TIMES; ++k)
            std::fprintf(f, "%.6f,%.8f,%.8f,%.8f\n",
                times[k], analytic[k].sx(), analytic[k].sy(), analytic[k].sz());
        std::fclose(f);
    }

    std::printf("Saved:\n");
    std::printf("  qubit_dynamics_liquid.csv — LiQuID MCWF results with SEM\n");
    std::printf("  qubit_dynamics_bloch.csv  — Bloch equation reference (200 pts)\n\n");

    return 0;
}
