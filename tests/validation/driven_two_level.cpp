// tests/validation/driven_two_level.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Validation Test 2: Driven two-level system (resonant, rotating frame)
//
// System (rotating frame, resonance):
//   H = Omega * sigma_x  (Omega = Rabi frequency)
//   L = sqrt(gamma) * sigma_minus
//
// Steady-state solution (rho_ee^ss at resonance, rotating wave approximation):
//   rho_ee^ss = (Omega/gamma)^2 / (1 + 2*(Omega/gamma)^2)
//
// This test verifies:
//   1. Steady state is reached and matches analytic formula
//   2. MCWF with coherent driving + dissipation works correctly
//   3. The interplay between unitary evolution and quantum jumps is correct
//
// Physical interpretation:
//   The Rabi drive continuously excites the system;
//   spontaneous emission continuously returns it to ground.
//   The steady state balances these two processes.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"

#include <cmath>
#include <vector>

using namespace liquid;
using namespace liquid::trajectory;
using namespace liquid::ode;

// ── System construction ───────────────────────────────────────────────────────

static DenseOpenSystem make_driven_two_level(double Omega, double gamma) {
    // H = Omega * sigma_x = Omega * [[0,1],[1,0]]
    DenseOperator H(2);
    H(0, 1) = Scalar{Omega, 0.0};
    H(1, 0) = Scalar{Omega, 0.0};

    // L = sqrt(gamma) * sigma_minus
    DenseOperator L(2);
    L(1, 0) = Scalar{std::sqrt(gamma), 0.0};

    std::vector<DenseOperator> ops;
    ops.push_back(std::move(L));
    LindbladSet<DenseTag> lindblad(std::move(ops));

    return DenseOpenSystem(std::move(H), std::move(lindblad));
}

// Correct MCWF expectation: normalize before measuring
static Real measure_rho_ee(const StateVector& psi) {
    const Real norm_sq = psi.norm_sq();
    if (norm_sq < 1e-30) return 0.0;
    return std::norm(psi[0]) / norm_sq;
}

static Real analytic_rho_ee_ss(double Omega, double gamma) {
    // Steady state for H = Omega*sigma_x, L = sqrt(gamma)*sigma_minus
    // Derived from Bloch equations at steady state:
    //   rho_ee^ss = (4*Omega^2/gamma^2) / (1 + 8*Omega^2/gamma^2)
    // Note: factor of 4 (not 1) because H = Omega*sigma_x (not Omega/2 * sigma_x)
    const double ratio_sq = (Omega * Omega) / (gamma * gamma);
    return 4.0 * ratio_sq / (1.0 + 8.0 * ratio_sq);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2a: Steady-state excited population for weak drive (Omega = 0.1*gamma)
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: driven system steady state, weak drive") {
    const double gamma = 1.0;
    const double Omega = 0.1 * gamma;  // Weak drive: rho_ee^ss ≈ 0.02/(1.04) ≈ 0.0196
    const double T_ss  = 20.0;         // Long enough to reach steady state
    const double T_avg = 5.0;          // Averaging window at end
    const int    N_traj = 3000;

    auto sys = make_driven_two_level(Omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 5e-4;  // Smaller dt: Rabi oscillations at rate Omega

    const double exact_ss = analytic_rho_ee_ss(Omega, gamma);

    // Run ensemble and average rho_ee in the steady-state window [T_ss, T_ss+T_avg]
    double sum_rho_ee = 0.0;
    double sum_rho_ee2 = 0.0;
    int N_samples = 0;

    const double t_measure_start = T_ss;
    const double t_measure_end   = T_ss + T_avg;
    const double dt_measure      = 0.5;

    for (int traj = 0; traj < N_traj; ++traj) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        StateVector psi0(2);
        psi0[0] = Scalar{1.0, 0.0};  // Start in excited state

        auto ts = make_trajectory_state(psi0, 0.0, t_measure_end,
                                         static_cast<TrajId>(traj), 54321ULL,
                                         DiagnosticLevel::None);

        // Advance to steady-state window
        StepOutcome outcome = StepOutcome::Advanced;
        while (outcome != StepOutcome::Completed && outcome != StepOutcome::Failed) {
            outcome = prop.advance(ts, t_measure_end);
        }

        // For simplicity in Phase 1, measure at the end time
        const double rho_ee = measure_rho_ee(ts.core.psi);
        sum_rho_ee  += rho_ee;
        sum_rho_ee2 += rho_ee * rho_ee;
        ++N_samples;
    }

    const double mean_rho = sum_rho_ee / N_samples;
    const double var_rho  = sum_rho_ee2 / N_samples - mean_rho * mean_rho;
    const double sem_rho  = std::sqrt(std::max(0.0, var_rho) / N_samples);

    std::printf("    Omega/gamma = %.2f\n", Omega / gamma);
    std::printf("    rho_ee(MCWF) = %.4f +/- %.4f\n", mean_rho, sem_rho);
    std::printf("    rho_ee(exact) = %.4f\n", exact_ss);
    std::printf("    |diff| = %.4f (tol = %.4f)\n",
                std::abs(mean_rho - exact_ss), std::max(5.0 * sem_rho, 5e-3));

    // 5 SEM tolerance or 5e-3 absolute, whichever is larger
    const double tol = std::max(5.0 * sem_rho, 5e-3);
    REQUIRE_CLOSE(mean_rho, exact_ss, tol);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 2b: Steady-state for strong drive (Omega = gamma)
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: driven system steady state, strong drive") {
    const double gamma = 1.0;
    const double Omega = 1.0 * gamma;  // Strong drive: rho_ee^ss = 1/3
    const double T_ss  = 15.0;
    const int    N_traj = 3000;

    auto sys = make_driven_two_level(Omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 5e-4;

    const double exact_ss = analytic_rho_ee_ss(Omega, gamma);

    double sum_rho_ee = 0.0, sum_rho_ee2 = 0.0;

    for (int traj = 0; traj < N_traj; ++traj) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        StateVector psi0(2);
        psi0[0] = Scalar{1.0, 0.0};

        auto ts = make_trajectory_state(psi0, 0.0, T_ss,
                                         static_cast<TrajId>(traj), 99999ULL,
                                         DiagnosticLevel::None);

        prop.run_to_completion(ts);

        const double rho_ee = measure_rho_ee(ts.core.psi);
        sum_rho_ee  += rho_ee;
        sum_rho_ee2 += rho_ee * rho_ee;
    }

    const double mean_rho = sum_rho_ee / N_traj;
    const double var_rho  = sum_rho_ee2 / N_traj - mean_rho * mean_rho;
    const double sem_rho  = std::sqrt(std::max(0.0, var_rho) / N_traj);

    std::printf("    Omega/gamma = %.2f\n", Omega / gamma);
    std::printf("    rho_ee(MCWF) = %.4f +/- %.4f\n", mean_rho, sem_rho);
    std::printf("    rho_ee(exact) = %.4f\n", exact_ss);

    const double tol = std::max(5.0 * sem_rho, 5e-3);
    REQUIRE_CLOSE(mean_rho, exact_ss, tol);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 2c: Closed system (gamma=0 limit) — Rabi oscillations
// Without dissipation, we should see pure Rabi oscillations.
// <sigma_z(t)> = -cos(2*Omega*t) for initial state |e> and H = Omega*sigma_x
//
// NOTE: gamma=0 means no Lindblad operators, but our LindbladSet
// requires at least one operator. We use a very small gamma to approximate.
// The test verifies the regime where coherent dynamics dominate.
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: weakly dissipative Rabi oscillations") {
    const double gamma = 0.01;  // Very weak dissipation: almost closed
    const double Omega = 1.0;
    const double T     = 1.0;   // One Rabi period: T_Rabi = pi/(2*Omega) ≈ 1.57
                                 // At T=1: <sz> = -cos(2*Omega*T) = -cos(2) ≈ -0.416
    const int    N_traj = 5000;

    auto sys = make_driven_two_level(Omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 1e-3;

    double sum_sz = 0.0, sum_sz2 = 0.0;

    for (int traj = 0; traj < N_traj; ++traj) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        StateVector psi0(2);
        psi0[0] = Scalar{1.0, 0.0};  // Start in |e>

        auto ts = make_trajectory_state(psi0, 0.0, T,
                                         static_cast<TrajId>(traj), 11111ULL,
                                         DiagnosticLevel::None);

        prop.run_to_completion(ts);

        const double norm_sq = ts.core.psi.norm_sq();
        const double sz = norm_sq > 1e-30
            ? (std::norm(ts.core.psi[0]) - std::norm(ts.core.psi[1])) / norm_sq
            : -1.0;
        sum_sz  += sz;
        sum_sz2 += sz * sz;
    }

    const double mean_sz = sum_sz / N_traj;
    const double var_sz  = sum_sz2 / N_traj - mean_sz * mean_sz;
    const double sem_sz  = std::sqrt(std::max(0.0, var_sz) / N_traj);

    // Analytic (closed system, H = Omega*sigma_x, initial state |e>):
    // Starting at north pole, rotating around x-axis at angular rate 2*Omega:
    //   <sigma_z(t)> = cos(2*Omega*t) * exp(-gamma*t/2)
    // At t=1, Omega=1, gamma=0.01:
    //   cos(2) * exp(-0.005) ≈ -0.416 * 0.995 ≈ -0.414
    const double exact_closed = std::cos(2.0 * Omega * T) * std::exp(-gamma * T / 2.0);

    std::printf("    Omega=%.1f, gamma=%.3f, T=%.1f\n", Omega, gamma, T);
    std::printf("    <sz>(MCWF) = %.4f +/- %.4f\n", mean_sz, sem_sz);
    std::printf("    <sz>(approx analytic) = %.4f\n", exact_closed);

    // Tolerance: 5 SEM or 2% of range (|sz| ≤ 1)
    const double tol = std::max(5.0 * sem_sz, 0.02);
    REQUIRE_CLOSE(mean_sz, exact_closed, tol);
}
END_TEST

int main() {
    std::printf("=== Validation Test 2: Driven Two-Level System ===\n\n");
    return RUN_ALL_TESTS();
}
