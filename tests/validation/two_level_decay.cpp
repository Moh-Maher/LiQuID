// tests/validation/two_level_decay.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Validation Test 1: Two-level spontaneous decay
//
// System:
//   H = (omega/2) sigma_z
//   L = sqrt(gamma) sigma_minus
//
// Exact analytic solution (initial state |e>):
//   <sigma_z(t)> = 2*exp(-gamma*t) - 1
//   <sigma_plus(t)> = 0  (off-diagonal: starts at 0, stays 0 from |e>)
//   rho_ee(t) = exp(-gamma*t)
//
// MCWF-specific predictions:
//   - Jump times are exponentially distributed: F(t) = 1 - exp(-gamma*t)
//   - After a jump, state is always |g> (only one channel)
//   - Between jumps, |psi> decays as exp(-(i*omega/2 + gamma/4)*t)|e>
//     (with appropriate normalization)
//
// This test verifies:
//   1. Single-trajectory jump statistics (Kolmogorov-Smirnov test)
//   2. Ensemble-averaged <sigma_z(t)> vs analytic solution
//   3. Norm decay between jumps
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace liquid;
using namespace liquid::trajectory;
using namespace liquid::ode;

// ── System construction ───────────────────────────────────────────────────────

static DenseOpenSystem make_two_level_decay(double omega, double gamma) {
    // H = (omega/2) sigma_z
    DenseOperator H(2);
    H(0, 0) = Scalar{ omega / 2.0, 0.0};
    H(1, 1) = Scalar{-omega / 2.0, 0.0};

    // L = sqrt(gamma) sigma_minus
    DenseOperator L(2);
    L(1, 0) = Scalar{std::sqrt(gamma), 0.0};

    std::vector<DenseOperator> ops;
    ops.push_back(std::move(L));
    LindbladSet<DenseTag> lindblad(std::move(ops));

    return DenseOpenSystem(std::move(H), std::move(lindblad));
}

// ── Excited-state initial state ───────────────────────────────────────────────

static StateVector excited_state() {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};
    return psi;
}

// ── Observable: <sigma_z> on a (possibly unnormalized) MCWF state ────────────
// MCWF states between jumps are intentionally unnormalized.
// Correct expectation value: <O> = <psi|O|psi> / <psi|psi>

static Real measure_sigma_z(const StateVector& psi) {
    const Real norm_sq = psi.norm_sq();
    if (norm_sq < 1e-30) return -1.0;
    return (std::norm(psi[0]) - std::norm(psi[1])) / norm_sq;
}

// ── Analytic reference ────────────────────────────────────────────────────────

static Real analytic_sigma_z(double t, double gamma) {
    return 2.0 * std::exp(-gamma * t) - 1.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1a: Norm decreases monotonically between jumps
// (physical requirement from non-Hermitian evolution)
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: norm decreases monotonically between jumps") {
    const double omega = 1.0, gamma = 1.0;
    auto sys = make_two_level_decay(omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 1e-3;

    // Run several trajectories and verify that in each one,
    // the norm is monotonically non-increasing between jumps.
    // We also verify that at least some trajectory shows actual decrease.
    bool any_trajectory_showed_decrease = false;

    for (int seed_offset = 0; seed_offset < 20; ++seed_offset) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        auto ts = make_trajectory_state(excited_state(), 0.0, 2.0,
                                         static_cast<TrajId>(seed_offset),
                                         static_cast<Seed>(seed_offset * 137 + 17),
                                         DiagnosticLevel::Full);

        // We need to sample norm at fine intervals to check monotonicity.
        // Run step by step using small t_target increments.
        Real prev_norm_sq = 1.0;
        bool found_decrease_this_traj = false;
        const Real dt_check = 0.05;
        Real t_check = dt_check;
        StepOutcome outcome = StepOutcome::Advanced;

        while (outcome != StepOutcome::Completed && outcome != StepOutcome::Failed
               && t_check <= 2.0 + 1e-10) {
            outcome = prop.advance(ts, t_check);

            if (outcome == StepOutcome::JumpOccurred) {
                // Post-jump: norm must be 1 (renormalized)
                REQUIRE_CLOSE(ts.core.psi.norm_sq(), 1.0, 1e-12);
                prev_norm_sq = 1.0;
            } else {
                // Between jumps: norm must be non-increasing
                const Real norm_sq = ts.core.psi.norm_sq();
                REQUIRE(norm_sq <= prev_norm_sq + 1e-9);
                if (norm_sq < prev_norm_sq - 1e-9) found_decrease_this_traj = true;
                prev_norm_sq = norm_sq;
            }
            t_check += dt_check;
        }

        if (found_decrease_this_traj) any_trajectory_showed_decrease = true;
    }

    // At least one trajectory must have shown actual norm decrease.
    // (If all 20 trajectories jumped immediately and stayed in |g>, something is wrong.)
    REQUIRE(any_trajectory_showed_decrease);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 1b: Single-trajectory jump statistics
// With gamma=1, omega=0, the jump time distribution is Exp(1).
// We run N_traj single-trajectory simulations and collect first-jump times,
// then verify they are approximately exponentially distributed.
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: jump times are exponentially distributed") {
    const double omega = 0.0;  // No precession — cleaner jump statistics
    const double gamma = 1.0;
    const double T     = 5.0;  // Long enough to catch most jumps
    const int N_traj   = 2000;

    auto sys = make_two_level_decay(omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 1e-3;

    std::vector<Real> first_jump_times;
    first_jump_times.reserve(N_traj);

    for (int traj = 0; traj < N_traj; ++traj) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        auto ts = make_trajectory_state(excited_state(), 0.0, T,
                                         static_cast<TrajId>(traj), 42,
                                         DiagnosticLevel::Summary);

        prop.run_to_completion(ts);

        // Record time of first jump (if any)
        if (!ts.diag().jumps.empty()) {
            first_jump_times.push_back(ts.diag().jumps[0].t);
        }
    }

    // Most trajectories should have jumped in [0, T=5] with gamma=1
    // Expected fraction with no jump: exp(-gamma*T) = exp(-5) ≈ 0.007
    const double frac_no_jump = 1.0 - static_cast<double>(first_jump_times.size()) / N_traj;
    REQUIRE_CLOSE(frac_no_jump, std::exp(-gamma * T), 0.02);  // within 2%

    // Kolmogorov-Smirnov test against Exp(gamma) distribution.
    // CDF: F(t) = 1 - exp(-gamma*t)
    // KS statistic D = max |F_empirical(t) - F_theoretical(t)|
    // For N=2000 at alpha=0.001: critical value D_crit ≈ 0.044
    std::sort(first_jump_times.begin(), first_jump_times.end());
    const int N = static_cast<int>(first_jump_times.size());

    double D = 0.0;
    for (int i = 0; i < N; ++i) {
        const double t = first_jump_times[i];
        const double F_theoretical = 1.0 - std::exp(-gamma * t);
        const double F_empirical_upper = static_cast<double>(i + 1) / N;
        const double F_empirical_lower = static_cast<double>(i) / N;

        D = std::max(D, std::abs(F_empirical_upper - F_theoretical));
        D = std::max(D, std::abs(F_empirical_lower - F_theoretical));
    }

    // KS critical value at alpha=0.001 for N=2000: ≈ 0.044
    // We use 0.05 for safety margin
    REQUIRE_LT(D, 0.05);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 1c: Ensemble average <sigma_z(t)> vs analytic solution
// This is the primary physics validation.
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: ensemble <sigma_z(t)> matches analytic solution") {
    const double omega   = 1.0;
    const double gamma   = 1.0;
    const double T       = 3.0;
    const int    N_traj  = 3000;
    const int    N_times = 6;  // Measurement times

    // Times at which we measure <sigma_z>
    const double t_measure[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    auto sys = make_two_level_decay(omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 1e-3;

    // Accumulate <sigma_z> at each measurement time
    std::vector<double> sum_sz(N_times, 0.0);
    std::vector<double> sum_sz2(N_times, 0.0);

    for (int traj = 0; traj < N_traj; ++traj) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        auto ts = make_trajectory_state(excited_state(), 0.0, T,
                                         static_cast<TrajId>(traj),
                                         12345ULL,
                                         DiagnosticLevel::None);

        int time_idx = 0;
        StepOutcome outcome = StepOutcome::Advanced;

        while (outcome != StepOutcome::Completed &&
               outcome != StepOutcome::Failed  &&
               time_idx < N_times) {

            outcome = prop.advance(ts, t_measure[time_idx]);

            if (outcome == StepOutcome::Advanced ||
                outcome == StepOutcome::Completed) {
                // At t_measure[time_idx], record <sigma_z>
                const double sz = measure_sigma_z(ts.core.psi);
                sum_sz[time_idx]  += sz;
                sum_sz2[time_idx] += sz * sz;
                ++time_idx;
            }
            // If JumpOccurred, we don't advance time_idx — keep advancing
            // until we reach the measurement time
        }

        // Fill remaining times if trajectory completed early (shouldn't happen)
        for (; time_idx < N_times; ++time_idx) {
            const double sz = measure_sigma_z(ts.core.psi);
            sum_sz[time_idx]  += sz;
            sum_sz2[time_idx] += sz * sz;
        }
    }

    // Compare means to analytic solution
    // Tolerance: 3 * SEM (should pass ~99.7% of the time)
    for (int k = 0; k < N_times; ++k) {
        const double mean_sz = sum_sz[k] / N_traj;
        const double var_sz  = sum_sz2[k] / N_traj - mean_sz * mean_sz;
        const double sem_sz  = std::sqrt(std::max(0.0, var_sz) / N_traj);

        const double exact = analytic_sigma_z(t_measure[k], gamma);

        // Allow 4 SEM to be safe (probability of failure ≈ 0.006%)
        // Also enforce absolute tolerance floor of 1e-2
        const double tol = std::max(4.0 * sem_sz, 1e-2);

        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "t=%.1f: MCWF=%.4f, exact=%.4f, diff=%.4f, tol=%.4f",
            t_measure[k], mean_sz, exact, std::abs(mean_sz - exact), tol);

        // Print for diagnostic visibility
        std::printf("    %s\n", msg);

        REQUIRE_CLOSE(mean_sz, exact, tol);
    }
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 1d: Post-jump state is always |g>
// With only one jump channel (sigma_minus), every jump must send
// the system to the ground state.
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: post-jump state is always ground state") {
    const double omega = 1.0, gamma = 1.0, T = 10.0;
    const int N_traj = 200;

    auto sys = make_two_level_decay(omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 1e-3;

    for (int traj = 0; traj < N_traj; ++traj) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        auto ts = make_trajectory_state(excited_state(), 0.0, T,
                                         static_cast<TrajId>(traj), 777,
                                         DiagnosticLevel::Summary);

        StepOutcome outcome = StepOutcome::Advanced;
        while (outcome != StepOutcome::Completed && outcome != StepOutcome::Failed) {
            outcome = prop.advance(ts, ts.core.t_final);

            if (outcome == StepOutcome::JumpOccurred) {
                // State must be |g> = [0, 1] after sigma_minus jump
                REQUIRE_CLOSE(ts.core.psi[0].real(), 0.0, 1e-12);
                REQUIRE_CLOSE(ts.core.psi[0].imag(), 0.0, 1e-12);
                REQUIRE_CLOSE(ts.core.psi.norm_sq(), 1.0, 1e-12);
            }
        }
    }
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 1e: Ground state never jumps
// Once in |g>, sigma_minus|g> = 0, so no jump should ever occur.
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation: ground state produces no jumps") {
    const double omega = 1.0, gamma = 1.0, T = 5.0;

    auto sys = make_two_level_decay(omega, gamma);

    PropagatorConfig cfg;
    cfg.dt_initial = 1e-3;

    // Start in ground state
    StateVector ground(2);
    ground[1] = Scalar{1.0, 0.0};

    for (int traj = 0; traj < 50; ++traj) {
        RK4Stepper stepper(cfg.dt_initial);
        MCWFPropagator<DenseOpenSystem, RK4Stepper> prop(&sys, stepper, cfg);

        auto ts = make_trajectory_state(ground, 0.0, T,
                                         static_cast<TrajId>(traj), 42,
                                         DiagnosticLevel::Summary);

        prop.run_to_completion(ts);

        REQUIRE(ts.diag().total_jumps == 0);
        REQUIRE(ts.is_completed());
    }
}
END_TEST

int main() {
    std::printf("=== Validation Test 1: Two-Level Spontaneous Decay ===\n\n");
    return RUN_ALL_TESTS();
}
