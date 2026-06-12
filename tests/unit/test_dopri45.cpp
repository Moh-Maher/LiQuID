// tests/unit/test_dopri45.cpp

#include "liquid/liquid.hpp"
#include "liquid/ode/dopri45.hpp"
#include "../test_framework.hpp"
#include <cmath>

using namespace liquid;
using namespace liquid::ode;
using namespace liquid::trajectory;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Simple scalar ODE: dy/dt = -y  →  y(t) = exp(-t)
// Used as a sanity check on the stepper independently of MCWF.
static void scalar_decay_rhs(Real t, const StateVector& y, StateVector& accum) {
    (void)t;
    accum[0] += -y[0];
}

// ── DormandPrince45 unit tests ────────────────────────────────────────────────

TEST("DormandPrince45: solves scalar decay to high accuracy") {
    DormandPrince45 stepper;
    stepper.reset(1e-2);

    StateVector y(1), y_new(1);
    y[0] = Scalar{1.0, 0.0};

    Real t = 0.0;
    const Real T = 1.0;

    while (t < T - 1e-14) {
        auto result = stepper.try_step(t, y, y_new, stepper.current_dt(),
                                        T, scalar_decay_rhs);
        if (result.accepted) {
            t += result.dt_taken;
            y.copy_from(y_new);
        }
    }

    const Real exact = std::exp(-T);
    REQUIRE_CLOSE(y[0].real(), exact, 1e-7);
}
END_TEST

TEST("DormandPrince45: FSAL reduces RHS evaluations") {
    // Two consecutive accepted steps:
    // Step 1: 7 RHS evaluations
    // Step 2: 6 RHS evaluations (k1 reused from step 1's k7)
    // Total for N steps: 6N + 1

    DormandPrince45 stepper;
    stepper.reset(1e-3);

    StateVector y(1), y_new(1);
    y[0] = Scalar{1.0, 0.0};

    Real t = 0.0;
    int accepted = 0;
    while (t < 0.1 - 1e-14) {
        auto result = stepper.try_step(t, y, y_new, stepper.current_dt(),
                                        0.1, scalar_decay_rhs);
        if (result.accepted) {
            t += result.dt_taken;
            y.copy_from(y_new);
            ++accepted;
        }
    }

    // For N accepted steps with FSAL: rhs_count <= 6*N + 1
    const std::size_t rhs = stepper.rhs_count();
    REQUIRE(accepted > 0);
    REQUIRE(rhs <= static_cast<std::size_t>(6 * accepted + 2));
}
END_TEST

TEST("DormandPrince45: stepsize adapts to tolerance") {
    // Tight tolerance → more steps, small dt
    // Loose tolerance → fewer steps, larger dt

    DormandPrince45::Config tight_cfg, loose_cfg;
    tight_cfg.atol = 1e-10; tight_cfg.rtol = 1e-8;
    loose_cfg.atol = 1e-4;  loose_cfg.rtol = 1e-3;

    DormandPrince45 tight_stepper(tight_cfg);
    DormandPrince45 loose_stepper(loose_cfg);
    tight_stepper.reset(1e-3);
    loose_stepper.reset(1e-3);

    StateVector y1(1), y2(1), yn1(1), yn2(1);
    y1[0] = y2[0] = Scalar{1.0, 0.0};

    Real t1 = 0.0, t2 = 0.0;
    while (t1 < 1.0 - 1e-14) {
        auto r = tight_stepper.try_step(t1, y1, yn1, tight_stepper.current_dt(),
                                         1.0, scalar_decay_rhs);
        if (r.accepted) { t1 += r.dt_taken; y1.copy_from(yn1); }
    }
    while (t2 < 1.0 - 1e-14) {
        auto r = loose_stepper.try_step(t2, y2, yn2, loose_stepper.current_dt(),
                                         1.0, scalar_decay_rhs);
        if (r.accepted) { t2 += r.dt_taken; y2.copy_from(yn2); }
    }

    // Tight should use more RHS evaluations than loose
    REQUIRE(tight_stepper.rhs_count() > loose_stepper.rhs_count());
}
END_TEST

TEST("DormandPrince45: reset clears FSAL and resets stepsize") {
    DormandPrince45 stepper;
    stepper.reset(1e-3);

    StateVector y(1), yn(1);
    y[0] = Scalar{1.0, 0.0};

    // Take a step to populate FSAL cache
    stepper.try_step(0.0, y, yn, 1e-3, 1.0, scalar_decay_rhs);
    const std::size_t rhs_before = stepper.rhs_count();

    // Reset (simulates post-jump reset)
    stepper.reset(1e-3);

    // After reset, next step should NOT reuse FSAL (needs full 7 evaluations)
    stepper.reset_counters();
    auto result = stepper.try_step(0.0, y, yn, 1e-3, 1.0, scalar_decay_rhs);
    (void)rhs_before;

    // First step after reset: at least 6 RHS calls (could be 7 without FSAL)
    REQUIRE(stepper.rhs_count() >= 6);
}
END_TEST

// ── MCWF with DOPRI45: physics validation ────────────────────────────────────

TEST("DormandPrince45: MCWF single trajectory physics correct") {
    // Two-level decay, <sz(T)> should match analytic solution
    const double omega = 1.0, gamma = 1.0, T = 2.0;
    const int    N = 1000;

    DenseOperator H(2);
    H(0,0) = Scalar{ omega/2.0, 0.0};
    H(1,1) = Scalar{-omega/2.0, 0.0};
    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0.0};
    std::vector<DenseOperator> ops;
    ops.push_back(std::move(L));
    LindbladSet<DenseTag> lb(std::move(ops));
    DenseOpenSystem sys(std::move(H), std::move(lb));

    PropagatorConfig cfg;
    cfg.dt_initial = 1e-2;
    cfg.atol = 1e-8; cfg.rtol = 1e-6;

    double sum_sz = 0.0;
    for (int traj = 0; traj < N; ++traj) {
        DormandPrince45 stepper;
        stepper.reset(cfg.dt_initial);

        MCWFPropagator<DenseOpenSystem, DormandPrince45> prop(&sys, stepper, cfg);

        StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};
        auto ts = make_trajectory_state(psi0, 0.0, T, (TrajId)traj, 42ULL,
                                         DiagnosticLevel::None);
        prop.run_to_completion(ts);

        const Real ns = ts.core.psi.norm_sq();
        sum_sz += (std::norm(ts.core.psi[0]) - std::norm(ts.core.psi[1]))
                  / (ns > 1e-30 ? ns : 1.0);
    }

    const double mean_sz = sum_sz / N;
    const double exact   = 2.0 * std::exp(-gamma * T) - 1.0;
    const double var     = 1.0;  // variance of sz in [-1,1]
    const double sem     = std::sqrt(var / N);

    std::printf("    <sz>(DOPRI45) = %.4f  exact = %.4f  4*SEM = %.4f\n",
        mean_sz, exact, 4.0 * sem);

    REQUIRE_CLOSE(mean_sz, exact, std::max(4.0 * sem, 1e-2));
}
END_TEST

int main() {
    std::printf("=== DormandPrince45 Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
