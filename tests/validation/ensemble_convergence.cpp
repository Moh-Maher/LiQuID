// tests/validation/ensemble_convergence.cpp
// Phase 2 Validation: Ensemble statistics and convergence

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>
#include <vector>

using namespace liquid;
using namespace liquid::ensemble;
using namespace liquid::trajectory;
using namespace liquid::ode;

// ── Operator construction helpers ─────────────────────────────────────────────

static DenseOperator make_H_decay(double omega) {
    DenseOperator H(2);
    H(0,0) = Scalar{ omega / 2.0, 0.0};
    H(1,1) = Scalar{-omega / 2.0, 0.0};
    return H;
}

static DenseOperator make_L_decay(double gamma) {
    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0.0};
    return L;
}

static DenseOperator make_sigma_z() {
    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};
    return sz;
}

static DenseOperator make_rho_ee_op() {
    DenseOperator op(2);
    op(0,0) = Scalar{1.0, 0.0};
    return op;
}

static StateVector make_excited() {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};
    return psi;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: MCWF ensemble matches Lindblad analytic solution at multiple times
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 2: MCWF ensemble matches Lindblad solution") {
    const double omega = 1.0;
    const double gamma = 1.0;
    const int    N     = 5000;
    const double times[6] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    StateVector psi0 = make_excited();

    for (int k = 0; k < 6; ++k) {
        DenseOperator H  = make_H_decay(omega);
        DenseOperator L  = make_L_decay(gamma);
        DenseOperator sz = make_sigma_z();

        Simulation sim = SimulationBuilder{}
            .hamiltonian(std::move(H))
            .collapse_operator(std::move(L))
            .observe("sigma_z", std::move(sz))
            .seed(static_cast<Seed>(k * 1000 + 42))
            .dt(1e-3)
            .min_trajectories(N)
            .max_trajectories(N)
            .diagnostics(DiagnosticLevel::None)
            .build();

        EnsembleResult result = sim.run(psi0, 0.0, times[k]);

        const double mean_sz = result.observables[0].mean;
        const double sem_sz  = result.observables[0].sem;
        const double exact   = 2.0 * std::exp(-gamma * times[k]) - 1.0;
        const double tol     = std::max(4.0 * sem_sz, 1e-2);

        std::printf("    t=%.1f: MCWF=%.4f+/-%.4f exact=%.4f |diff|=%.4f tol=%.4f\n",
            times[k], mean_sz, sem_sz, exact,
            std::abs(mean_sz - exact), tol);

        REQUIRE_CLOSE(mean_sz, exact, tol);
    }
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: SEM scales as 1/sqrt(N)
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 2: SEM scales as 1/sqrt(N)") {
    const double omega   = 1.0;
    const double gamma   = 1.0;
    const double T       = 1.0;
    const int    N_small = 200;
    const int    N_large = 800;

    StateVector psi0 = make_excited();

    DenseOperator H1  = make_H_decay(omega);
    DenseOperator L1  = make_L_decay(gamma);
    DenseOperator sz1 = make_sigma_z();

    Simulation sim_small = SimulationBuilder{}
        .hamiltonian(std::move(H1))
        .collapse_operator(std::move(L1))
        .observe("sigma_z", std::move(sz1))
        .seed(1111)
        .dt(1e-3)
        .min_trajectories(N_small)
        .max_trajectories(N_small)
        .diagnostics(DiagnosticLevel::None)
        .build();

    EnsembleResult r_small = sim_small.run(psi0, 0.0, T);

    DenseOperator H2  = make_H_decay(omega);
    DenseOperator L2  = make_L_decay(gamma);
    DenseOperator sz2 = make_sigma_z();

    Simulation sim_large = SimulationBuilder{}
        .hamiltonian(std::move(H2))
        .collapse_operator(std::move(L2))
        .observe("sigma_z", std::move(sz2))
        .seed(2222)
        .dt(1e-3)
        .min_trajectories(N_large)
        .max_trajectories(N_large)
        .diagnostics(DiagnosticLevel::None)
        .build();

    EnsembleResult r_large = sim_large.run(psi0, 0.0, T);

    const double sem_small      = r_small.observables[0].sem;
    const double sem_large      = r_large.observables[0].sem;
    const double expected_ratio = std::sqrt(
        static_cast<double>(N_small) / static_cast<double>(N_large));
    const double actual_ratio   = sem_large / sem_small;

    std::printf("    SEM(N=%d)=%.4f  SEM(N=%d)=%.4f\n",
        N_small, sem_small, N_large, sem_large);
    std::printf("    ratio=%.4f  expected=%.4f\n", actual_ratio, expected_ratio);

    REQUIRE_CLOSE(actual_ratio, expected_ratio, 0.20);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Convergence monitor stops at target SEM
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 2: convergence monitor stops at target SEM") {
    const double omega          = 1.0;
    const double gamma          = 1.0;
    const double target_rel_sem = 0.05;

    DenseOperator H  = make_H_decay(omega);
    DenseOperator L  = make_L_decay(gamma);
    DenseOperator sz = make_sigma_z();

    Simulation sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .seed(7777)
        .dt(1e-3)
        .stop_at_sem(target_rel_sem)
        .min_trajectories(50)
        .max_trajectories(100000)
        .diagnostics(DiagnosticLevel::None)
        .build();

    StateVector psi0 = make_excited();
    EnsembleResult result = sim.run(psi0, 0.0, 1.0);

    const double achieved = result.observables[0].rel_sem;

    std::printf("    Stopped at N=%zu  rel_sem=%.4f  target=%.4f\n",
        result.total_trajectories, achieved, target_rel_sem);

    REQUIRE(result.convergence.decision == StoppingDecision::Converged);
    REQUIRE_LT(achieved, target_rel_sem * 1.1);
    REQUIRE(result.total_trajectories >= 50);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Multi-observable ensemble
// Consistency check: rho_ee = (1 + <sigma_z>) / 2
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 2: multi-observable ensemble") {
    const double omega = 0.0;
    const double gamma = 1.0;
    const double T     = 1.0;
    const int    N     = 3000;

    DenseOperator H      = make_H_decay(omega);
    DenseOperator L      = make_L_decay(gamma);
    DenseOperator sz     = make_sigma_z();
    DenseOperator rho_ee = make_rho_ee_op();

    Simulation sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .observe("rho_ee",  std::move(rho_ee))
        .seed(31415)
        .dt(1e-3)
        .min_trajectories(N)
        .max_trajectories(N)
        .diagnostics(DiagnosticLevel::None)
        .build();

    StateVector psi0 = make_excited();
    EnsembleResult result = sim.run(psi0, 0.0, T);

    REQUIRE(result.observables.size() == 2);
    REQUIRE(result.observables[0].name == "sigma_z");
    REQUIRE(result.observables[1].name == "rho_ee");

    const double sz_mean  = result.observables[0].mean;
    const double sz_sem   = result.observables[0].sem;
    const double rho_mean = result.observables[1].mean;
    const double rho_sem  = result.observables[1].sem;

    const double exact_sz  = 2.0 * std::exp(-gamma * T) - 1.0;
    const double exact_rho = std::exp(-gamma * T);

    std::printf("    <sz>   = %.4f+/-%.4f  exact=%.4f\n",
        sz_mean, sz_sem, exact_sz);
    std::printf("    rho_ee = %.4f+/-%.4f  exact=%.4f\n",
        rho_mean, rho_sem, exact_rho);

    REQUIRE_CLOSE(sz_mean,  exact_sz,  std::max(4.0 * sz_sem,  1e-2));
    REQUIRE_CLOSE(rho_mean, exact_rho, std::max(4.0 * rho_sem, 1e-2));

    // Consistency: rho_ee = (1 + <sz>) / 2
    REQUIRE_CLOSE(rho_mean, (1.0 + sz_mean) / 2.0, 0.01);
}
END_TEST

int main() {
    std::printf("=== Phase 2 Validation: Ensemble Convergence ===\n\n");
    return RUN_ALL_TESTS();
}

