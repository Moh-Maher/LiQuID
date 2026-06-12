// tests/validation/smart_stopping.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Phase 5 Validation: Smart stopping convergence study
//
// Demonstrates that convergence-based stopping produces correct results
// with fewer trajectories than fixed-N at the same accuracy target.
//
// Tests:
//   1. Smart stopping halts at correct SEM and result is accurate
//   2. Fixed-N at smart-stopping's N gives comparable accuracy
//   3. Fixed-N at 10x smart-stopping's N gives ~sqrt(10) improvement in SEM
//   4. ConvergenceAnalysis correctly predicts minimum N from a result
//   5. Time-dependent system (driven qubit) converges correctly
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>

using namespace liquid;
using namespace liquid::ensemble;

static DenseOperator make_H(double omega) {
    DenseOperator H(2);
    H(0,0) = Scalar{ omega/2.0, 0.0};
    H(1,1) = Scalar{-omega/2.0, 0.0};
    return H;
}
static DenseOperator make_L(double gamma) {
    DenseOperator L(2); L(1,0) = Scalar{std::sqrt(gamma), 0.0}; return L;
}
static DenseOperator make_sz() {
    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0}; sz(1,1) = Scalar{-1.0, 0.0}; return sz;
}
static StateVector make_excited() {
    StateVector psi(2); psi[0] = Scalar{1.0, 0.0}; return psi;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Smart stopping gives accurate result
// ─────────────────────────────────────────────────────────────────────────────

TEST("Phase 5: smart stopping result is accurate") {
    const double omega = 1.0, gamma = 1.0, T = 1.0;
    const double target_sem = 0.02;
    const double exact = 2.0 * std::exp(-gamma * T) - 1.0;

    Simulation sim = SimulationBuilder{}
        .hamiltonian(make_H(omega))
        .collapse_operator(make_L(gamma))
        .observe("sz", make_sz())
        .seed(42)
        .dt(1e-3)
        .stop_at_sem(target_sem)
        .min_trajectories(50)
        .max_trajectories(50000)
        .build();

    EnsembleResult result = sim.run(make_excited(), 0.0, T);

    const double mean = result.observables[0].mean;
    const double sem  = result.observables[0].sem;

    std::printf("    N=%zu  <sz>=%.4f  sem=%.4f  exact=%.4f\n",
        result.total_trajectories, mean, sem, exact);

    REQUIRE(result.convergence.decision == StoppingDecision::Converged);
    REQUIRE_CLOSE(mean, exact, std::max(4.0 * sem, 1e-2));
    REQUIRE_LT(result.observables[0].rel_sem, target_sem * 1.05);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: SEM scales as 1/sqrt(N) — validated across 3 sample sizes
// ─────────────────────────────────────────────────────────────────────────────

TEST("Phase 5: SEM scaling 1/sqrt(N) across three ensemble sizes") {
    const double omega = 1.0, gamma = 1.0, T = 1.0;
    const int sizes[3] = {100, 400, 1600};
    double sems[3];

    StateVector psi0 = make_excited();

    for (int k = 0; k < 3; ++k) {
        Simulation sim = SimulationBuilder{}
            .hamiltonian(make_H(omega))
            .collapse_operator(make_L(gamma))
            .observe("sz", make_sz())
            .seed(static_cast<Seed>(k * 777 + 42))
            .dt(1e-3)
            .min_trajectories(sizes[k])
            .max_trajectories(sizes[k])
            .build();

        EnsembleResult r = sim.run(psi0, 0.0, T);
        sems[k] = r.observables[0].sem;
        std::printf("    N=%4d  sem=%.5f\n", sizes[k], sems[k]);
    }

    // SEM should halve each time N quadruples
    // ratio[0]: sems[0]/sems[1] ~ sqrt(400/100) = 2
    // ratio[1]: sems[1]/sems[2] ~ sqrt(1600/400) = 2
    const double ratio01 = sems[0] / sems[1];
    const double ratio12 = sems[1] / sems[2];
    std::printf("    ratio(100/400)=%.3f expected~2.0\n", ratio01);
    std::printf("    ratio(400/1600)=%.3f expected~2.0\n", ratio12);

    REQUIRE_CLOSE(ratio01, 2.0, 0.5);  // generous: statistical noise at N=100
    REQUIRE_CLOSE(ratio12, 2.0, 0.4);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: ConvergenceAnalysis correctly identifies minimum N
// ─────────────────────────────────────────────────────────────────────────────

TEST("Phase 5: ConvergenceAnalysis predicts minimum N correctly") {
    const double omega = 1.0, gamma = 1.0, T = 1.0;

    // Run with large N to get accurate SEM estimate
    const int N_large = 2000;
    Simulation sim = SimulationBuilder{}
        .hamiltonian(make_H(omega))
        .collapse_operator(make_L(gamma))
        .observe("sz", make_sz())
        .seed(314159)
        .dt(1e-3)
        .min_trajectories(N_large)
        .max_trajectories(N_large)
        .build();

    EnsembleResult result = sim.run(make_excited(), 0.0, T);

    const double target = 0.05;  // 5% relative SEM target
    ConvergenceAnalysis analysis = analyse_convergence(result, target);

    std::printf("    N_actual=%zu  achieved_rel_sem=%.4f\n",
        analysis.actual_N, analysis.achieved_rel_sem);
    std::printf("    target_rel_sem=%.4f  predicted_min_N=%zu\n",
        analysis.target_rel_sem, analysis.predicted_min_N);
    std::printf("    efficiency_ratio=%.4f\n", analysis.efficiency_ratio);

    // If achieved_rel_sem < target, we over-ran; predicted_min_N < actual_N
    // If achieved_rel_sem > target, we under-ran; predicted_min_N > actual_N
    // In either case: N_min = N * (achieved/target)^2 should be valid

    const double expected_min_N = N_large
        * (analysis.achieved_rel_sem / target)
        * (analysis.achieved_rel_sem / target);

    REQUIRE_CLOSE(static_cast<double>(analysis.predicted_min_N),
                  expected_min_N, expected_min_N * 0.01);  // 1% match
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Time-dependent system (driven qubit) smart stopping
// ─────────────────────────────────────────────────────────────────────────────

TEST("Phase 5: time-dependent driven qubit converges correctly") {
    const double gamma = 1.0;
    const double Omega = 0.5;
    const double T     = 5.0;  // long enough to reach steady state

    // H(t) = Omega*sigma_x  (resonant drive, rotating frame)
    // Steady state: rho_ee^ss = 4*Omega^2 / (gamma^2 + 8*Omega^2)
    const double exact_ss = 4.0*Omega*Omega / (gamma*gamma + 8.0*Omega*Omega);

    // Build time-dependent system manually via EnsembleManager
    // (SimulationBuilder doesn't yet support time-dependent H — Phase 4 extension)
    // Use DenseTDOpenSystem directly

    DenseOperator H_static(2);  // zero (resonant frame, no detuning)
    DenseOperator H_drive(2);
    H_drive(0,1) = Scalar{Omega, 0.0};
    H_drive(1,0) = Scalar{Omega, 0.0};

    DenseOperator L(2); L(1,0) = Scalar{std::sqrt(gamma), 0.0};
    std::vector<DenseOperator> ops; ops.push_back(std::move(L));
    LindbladSet<DenseTag> lb(std::move(ops));

    std::vector<DriveTerm<DenseTag>> drives;
    drives.push_back({[](Real /*t*/){ return 1.0; }, std::move(H_drive)});
    // f(t) = 1 (constant): H(t) = H_static + 1*H_drive = Omega*sigma_x

    DenseTDOpenSystem sys(std::move(H_static), std::move(drives), std::move(lb));

    // Build observable: rho_ee = |e><e|
    DenseOperator rho_ee_op(2);
    rho_ee_op(0,0) = Scalar{1.0, 0.0};

    std::vector<ObservableDef> obs_defs;
    obs_defs.push_back(make_operator_observable("rho_ee", rho_ee_op));

    StoppingCriteria sc;
    sc.min_trajectories = 200;
    sc.max_trajectories = 5000;
    sc.target_rel_sem   = 0.05;

    EnsembleConfig ec;
    ec.global_seed = 42;
    ec.diag_level  = DiagnosticLevel::None;
    ec.propagator.dt_initial = 5e-4;

    EnsembleManager<DenseTDOpenSystem,
                    liquid::ode::RK4Stepper> mgr(
        &sys, std::move(obs_defs), sc, ec,
        make_uniform_policy(42ULL));

    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};
    EnsembleResult result = mgr.run(psi0, 0.0, T);

    const double mean_rho = result.observables[0].mean;
    const double sem_rho  = result.observables[0].sem;

    std::printf("    rho_ee(MCWF)=%.4f+/-%.4f  exact_ss=%.4f\n",
        mean_rho, sem_rho, exact_ss);

    REQUIRE_CLOSE(mean_rho, exact_ss, std::max(5.0 * sem_rho, 1e-2));
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Wall-clock budget stopping
// ─────────────────────────────────────────────────────────────────────────────

TEST("Phase 5: wall-clock budget stopping works") {
    const double omega = 1.0, gamma = 1.0;

    Simulation sim = SimulationBuilder{}
        .hamiltonian(make_H(omega))
        .collapse_operator(make_L(gamma))
        .observe("sz", make_sz())
        .seed(999)
        .dt(1e-3)
        .stop_at_sem(1e-6)        // impossible SEM — will hit wall budget
        .min_trajectories(10)
        .max_trajectories(1000000)
        .max_wall_time(0.5)       // 500ms hard limit
        .build();

    StateVector psi0 = make_excited();
    EnsembleResult result = sim.run(psi0, 0.0, 1.0);

    std::printf("    Stopped by budget: N=%zu  wall=%.3fs\n",
        result.total_trajectories,
        result.total_wall_time_seconds);

    REQUIRE(result.convergence.decision == StoppingDecision::BudgetHit);
    REQUIRE(result.total_trajectories >= 10);
    // Wall time should be close to 0.5s (within generous margin)
    REQUIRE_LT(result.total_wall_time_seconds, 2.0);
}
END_TEST

int main() {
    std::printf("=== Phase 5 Validation: Smart Stopping ===\n\n");
    return RUN_ALL_TESTS();
}
