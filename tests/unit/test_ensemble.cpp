// tests/unit/test_ensemble.cpp

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>

using namespace liquid;
using namespace liquid::ensemble;
using namespace liquid::trajectory;
using namespace liquid::ode;

// ── Helper: standard two-level decay system ───────────────────────────────────

static DenseOpenSystem make_decay_system(double omega = 1.0, double gamma = 1.0) {
    DenseOperator H(2);
    H(0,0) = Scalar{ omega/2.0, 0.0};
    H(1,1) = Scalar{-omega/2.0, 0.0};
    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0.0};
    std::vector<DenseOperator> ops;
    ops.push_back(std::move(L));
    LindbladSet<DenseTag> lb(std::move(ops));
    return DenseOpenSystem(std::move(H), std::move(lb));
}

static StateVector excited_state() {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};
    return psi;
}

// ── ObservableAccumulator tests ───────────────────────────────────────────────

TEST("ObservableAccumulator: accumulates correctly for excited state") {
    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};

    std::vector<ObservableDef> defs;
    defs.push_back(make_operator_observable("sigma_z", sz));
    ObservableAccumulator accum(std::move(defs));

    StateVector psi_e(2); psi_e[0] = Scalar{1.0, 0.0};
    StateVector psi_g(2); psi_g[1] = Scalar{1.0, 0.0};

    accum.accumulate(psi_e);  // <sz>=+1
    accum.accumulate(psi_g);  // <sz>=-1

    REQUIRE_CLOSE(accum.statistics().mean(0), 0.0, 1e-13);
}
END_TEST

TEST("ObservableAccumulator: merge works correctly") {
    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};

    auto make_accum = [&]() {
        std::vector<ObservableDef> defs;
        defs.push_back(make_operator_observable("sz", DenseOperator(sz)));
        return ObservableAccumulator(std::move(defs));
    };

    ObservableAccumulator a = make_accum();
    ObservableAccumulator b = make_accum();

    StateVector psi_e(2); psi_e[0] = Scalar{1.0, 0.0};
    StateVector psi_g(2); psi_g[1] = Scalar{1.0, 0.0};

    a.accumulate(psi_e); a.accumulate(psi_e);  // mean = +1
    b.accumulate(psi_g); b.accumulate(psi_g);  // mean = -1

    a.merge(b);  // combined mean = 0
    REQUIRE_CLOSE(a.statistics().mean(0), 0.0, 1e-13);
    REQUIRE(a.statistics().count() == 4);
}
END_TEST

// ── ConvergenceMonitor tests ──────────────────────────────────────────────────

TEST("ConvergenceMonitor: Continue when below min_trajectories") {
    StoppingCriteria sc;
    sc.min_trajectories = 100;
    sc.max_trajectories = 10000;
    sc.target_rel_sem   = 0.01;
    sc.enforce_min      = true;
    ConvergenceMonitor monitor(sc);

    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};
    std::vector<ObservableDef> defs;
    defs.push_back(make_operator_observable("sz", sz));
    ObservableAccumulator accum(std::move(defs));

    // Add 50 samples (below min=100)
    StateVector psi_e(2); psi_e[0] = Scalar{1.0, 0.0};
    for (int i = 0; i < 50; ++i) accum.accumulate(psi_e);

    auto report = monitor.evaluate(accum, 0.0);
    REQUIRE(report.decision == StoppingDecision::Continue);
}
END_TEST

TEST("ConvergenceMonitor: BudgetHit at max_trajectories") {
    StoppingCriteria sc;
    sc.min_trajectories = 10;
    sc.max_trajectories = 50;
    sc.target_rel_sem   = 0.001;  // very tight — won't converge at 50

    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};
    std::vector<ObservableDef> defs;
    defs.push_back(make_operator_observable("sz", sz));
    ObservableAccumulator accum(std::move(defs));

    // Alternate excited/ground to get finite variance
    StateVector pe(2); pe[0]=Scalar{1.0,0}; 
    StateVector pg(2); pg[1]=Scalar{1.0,0};
    for (int i = 0; i < 50; ++i)
        accum.accumulate(i % 2 == 0 ? pe : pg);

    ConvergenceMonitor monitor(sc);
    auto report = monitor.evaluate(accum, 0.0);
    REQUIRE(report.decision == StoppingDecision::BudgetHit);
}
END_TEST

TEST("ConvergenceMonitor: Forced on request_stop") {
    StoppingCriteria sc;
    sc.min_trajectories = 1;
    ConvergenceMonitor monitor(sc);

    DenseOperator sz(2);
    sz(0,0) = Scalar{1.0, 0.0};
    std::vector<ObservableDef> defs;
    defs.push_back(make_operator_observable("sz", sz));
    ObservableAccumulator accum(std::move(defs));

    monitor.request_stop();
    auto report = monitor.evaluate(accum, 0.0);
    REQUIRE(report.decision == StoppingDecision::Forced);
}
END_TEST

// ── EnsembleManager smoke tests ───────────────────────────────────────────────

TEST("EnsembleManager: runs exactly max_trajectories when tight budget") {
    auto sys = make_decay_system(1.0, 1.0);

    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};
    std::vector<ObservableDef> obs;
    obs.push_back(make_operator_observable("sz", sz));

    StoppingCriteria sc;
    sc.min_trajectories = 50;
    sc.max_trajectories = 50;
    sc.target_rel_sem   = 1e-6;  // impossible — will hit max

    EnsembleConfig ec;
    ec.global_seed = 42;
    ec.diag_level  = DiagnosticLevel::None;
    ec.propagator.dt_initial = 1e-3;

    EnsembleManager<DenseOpenSystem, RK4Stepper> mgr(
        &sys, std::move(obs), sc, ec, make_uniform_policy(42ULL));

    auto result = mgr.run(excited_state(), 0.0, 1.0);

    REQUIRE(result.total_trajectories == 50);
    REQUIRE(result.convergence.decision == StoppingDecision::BudgetHit);
}
END_TEST

TEST("EnsembleManager: result mean is finite and in [-1, 1] for sigma_z") {
    auto sys = make_decay_system(1.0, 1.0);

    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};
    std::vector<ObservableDef> obs;
    obs.push_back(make_operator_observable("sz", sz));

    StoppingCriteria sc;
    sc.min_trajectories = 100;
    sc.max_trajectories = 100;

    EnsembleConfig ec;
    ec.global_seed = 99;
    ec.diag_level  = DiagnosticLevel::None;
    ec.propagator.dt_initial = 1e-3;

    EnsembleManager<DenseOpenSystem, RK4Stepper> mgr(
        &sys, std::move(obs), sc, ec, make_uniform_policy(99ULL));

    auto result = mgr.run(excited_state(), 0.0, 2.0);
    const Real mean = result.observables[0].mean;

    REQUIRE(mean >= -1.0 - 1e-10);
    REQUIRE(mean <=  1.0 + 1e-10);
    REQUIRE(result.observables[0].sem > 0.0);
}
END_TEST

// ── SimulationBuilder tests ───────────────────────────────────────────────────

TEST("SimulationBuilder: throws if no Hamiltonian") {
    DenseOperator L(2); L(1,0) = Scalar{1.0, 0.0};
    DenseOperator O(2); O(0,0) = Scalar{1.0, 0.0};

    bool threw = false;
    try {
        auto sim = SimulationBuilder{}
            .collapse_operator(std::move(L))
            .observe("x", std::move(O))
            .max_trajectories(10)
            .build();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}
END_TEST

TEST("SimulationBuilder: throws if no collapse operators") {
    DenseOperator H(2); H(0,0) = Scalar{1.0, 0.0};
    DenseOperator O(2); O(0,0) = Scalar{1.0, 0.0};

    bool threw = false;
    try {
        auto sim = SimulationBuilder{}
            .hamiltonian(std::move(H))
            .observe("x", std::move(O))
            .max_trajectories(10)
            .build();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}
END_TEST

TEST("SimulationBuilder: throws on dimension mismatch") {
    DenseOperator H(2);
    DenseOperator L(3);  // wrong dimension
    L(1,0) = Scalar{1.0, 0.0};
    DenseOperator O(2); O(0,0) = Scalar{1.0, 0.0};

    bool threw = false;
    try {
        auto sim = SimulationBuilder{}
            .hamiltonian(std::move(H))
            .collapse_operator(std::move(L))
            .observe("x", std::move(O))
            .max_trajectories(10)
            .build();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}
END_TEST

TEST("SimulationBuilder: builds and runs successfully") {
    DenseOperator H(2);
    H(0,0) = Scalar{0.5, 0.0};
    H(1,1) = Scalar{-0.5, 0.0};

    DenseOperator L(2);
    L(1,0) = Scalar{1.0, 0.0};

    DenseOperator sz(2);
    sz(0,0) = Scalar{ 1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};

    auto sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .seed(42)
        .dt(1e-3)
        .min_trajectories(50)
        .max_trajectories(50)
        .build();

    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};
    auto result = sim.run(psi0, 0.0, 1.0);

    REQUIRE(result.total_trajectories == 50);
    REQUIRE(result.observables.size() == 1);
    REQUIRE(result.observables[0].name == "sigma_z");
    REQUIRE(result.observables[0].mean >= -1.0);
    REQUIRE(result.observables[0].mean <=  1.0);
}
END_TEST

int main() {
    std::printf("=== Ensemble Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
