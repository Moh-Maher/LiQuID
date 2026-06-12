// tests/validation/ml_policy_validation.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Phase 6 Validation: ML policy integration
//
// Tests:
//   1. ML policy produces physically correct results (same as uniform)
//   2. ML policy network trains online (updates increase, reward signal works)
//   3. ML policy is a drop-in replacement for uniform in SimulationBuilder
//   4. Comparative benchmark: ML vs uniform SEM at fixed trajectory count
//   5. Feature vector stability under edge cases (N=0, infinite SEM)
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>
#include <cstdio>

using namespace liquid;
using namespace liquid::ensemble;
using namespace liquid::ensemble::ml;

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
    sz(0,0) = Scalar{1.0,0}; sz(1,1) = Scalar{-1.0,0}; return sz;
}
static StateVector make_excited() {
    StateVector psi(2); psi[0] = Scalar{1.0,0.0}; return psi;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: ML policy gives correct physics (same <sz> as uniform)
// ─────────────────────────────────────────────────────────────────────────────

TEST("ML Policy: produces physically correct observable") {
    const double omega = 1.0, gamma = 1.0, T = 1.0;
    const int N = 1000;
    const double exact = 2.0 * std::exp(-gamma * T) - 1.0;

    // Run with ML policy
    Simulation sim_ml = SimulationBuilder{}
        .hamiltonian(make_H(omega))
        .collapse_operator(make_L(gamma))
        .observe("sz", make_sz())
        .seed(42)
        .dt(1e-3)
        .min_trajectories(N)
        .max_trajectories(N)
        .allocator_policy(make_ml_policy(42))
        .build();

    EnsembleResult r_ml = sim_ml.run(make_excited(), 0.0, T);

    const double mean = r_ml.observables[0].mean;
    const double sem  = r_ml.observables[0].sem;

    std::printf("    ML policy: <sz>=%.4f+/-%.4f  exact=%.4f\n",
        mean, sem, exact);

    REQUIRE_CLOSE(mean, exact, std::max(4.0 * sem, 1e-2));
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: ML policy trains online — updates accumulate during a run
// ─────────────────────────────────────────────────────────────────────────────

TEST("ML Policy: network updates during simulation run") {
    const double omega = 1.0, gamma = 1.0, T = 1.0;
    const int N = 200;

    // Build ML policy with direct access to the MLPolicy object
    auto ml_ptr = std::make_shared<MLPolicy>(42, 8, 0.01);
    AllocatorPolicy policy = [ml_ptr](const EnsembleSummary& s)
                                     -> AllocationDecision {
        return (*ml_ptr)(s);
    };

    DenseOperator H  = make_H(omega);
    DenseOperator L  = make_L(gamma);
    DenseOperator sz = make_sz();

    StoppingCriteria sc;
    sc.min_trajectories = N;
    sc.max_trajectories = N;

    EnsembleConfig ec;
    ec.global_seed = 42;
    ec.diag_level  = DiagnosticLevel::None;
    ec.propagator.dt_initial = 1e-3;

    std::vector<ObservableDef> obs;
    obs.push_back(make_operator_observable("sz", std::move(sz)));

    std::vector<DenseOperator> ops; ops.push_back(std::move(L));
    LindbladSet<DenseTag> lb(std::move(ops));
    DenseOpenSystem sys(std::move(H), std::move(lb));

    EnsembleManager<DenseOpenSystem, liquid::ode::RK4Stepper> mgr(
        &sys, std::move(obs), sc, ec, std::move(policy));

    mgr.run(make_excited(), 0.0, T);

    // The ML policy should have been called multiple times and trained
    std::printf("    ML calls=%zu  updates=%zu\n",
        ml_ptr->calls(), ml_ptr->updates());

    REQUIRE(ml_ptr->calls() > 0);
    REQUIRE(ml_ptr->updates() > 0);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: ML policy is a drop-in for uniform in SimulationBuilder
// ─────────────────────────────────────────────────────────────────────────────

TEST("ML Policy: SimulationBuilder accepts ML policy as drop-in") {
    const double omega = 1.0, gamma = 1.0;

    // Should compile and run without error
    Simulation sim = SimulationBuilder{}
        .hamiltonian(make_H(omega))
        .collapse_operator(make_L(gamma))
        .observe("sz", make_sz())
        .seed(99)
        .dt(1e-3)
        .min_trajectories(50)
        .max_trajectories(50)
        .allocator_policy(make_ml_policy(99, 4, 0.05))
        .build();

    EnsembleResult r = sim.run(make_excited(), 0.0, 0.5);

    REQUIRE(r.total_trajectories == 50);
    REQUIRE(r.observables[0].mean >= -1.0);
    REQUIRE(r.observables[0].mean <=  1.0);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Comparative benchmark — ML vs uniform at fixed N
//         Both should give statistically equivalent results.
//         The ML policy is not expected to outperform uniform at Phase 6
//         (same trajectory generation, different seeds only).
//         This test documents the baseline for future ML improvements.
// ─────────────────────────────────────────────────────────────────────────────

TEST("ML Policy: comparable accuracy to uniform at fixed N") {
    const double omega = 1.0, gamma = 1.0, T = 1.0;
    const int N = 500;
    const double exact = 2.0 * std::exp(-gamma * T) - 1.0;

    // Uniform policy
    Simulation sim_uni = SimulationBuilder{}
        .hamiltonian(make_H(omega))
        .collapse_operator(make_L(gamma))
        .observe("sz", make_sz())
        .seed(1111)
        .dt(1e-3)
        .min_trajectories(N)
        .max_trajectories(N)
        .build();

    EnsembleResult r_uni = sim_uni.run(make_excited(), 0.0, T);

    // ML policy
    Simulation sim_ml = SimulationBuilder{}
        .hamiltonian(make_H(omega))
        .collapse_operator(make_L(gamma))
        .observe("sz", make_sz())
        .seed(2222)
        .dt(1e-3)
        .min_trajectories(N)
        .max_trajectories(N)
        .allocator_policy(make_ml_policy(2222))
        .build();

    EnsembleResult r_ml = sim_ml.run(make_excited(), 0.0, T);

    std::printf("    Uniform: <sz>=%.4f+/-%.4f\n",
        r_uni.observables[0].mean, r_uni.observables[0].sem);
    std::printf("    ML:      <sz>=%.4f+/-%.4f\n",
        r_ml.observables[0].mean, r_ml.observables[0].sem);
    std::printf("    Exact:         %.4f\n", exact);

    // Both must be accurate
    REQUIRE_CLOSE(r_uni.observables[0].mean, exact,
        std::max(4.0 * r_uni.observables[0].sem, 1e-2));
    REQUIRE_CLOSE(r_ml.observables[0].mean, exact,
        std::max(4.0 * r_ml.observables[0].sem, 1e-2));

    // SEMs should be in the same ballpark (within 50% of each other)
    const double sem_ratio = r_ml.observables[0].sem
                           / r_uni.observables[0].sem;
    std::printf("    SEM ratio (ML/uniform) = %.3f\n", sem_ratio);
    REQUIRE(sem_ratio > 0.5);
    REQUIRE(sem_ratio < 2.0);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Feature extraction handles edge cases
// ─────────────────────────────────────────────────────────────────────────────

TEST("ML Policy: feature extraction handles N=0 and infinite SEM") {
    EnsembleSummary s{};
    s.trajectories_completed = 0;
    s.trajectories_failed    = 0;
    s.current_rel_sem        = std::numeric_limits<Real>::infinity();
    s.wall_time_elapsed      = 0.0;

    // Should not throw, not produce NaN or infinity
    auto f = extract_features(s);
    for (std::size_t i = 0; i < f.size(); ++i) {
        REQUIRE(std::isfinite(f[i]));
        REQUIRE(f[i] >= 0.0);
        REQUIRE(f[i] <= 1.0 + 1e-10);
    }
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: MLP gradient sanity — output moves toward reward signal
// ─────────────────────────────────────────────────────────────────────────────

TEST("ML Policy: MLP learns simple pattern over many steps") {
    // Train the network to output high (≈1) for input x=[1,1,...,1]
    // and low (≈0) for input x=[0,0,...,0].
    // After enough steps, these targets should be distinguishable.

    MLP net(FEATURE_DIM, 16, 0.05);

    std::vector<Real> x_high(FEATURE_DIM, 1.0);
    std::vector<Real> x_low (FEATURE_DIM, 0.0);

    // Train for 500 iterations
    for (int i = 0; i < 500; ++i) {
        net.backward(x_high,  3.0);  // positive reward for high input
        net.backward(x_low,  -3.0);  // negative reward for low input
    }

    const Real out_high = net.forward(x_high);
    const Real out_low  = net.forward(x_low);

    std::printf("    out(all-1)=%.4f  out(all-0)=%.4f\n", out_high, out_low);

    // High input should give higher output than low input
    REQUIRE(out_high > out_low);
    // After 500 steps of strong training, should be reasonably separated
    REQUIRE(out_high - out_low > 0.05);
}
END_TEST

int main() {
    std::printf("=== Phase 6 Validation: ML Policy ===\n\n");
    return RUN_ALL_TESTS();
}
