// tests/validation/parameter_sweep_validation.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Phase 7 Validation: Parameter sweep produces correct physics
//
// Tests:
//   1. Sweep over gamma reproduces analytic <sz(T)> at each point
//   2. CSV output can be parsed back and values match
//   3. Sweep over Rabi frequency reproduces driven steady-state formula
//   4. make_sparse_dopri45 factory produces correct results
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

using namespace liquid;
using namespace liquid::ensemble;

// ── Helpers ───────────────────────────────────────────────────────────────────

static DenseOperator make_H_decay(double omega = 1.0) {
    DenseOperator H(2);
    H(0,0)=Scalar{omega/2,0}; H(1,1)=Scalar{-omega/2,0};
    return H;
}
static DenseOperator make_L_decay(double gamma) {
    DenseOperator L(2); L(1,0)=Scalar{std::sqrt(gamma),0}; return L;
}
static DenseOperator make_sz() {
    DenseOperator sz(2); sz(0,0)=Scalar{1,0}; sz(1,1)=Scalar{-1,0};
    return sz;
}
static StateVector make_excited() {
    StateVector psi(2); psi[0]=Scalar{1.0,0}; return psi;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Sweep over decay rate reproduces <sigma_z(T)>
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 7: gamma sweep matches analytic solution") {
    const double T    = 1.0;
    const int    N    = 1000;
    const double gammas[] = {0.5, 1.0, 2.0};

    StateVector psi0 = make_excited();

    auto sweep = ParameterSweepBuilder{}
        .parameter("gamma", {0.5, 1.0, 2.0})
        .simulation_factory([N](double gamma) {
            return SimulationBuilder{}
                .hamiltonian(make_H_decay())
                .collapse_operator(make_L_decay(gamma))
                .observe("sz", make_sz())
                .seed(42)
                .dt(1e-3)
                .min_trajectories(N)
                .max_trajectories(N)
                .build();
        })
        .initial_state(psi0)
        .time_interval(0.0, T)
        .build();

    SweepResult result = sweep.run();

    REQUIRE(result.points.size() == 3);

    for (int k = 0; k < 3; ++k) {
        const double g     = gammas[k];
        const double exact = 2.0 * std::exp(-g * T) - 1.0;
        const double mean  = result.points[k].result.observables[0].mean;
        const double sem   = result.points[k].result.observables[0].sem;
        const double tol   = std::max(4.0 * sem, 1e-2);

        std::printf("    gamma=%.1f: <sz>=%.4f exact=%.4f sem=%.4f\n",
            g, mean, exact, sem);

        REQUIRE_CLOSE(mean, exact, tol);
    }
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: CSV output round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 7: CSV output is parseable and correct") {
    StateVector psi0 = make_excited();

    auto sweep = ParameterSweepBuilder{}
        .parameter("gamma", {0.5, 1.5})
        .simulation_factory([](double gamma) {
            return SimulationBuilder{}
                .hamiltonian(make_H_decay())
                .collapse_operator(make_L_decay(gamma))
                .observe("sz", make_sz())
                .seed(99)
                .dt(1e-3)
                .min_trajectories(200)
                .max_trajectories(200)
                .build();
        })
        .initial_state(psi0)
        .time_interval(0.0, 0.5)
        .build();

    SweepResult result = sweep.run();

    const char* path = "/tmp/validation_sweep.csv";
    result.save_csv(path);

    // Parse the CSV manually
    FILE* f = std::fopen(path, "r");
    REQUIRE(f != nullptr);

    char header[256];
    REQUIRE(std::fgets(header, sizeof(header), f) != nullptr);

    // Parse row 1: gamma=0.5
    double gamma_val, sz_mean, sz_sem, sz_relsem;
    std::size_t N_traj, N_failed;
    double mean_jumps, total_wall, traj_wall;

    int parsed = std::fscanf(f,
        "%lf,%lf,%lf,%lf,%zu,%zu,%lf,%lf,%lf",
        &gamma_val, &sz_mean, &sz_sem, &sz_relsem,
        &N_traj, &N_failed, &mean_jumps, &total_wall, &traj_wall);
    std::fclose(f);

    REQUIRE(parsed == 9);
    REQUIRE_CLOSE(gamma_val, 0.5, 1e-10);
    REQUIRE(N_traj == 200);

    // The CSV value should match the in-memory result
    REQUIRE_CLOSE(sz_mean,
        result.points[0].result.observables[0].mean, 1e-10);

    std::printf("    CSV round-trip: gamma=%.2f  <sz>=%.4f  N=%zu\n",
        gamma_val, sz_mean, N_traj);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Sweep over Rabi frequency — driven qubit steady state
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 7: Rabi frequency sweep matches steady-state formula") {
    // H = Omega * sigma_x,  L = sqrt(gamma) sigma_-
    // Steady state: rho_ee^ss = 4*Omega^2 / (gamma^2 + 8*Omega^2)
    const double gamma = 1.0;
    const double T_ss  = 15.0;

    const double omegas[] = {0.1, 0.5, 1.0, 2.0};
    const int    N_omega  = 4;

    StateVector psi0 = make_excited();

    auto sweep = ParameterSweepBuilder{}
        .parameter("Omega", {0.1, 0.5, 1.0, 2.0})
        .simulation_factory([gamma](double Omega) {
            DenseOperator H(2);
            H(0,1)=Scalar{Omega,0}; H(1,0)=Scalar{Omega,0};
            DenseOperator L(2); L(1,0)=Scalar{std::sqrt(gamma),0};
            DenseOperator rho_ee(2); rho_ee(0,0)=Scalar{1,0};

            return SimulationBuilder{}
                .hamiltonian(std::move(H))
                .collapse_operator(std::move(L))
                .observe("rho_ee", std::move(rho_ee))
                .seed(42)
                .dt(5e-4)
                .min_trajectories(500)
                .max_trajectories(500)
                .build();
        })
        .initial_state(psi0)
        .time_interval(0.0, T_ss)
        .build();

    SweepResult result = sweep.run();

    REQUIRE(result.points.size() == static_cast<std::size_t>(N_omega));

    for (int k = 0; k < N_omega; ++k) {
        const double Omega = omegas[k];
        // Exact steady-state formula for H = Omega*sigma_x
        const double exact_ss = 4.0*Omega*Omega
                              / (gamma*gamma + 8.0*Omega*Omega);
        const double mean = result.points[k].result.observables[0].mean;
        const double sem  = result.points[k].result.observables[0].sem;
        const double tol  = std::max(5.0 * sem, 5e-3);

        std::printf("    Omega=%.2f: rho_ee=%.4f exact=%.4f sem=%.4f\n",
            Omega, mean, exact_ss, sem);

        REQUIRE_CLOSE(mean, exact_ss, tol);
    }
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Simulation::make_sparse_dopri45 in a parameter sweep
// ─────────────────────────────────────────────────────────────────────────────

TEST("Validation Phase 7: sparse+DOPRI45 factory in parameter sweep") {
    const double T = 1.0;
    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter("gamma", {0.5, 1.0, 2.0})
        .simulation_factory([](double gamma) {
            // Build sparse 2-level decay system
            std::vector<Triplet> H_trips = {
                {0,0,Scalar{0.5,0}}, {1,1,Scalar{-0.5,0}}};
            std::vector<Triplet> L_trips = {
                {1,0,Scalar{std::sqrt(gamma),0}}};
            std::vector<Triplet> sz_trips = {
                {0,0,Scalar{1,0}}, {1,1,Scalar{-1,0}}};

            SparseOperator H(2, std::move(H_trips));
            std::vector<SparseOperator> ops;
            ops.emplace_back(2, std::move(L_trips));
            SparseOpenSystem sys(std::move(H),
                LindbladSet<SparseTag>(std::move(ops)));

            SparseOperator sz_op(2, std::move(sz_trips));
            std::vector<ObservableDef> obs;
            obs.push_back({"sz",
                [op = std::move(sz_op)](const StateVector& psi) -> Real {
                    const Real ns = psi.norm_sq();
                    return ns > 1e-30
                        ? sparse_expectation(op, psi).real() / ns
                        : 0.0;
                }});

            StoppingCriteria sc;
            sc.min_trajectories = 500;
            sc.max_trajectories = 500;
            EnsembleConfig ec;
            ec.global_seed = 42;
            ec.diag_level  = DiagnosticLevel::None;
            ec.propagator.dt_initial = 1e-2;

            return Simulation::make_sparse_dopri45(
                std::move(sys), std::move(obs), sc, ec);
        })
        .initial_state(psi0)
        .time_interval(0.0, T)
        .build();

    SweepResult result = sweep.run();
    REQUIRE(result.points.size() == 3);

    const double gammas[] = {0.5, 1.0, 2.0};
    for (int k = 0; k < 3; ++k) {
        const double exact = 2.0 * std::exp(-gammas[k] * T) - 1.0;
        const double mean  = result.points[k].result.observables[0].mean;
        const double sem   = result.points[k].result.observables[0].sem;
        const double tol   = std::max(4.0 * sem, 0.02);

        std::printf("    gamma=%.1f (sparse+DOPRI45): <sz>=%.4f exact=%.4f\n",
            gammas[k], mean, exact);

        REQUIRE_CLOSE(mean, exact, tol);
    }
}
END_TEST

int main() {
    std::printf("=== Phase 7 Validation: Parameter Sweep ===\n\n");
    return RUN_ALL_TESTS();
}
