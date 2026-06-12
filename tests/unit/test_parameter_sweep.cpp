// tests/unit/test_parameter_sweep.cpp

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace liquid;
using namespace liquid::ensemble;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Simulation make_decay_sim(double gamma, int N = 200) {
    DenseOperator H(2);
    H(0,0) = Scalar{0.5, 0}; H(1,1) = Scalar{-0.5, 0};
    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0};
    DenseOperator sz(2);
    sz(0,0) = Scalar{1,0}; sz(1,1) = Scalar{-1,0};

    return SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .seed(42)
        .dt(1e-3)
        .min_trajectories(N)
        .max_trajectories(N)
        .build();
}

// ── Builder tests ─────────────────────────────────────────────────────────────

TEST("ParameterSweep: builder throws with no parameter") {
    bool threw = false;
    try {
        ParameterSweepBuilder{}
            .simulation_factory([](double){ return make_decay_sim(1.0); })
            .initial_state([]{StateVector p(2);p[0]=Scalar{1,0};return p;}())
            .time_interval(0.0, 1.0)
            .build();
    } catch (const std::runtime_error&) { threw = true; }
    REQUIRE(threw);
}
END_TEST

TEST("ParameterSweep: builder throws with no factory") {
    bool threw = false;
    try {
        ParameterSweepBuilder{}
            .parameter("gamma", {1.0, 2.0})
            .initial_state([]{StateVector p(2);p[0]=Scalar{1,0};return p;}())
            .time_interval(0.0, 1.0)
            .build();
    } catch (const std::runtime_error&) { threw = true; }
    REQUIRE(threw);
}
END_TEST

TEST("ParameterSweep: builder throws with t_final <= t0") {
    bool threw = false;
    try {
        ParameterSweepBuilder{}
            .parameter("gamma", {1.0})
            .simulation_factory([](double){ return make_decay_sim(1.0); })
            .initial_state([]{StateVector p(2);p[0]=Scalar{1,0};return p;}())
            .time_interval(1.0, 0.5)  // bad: t_final < t0
            .build();
    } catch (const std::runtime_error&) { threw = true; }
    REQUIRE(threw);
}
END_TEST

// ── Parameter range tests ─────────────────────────────────────────────────────

TEST("ParameterSweep: parameter_range produces correct endpoints") {
    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter_range("gamma", 0.5, 2.5, 5)
        .simulation_factory([](double g){ return make_decay_sim(g); })
        .initial_state(psi0)
        .time_interval(0.0, 0.5)
        .build();

    SweepResult result = sweep.run();

    REQUIRE(result.points.size() == 5);
    REQUIRE_CLOSE(result.points.front().param_value, 0.5, 1e-12);
    REQUIRE_CLOSE(result.points.back().param_value,  2.5, 1e-12);
}
END_TEST

TEST("ParameterSweep: parameter_logrange produces correct endpoints") {
    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter_logrange("gamma", 0.1, 10.0, 4)
        .simulation_factory([](double g){ return make_decay_sim(g, 50); })
        .initial_state(psi0)
        .time_interval(0.0, 0.2)
        .build();

    SweepResult result = sweep.run();

    REQUIRE(result.points.size() == 4);
    REQUIRE_CLOSE(result.points.front().param_value, 0.1,  1e-12);
    REQUIRE_CLOSE(result.points.back().param_value,  10.0, 1e-10);
}
END_TEST

// ── Run correctness ───────────────────────────────────────────────────────────

TEST("ParameterSweep: results match standalone simulations") {
    const double gamma1 = 0.5, gamma2 = 2.0;
    const double T      = 0.5;
    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter("gamma", {gamma1, gamma2})
        .simulation_factory([](double g){ return make_decay_sim(g, 300); })
        .initial_state(psi0)
        .time_interval(0.0, T)
        .build();

    SweepResult result = sweep.run();

    REQUIRE(result.points.size() == 2);

    // Each result should match analytic <sz(T)> = 2*exp(-gamma*T) - 1
    for (std::size_t k = 0; k < 2; ++k) {
        const double g      = result.points[k].param_value;
        const double exact  = 2.0 * std::exp(-g * T) - 1.0;
        const double mean   = result.points[k].result.observables[0].mean;
        const double sem    = result.points[k].result.observables[0].sem;
        const double tol    = std::max(4.0 * sem, 0.02);

        std::printf("    gamma=%.2f: <sz>=%.4f exact=%.4f tol=%.4f\n",
            g, mean, exact, tol);

        REQUIRE_CLOSE(mean, exact, tol);
    }
}
END_TEST

TEST("ParameterSweep: parameter name propagates to result") {
    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter("decay_rate", {1.0, 2.0})
        .simulation_factory([](double g){ return make_decay_sim(g, 50); })
        .initial_state(psi0)
        .time_interval(0.0, 0.2)
        .build();

    SweepResult result = sweep.run();
    REQUIRE(result.param_name == "decay_rate");
    for (const auto& pt : result.points)
        REQUIRE(pt.param_name == "decay_rate");
}
END_TEST

// ── CSV/JSON output ───────────────────────────────────────────────────────────

TEST("SweepResult: save_csv produces valid file") {
    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter("gamma", {0.5, 1.0, 2.0})
        .simulation_factory([](double g){ return make_decay_sim(g, 50); })
        .initial_state(psi0)
        .time_interval(0.0, 0.2)
        .build();

    SweepResult result = sweep.run();

    const char* csv_path = "/tmp/test_sweep_output.csv";
    result.save_csv(csv_path);

    // Read file and verify it has content
    FILE* f = std::fopen(csv_path, "r");
    REQUIRE(f != nullptr);

    char line[512];
    int line_count = 0;
    while (std::fgets(line, sizeof(line), f))
        ++line_count;
    std::fclose(f);

    // Header + 3 data rows = 4 lines
    REQUIRE(line_count == 4);
}
END_TEST

TEST("SweepResult: save_json produces valid file") {
    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};

    auto sweep = ParameterSweepBuilder{}
        .parameter("gamma", {1.0, 2.0})
        .simulation_factory([](double g){ return make_decay_sim(g, 50); })
        .initial_state(psi0)
        .time_interval(0.0, 0.2)
        .build();

    SweepResult result = sweep.run();

    const char* json_path = "/tmp/test_sweep_output.json";
    result.save_json(json_path);

    FILE* f = std::fopen(json_path, "r");
    REQUIRE(f != nullptr);

    char buf[4096];
    std::size_t nread = std::fread(buf, 1, sizeof(buf)-1, f);
    buf[nread] = '\0';
    std::fclose(f);

    // Basic JSON structure checks
    REQUIRE(std::strstr(buf, "\"parameter\"") != nullptr);
    REQUIRE(std::strstr(buf, "\"points\"")    != nullptr);
    REQUIRE(std::strstr(buf, "\"gamma\"")     != nullptr);
    REQUIRE(std::strstr(buf, "sigma_z")       != nullptr);
}
END_TEST

// ── Simulation::make factories ────────────────────────────────────────────────

TEST("Simulation::make: generic factory with dense system") {
    DenseOperator H(2);
    H(0,0) = Scalar{0.5, 0}; H(1,1) = Scalar{-0.5, 0};
    DenseOperator L(2); L(1,0) = Scalar{1.0, 0};
    DenseOperator sz(2); sz(0,0) = Scalar{1,0}; sz(1,1) = Scalar{-1,0};

    std::vector<DenseOperator> ops; ops.push_back(std::move(L));
    LindbladSet<DenseTag> lb(std::move(ops));
    DenseOpenSystem sys(std::move(H), std::move(lb));

    std::vector<ObservableDef> obs;
    obs.push_back(make_operator_observable("sz", sz));

    StoppingCriteria sc;
    sc.min_trajectories = 100;
    sc.max_trajectories = 100;

    EnsembleConfig ec;
    ec.global_seed = 42;
    ec.diag_level  = DiagnosticLevel::None;
    ec.propagator.dt_initial = 1e-3;

    auto sim = Simulation::make<DenseOpenSystem, liquid::ode::RK4Stepper>(
        std::move(sys), std::move(obs), sc, ec);

    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};
    auto result = sim.run(psi0, 0.0, 1.0);

    REQUIRE(result.total_trajectories == 100);
    REQUIRE(result.observables[0].mean >= -1.0);
    REQUIRE(result.observables[0].mean <=  1.0);
}
END_TEST

TEST("Simulation::make_sparse_dopri45: works with sparse system") {
    // 2-level system as sparse
    std::vector<Triplet> H_trips = {
        {0, 0, Scalar{0.5, 0}}, {1, 1, Scalar{-0.5, 0}}
    };
    std::vector<Triplet> L_trips = {{1, 0, Scalar{1.0, 0}}};

    SparseOperator H(2, std::move(H_trips));
    std::vector<SparseOperator> ops;
    ops.emplace_back(2, std::move(L_trips));
    SparseOpenSystem sys(std::move(H), LindbladSet<SparseTag>(std::move(ops)));

    SparseOperator sz_sparse(2, std::vector<Triplet>{
        {0, 0, Scalar{1, 0}}, {1, 1, Scalar{-1, 0}}});
    std::vector<ObservableDef> obs;
    obs.push_back({"sz",
        [op = std::move(sz_sparse)](const StateVector& psi) -> Real {
            const Real ns = psi.norm_sq();
            return ns > 1e-30
                ? sparse_expectation(op, psi).real() / ns
                : 0.0;
        }});

    StoppingCriteria sc;
    sc.min_trajectories = 100;
    sc.max_trajectories = 100;

    EnsembleConfig ec;
    ec.global_seed = 42;
    ec.diag_level  = DiagnosticLevel::None;
    ec.propagator.dt_initial = 1e-2;

    auto sim = Simulation::make_sparse_dopri45(
        std::move(sys), std::move(obs), sc, ec);

    StateVector psi0(2); psi0[0] = Scalar{1.0, 0.0};
    auto result = sim.run(psi0, 0.0, 1.0);

    const double exact = 2.0 * std::exp(-1.0 * 1.0) - 1.0;
    const double mean  = result.observables[0].mean;
    const double sem   = result.observables[0].sem;

    REQUIRE_CLOSE(mean, exact, std::max(4.0 * sem, 0.02));
}
END_TEST

int main() {
    std::printf("=== Parameter Sweep Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
