// tests/unit/test_trajectory_state.cpp
// Unit tests for TrajectoryState and make_trajectory_state

#include "liquid/trajectory/trajectory_state.hpp"
#include "../test_framework.hpp"

#include <cmath>

using namespace liquid;
using namespace liquid::trajectory;

TEST("TrajectoryState: initial state is normalized") {
    StateVector psi(2);
    psi[0] = Scalar{3.0, 0.0};
    psi[1] = Scalar{4.0, 0.0};  // unnormalized: norm = 5

    auto ts = make_trajectory_state(psi, 0.0, 10.0, 0, 42);

    // INV-1: must be normalized at t=t0
    REQUIRE_CLOSE(ts.core.psi.norm_sq(), 1.0, 1e-14);
}
END_TEST

TEST("TrajectoryState: initial time is set correctly") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    auto ts = make_trajectory_state(psi, 1.5, 5.0, 3, 99);

    REQUIRE_CLOSE(ts.core.t, 1.5, 1e-14);
    REQUIRE_CLOSE(ts.core.t_final, 5.0, 1e-14);
}
END_TEST

TEST("TrajectoryState: traj_id is set and immutable") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    const TrajId id = 42;
    auto ts = make_trajectory_state(psi, 0.0, 1.0, id, 0);

    REQUIRE(ts.core.traj_id == id);
}
END_TEST

TEST("TrajectoryState: initial status is Initialized") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    auto ts = make_trajectory_state(psi, 0.0, 1.0, 0, 0);
    REQUIRE(ts.core.status == TrajectoryStatus::Initialized);
}
END_TEST

TEST("TrajectoryState: initial jump threshold r is in (0, 1]") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    // Test multiple seeds to check the range
    for (Seed s = 0; s < 20; ++s) {
        auto ts = make_trajectory_state(psi, 0.0, 1.0, s, s * 7);
        REQUIRE(ts.core.r > 0.0);
        REQUIRE(ts.core.r <= 1.0);
    }
}
END_TEST

TEST("TrajectoryState: different seeds give different r values") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    auto ts0 = make_trajectory_state(psi, 0.0, 1.0, 0, 42);
    auto ts1 = make_trajectory_state(psi, 0.0, 1.0, 1, 42);

    // With overwhelming probability, different traj_ids yield different r
    // (this could theoretically fail but probability is < 2^{-53})
    REQUIRE(ts0.core.r != ts1.core.r);
}
END_TEST

TEST("TrajectoryState: diagnostics present when level != None") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    auto ts_summary = make_trajectory_state(psi, 0.0, 1.0, 0, 0,
                                             DiagnosticLevel::Summary);
    REQUIRE(ts_summary.has_diagnostics());

    auto ts_none = make_trajectory_state(psi, 0.0, 1.0, 0, 0,
                                          DiagnosticLevel::None);
    REQUIRE(!ts_none.has_diagnostics());
}
END_TEST

TEST("TrajectoryState: diagnostics summary counters start at zero") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    auto ts = make_trajectory_state(psi, 0.0, 1.0, 0, 0,
                                     DiagnosticLevel::Summary);
    REQUIRE(ts.diag().total_steps == 0);
    REQUIRE(ts.diag().total_jumps == 0);
    REQUIRE(ts.diag().rejected_steps == 0);
    REQUIRE(ts.diag().jumps.empty());
}
END_TEST

TEST("TrajectoryState: record_jump increments total_jumps") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    auto ts = make_trajectory_state(psi, 0.0, 10.0, 0, 0,
                                     DiagnosticLevel::Summary);
    ts.diag().record_jump(1.0, 0, 0.3);
    ts.diag().record_jump(3.0, 0, 0.5);

    REQUIRE(ts.diag().total_jumps == 2);
    REQUIRE(ts.diag().jumps.size() == 2);
    REQUIRE_CLOSE(ts.diag().jumps[0].t, 1.0, 1e-14);
    REQUIRE_CLOSE(ts.diag().jumps[1].t, 3.0, 1e-14);
}
END_TEST

TEST("TrajectoryState: record_step updates running mean dt") {
    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};

    auto ts = make_trajectory_state(psi, 0.0, 10.0, 0, 0,
                                     DiagnosticLevel::Summary);
    ts.diag().record_step(0.0, 0.01, 0.0, 0);
    ts.diag().record_step(0.01, 0.02, 0.0, 0);
    ts.diag().record_step(0.03, 0.03, 0.0, 0);

    REQUIRE(ts.diag().total_steps == 3);
    // Mean of {0.01, 0.02, 0.03} = 0.02
    REQUIRE_CLOSE(ts.diag().mean_dt, 0.02, 1e-12);
    REQUIRE_CLOSE(ts.diag().min_dt,  0.01, 1e-14);
}
END_TEST

int main() {
    std::printf("=== TrajectoryState Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
