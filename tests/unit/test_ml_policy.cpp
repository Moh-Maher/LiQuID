// tests/unit/test_ml_policy.cpp

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>
#include <numeric>

using namespace liquid;
using namespace liquid::ensemble;
using namespace liquid::ensemble::ml;

// ── Feature extraction ────────────────────────────────────────────────────────

TEST("ML: feature extraction produces correct dimensions") {
    EnsembleSummary s{};
    s.trajectories_completed = 100;
    s.trajectories_failed    = 2;
    s.current_rel_sem        = 0.05;
    s.wall_time_elapsed      = 30.0;

    auto f = extract_features(s);
    REQUIRE(f.size() == FEATURE_DIM);
}
END_TEST

TEST("ML: features are in [0, 1] for typical inputs") {
    EnsembleSummary s{};
    s.trajectories_completed = 500;
    s.trajectories_failed    = 5;
    s.current_rel_sem        = 0.03;
    s.wall_time_elapsed      = 45.0;

    auto f = extract_features(s);
    for (std::size_t i = 0; i < f.size(); ++i) {
        REQUIRE(f[i] >= 0.0);
        REQUIRE(f[i] <= 1.0 + 1e-10);
    }
}
END_TEST

TEST("ML: feature 1 is 1.0 when rel_sem >= 1.0") {
    EnsembleSummary s{};
    s.current_rel_sem = 99.0;  // absurdly large
    auto f = extract_features(s);
    REQUIRE_CLOSE(f[1], 1.0, 1e-14);
}
END_TEST

TEST("ML: feature 0 increases monotonically with N") {
    EnsembleSummary s{};
    s.trajectories_completed = 10;
    auto f10 = extract_features(s);
    s.trajectories_completed = 100;
    auto f100 = extract_features(s);
    s.trajectories_completed = 1000;
    auto f1000 = extract_features(s);

    REQUIRE(f10[0] < f100[0]);
    REQUIRE(f100[0] < f1000[0]);
}
END_TEST

// ── MLP ───────────────────────────────────────────────────────────────────────

TEST("MLP: forward output is in (0, 1)") {
    MLP net(FEATURE_DIM, 8, 0.01);
    EnsembleSummary s{};
    s.current_rel_sem = 0.1;
    auto f = extract_features(s);

    const Real out = net.forward(f);
    REQUIRE(out > 0.0);
    REQUIRE(out < 1.0);
}
END_TEST

TEST("MLP: forward is deterministic for same input") {
    MLP net(FEATURE_DIM, 8, 0.01);
    std::vector<Real> x(FEATURE_DIM, 0.5);

    const Real out1 = net.forward(x);
    const Real out2 = net.forward(x);
    REQUIRE_CLOSE(out1, out2, 1e-15);
}
END_TEST

TEST("MLP: backward changes weights (output shifts after training)") {
    MLP net(FEATURE_DIM, 8, 0.1);  // large lr for visible effect
    std::vector<Real> x(FEATURE_DIM, 0.5);

    const Real before = net.forward(x);

    // Train with a strong positive reward 50 times
    for (int i = 0; i < 50; ++i)
        net.backward(x, 5.0);

    const Real after = net.forward(x);

    // After positive reward, output should increase (network learns to output high)
    REQUIRE(after > before);
}
END_TEST

TEST("MLP: negative reward decreases output") {
    MLP net(FEATURE_DIM, 8, 0.1);
    std::vector<Real> x(FEATURE_DIM, 0.5);

    const Real before = net.forward(x);

    for (int i = 0; i < 50; ++i)
        net.backward(x, -5.0);

    const Real after = net.forward(x);
    REQUIRE(after < before);
}
END_TEST

TEST("MLP: two different networks with same init produce same output") {
    MLP net1(FEATURE_DIM, 8, 0.01);
    MLP net2(FEATURE_DIM, 8, 0.01);
    std::vector<Real> x(FEATURE_DIM, 0.3);

    REQUIRE_CLOSE(net1.forward(x), net2.forward(x), 1e-14);
}
END_TEST

// ── MLPolicy ─────────────────────────────────────────────────────────────────

TEST("MLPolicy: produces valid AllocationDecision") {
    MLPolicy policy(42, 8, 0.01);
    EnsembleSummary s{};
    s.trajectories_completed = 50;
    s.current_rel_sem = 0.1;

    auto decision = policy(s);
    REQUIRE(decision.action == AllocAction::SpawnNew);
}
END_TEST

TEST("MLPolicy: sequential calls produce different traj_ids") {
    MLPolicy policy(42, 8, 0.01);
    EnsembleSummary s{};
    s.current_rel_sem = 0.1;

    auto d1 = policy(s);
    s.trajectories_completed = 1;
    auto d2 = policy(s);

    REQUIRE(d2.new_traj_id == d1.new_traj_id + 1);
}
END_TEST

TEST("MLPolicy: call count tracks correctly") {
    MLPolicy policy(42, 8, 0.01);
    EnsembleSummary s{};
    s.current_rel_sem = 0.1;

    REQUIRE(policy.calls() == 0);
    policy(s); REQUIRE(policy.calls() == 1);
    s.trajectories_completed = 1;
    policy(s); REQUIRE(policy.calls() == 2);
}
END_TEST

TEST("MLPolicy: updates increase after second call") {
    MLPolicy policy(42, 8, 0.01);
    EnsembleSummary s{};
    s.trajectories_completed = 0;
    s.current_rel_sem = 0.5;

    policy(s);
    REQUIRE(policy.updates() == 0);  // no update on first call

    s.trajectories_completed = 100;
    s.current_rel_sem = 0.1;
    policy(s);
    REQUIRE(policy.updates() == 1);  // update on second call
}
END_TEST

// ── make_ml_policy factory ────────────────────────────────────────────────────

TEST("make_ml_policy: returns valid AllocatorPolicy") {
    auto policy = make_ml_policy(42, 8, 0.01);

    EnsembleSummary s{};
    s.trajectories_completed = 50;
    s.current_rel_sem = 0.05;

    auto decision = policy(s);
    REQUIRE(decision.action == AllocAction::SpawnNew);
}
END_TEST

TEST("make_ml_policy: state persists across calls") {
    auto policy = make_ml_policy(42, 8, 0.01);

    // First call
    EnsembleSummary s{};
    s.trajectories_completed = 0; s.current_rel_sem = 1.0;
    auto d1 = policy(s);

    // Second call with progress
    s.trajectories_completed = 100; s.current_rel_sem = 0.1;
    auto d2 = policy(s);

    // IDs should be sequential (state persists between calls)
    REQUIRE(d2.new_traj_id == d1.new_traj_id + 1);
}
END_TEST

int main() {
    std::printf("=== ML Policy Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
