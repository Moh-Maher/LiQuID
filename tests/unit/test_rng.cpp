// tests/unit/test_rng.cpp
// Unit tests for liquid::RNGState (xoshiro256**)

#include "liquid/core/rng.hpp"
#include "../test_framework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>

using namespace liquid;

// ── Reproducibility ───────────────────────────────────────────────────────────

TEST("RNG: same seed produces same sequence") {
    RNGState r1, r2;
    r1.seed(42, 0);
    r2.seed(42, 0);
    for (int i = 0; i < 1000; ++i) {
        REQUIRE(r1.next_uint64() == r2.next_uint64());
    }
}
END_TEST

// ── Independence across trajectories ─────────────────────────────────────────

TEST("RNG: different traj_ids produce different sequences") {
    RNGState r0, r1;
    r0.seed(42, 0);
    r1.seed(42, 1);
    // The first output should differ (with astronomically high probability)
    REQUIRE(r0.next_uint64() != r1.next_uint64());
}
END_TEST

TEST("RNG: different global_seeds produce different sequences") {
    RNGState r0, r1;
    r0.seed(42,  0);
    r1.seed(123, 0);
    REQUIRE(r0.next_uint64() != r1.next_uint64());
}
END_TEST

// ── draw_uniform: range check ─────────────────────────────────────────────────

TEST("RNG: draw_uniform returns values in (0, 1]") {
    RNGState r;
    r.seed(7, 3);
    bool saw_positive = false;
    for (int i = 0; i < 100000; ++i) {
        const double u = r.draw_uniform();
        REQUIRE(u > 0.0);
        REQUIRE(u <= 1.0);
        if (u > 0.0) saw_positive = true;
    }
    REQUIRE(saw_positive);
}
END_TEST

// ── draw_uniform: uniformity (chi-squared bucket test) ───────────────────────

TEST("RNG: draw_uniform is approximately uniform") {
    RNGState r;
    r.seed(99, 17);

    const int N_bins   = 10;
    const int N_draws  = 100000;
    std::array<int, N_bins> counts{};

    for (int i = 0; i < N_draws; ++i) {
        const double u = r.draw_uniform();
        const int bin = std::min(static_cast<int>(u * N_bins), N_bins - 1);
        ++counts[bin];
    }

    // Each bin should have approximately N_draws / N_bins counts.
    // Chi-squared test: sum (observed - expected)^2 / expected < threshold.
    // For 10 bins, df=9, p=0.001 threshold is 27.9. We use 30.0 for safety.
    const double expected = static_cast<double>(N_draws) / N_bins;
    double chi2 = 0.0;
    for (int b = 0; b < N_bins; ++b) {
        const double diff = counts[b] - expected;
        chi2 += (diff * diff) / expected;
    }
    REQUIRE_LT(chi2, 30.0);
}
END_TEST

// ── draw_int: range check ─────────────────────────────────────────────────────

TEST("RNG: draw_int returns values in [0, n-1]") {
    RNGState r;
    r.seed(1, 1);
    for (std::size_t n : {1UL, 2UL, 3UL, 5UL, 10UL, 100UL}) {
        for (int i = 0; i < 1000; ++i) {
            const std::size_t v = r.draw_int(n);
            REQUIRE(v < n);
        }
    }
}
END_TEST

TEST("RNG: draw_int(1) always returns 0") {
    RNGState r;
    r.seed(5, 5);
    for (int i = 0; i < 1000; ++i) {
        REQUIRE(r.draw_int(1) == 0);
    }
}
END_TEST

// ── Serialization round-trip ──────────────────────────────────────────────────

TEST("RNG: serialize/deserialize preserves state") {
    RNGState r;
    r.seed(314159, 2718);

    // Advance a bit
    for (int i = 0; i < 50; ++i) r.next_uint64();

    // Serialize
    std::ostringstream oss;
    r.serialize(oss);

    // Deserialize
    std::istringstream iss(oss.str());
    RNGState r2 = RNGState::deserialize(iss);

    // Both should produce identical future sequences
    for (int i = 0; i < 100; ++i) {
        REQUIRE(r.next_uint64() == r2.next_uint64());
    }
}
END_TEST

// ── Main ──────────────────────────────────────────────────────────────────────

#include <sstream>

int main() {
    std::printf("=== RNG Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
