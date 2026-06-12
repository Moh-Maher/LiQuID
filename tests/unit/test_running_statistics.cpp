// tests/unit/test_running_statistics.cpp

#include "liquid/ensemble/running_statistics.hpp"
#include "../test_framework.hpp"
#include <cmath>
#include <numeric>
#include <vector>

using namespace liquid;
using namespace liquid::ensemble;

// ── Basic correctness ─────────────────────────────────────────────────────────

TEST("RunningStatistics: mean of {1,2,3} = 2") {
    RunningStatistics s(1);
    s.update(1.0);
    s.update(2.0);
    s.update(3.0);
    REQUIRE_CLOSE(s.mean(0), 2.0, 1e-13);
}
END_TEST

TEST("RunningStatistics: variance of {1,2,3} = 1 (unbiased)") {
    RunningStatistics s(1);
    s.update(1.0);
    s.update(2.0);
    s.update(3.0);
    // unbiased: sum((x-mean)^2) / (n-1) = (1+0+1)/2 = 1
    REQUIRE_CLOSE(s.variance(0), 1.0, 1e-13);
}
END_TEST

TEST("RunningStatistics: SEM of {1,2,3} = 1/sqrt(3)") {
    RunningStatistics s(1);
    s.update(1.0);
    s.update(2.0);
    s.update(3.0);
    REQUIRE_CLOSE(s.sem(0), 1.0 / std::sqrt(3.0), 1e-12);
}
END_TEST

TEST("RunningStatistics: count tracks correctly") {
    RunningStatistics s(2);
    REQUIRE(s.count() == 0);
    s.update(std::vector<Real>{1.0, 2.0});
    REQUIRE(s.count() == 1);
    s.update(std::vector<Real>{3.0, 4.0});
    REQUIRE(s.count() == 2);
}
END_TEST

TEST("RunningStatistics: single sample has zero variance") {
    RunningStatistics s(1);
    s.update(5.0);
    REQUIRE_CLOSE(s.variance(0), 0.0, 1e-14);
    REQUIRE(s.count() == 1);
}
END_TEST

TEST("RunningStatistics: vector update, two observables") {
    RunningStatistics s(2);
    s.update(std::vector<Real>{1.0, 10.0});
    s.update(std::vector<Real>{3.0, 20.0});
    s.update(std::vector<Real>{5.0, 30.0});
    REQUIRE_CLOSE(s.mean(0), 3.0, 1e-13);
    REQUIRE_CLOSE(s.mean(1), 20.0, 1e-13);
}
END_TEST

// ── Numerical stability (large values) ───────────────────────────────────────

TEST("RunningStatistics: numerically stable for large offset") {
    // Naive mean accumulation: (sum_x / n) loses precision for large offset.
    // Welford is stable because it tracks deviations from running mean.
    RunningStatistics s(1);
    const Real offset = 1e10;
    // Add {1e10+1, 1e10+2, 1e10+3}
    s.update(offset + 1.0);
    s.update(offset + 2.0);
    s.update(offset + 3.0);
    REQUIRE_CLOSE(s.mean(0), offset + 2.0, 1e-4);  // large absolute, small relative
    REQUIRE_CLOSE(s.variance(0), 1.0, 1e-6);        // variance should be unaffected
}
END_TEST

// ── Parallel merge correctness ────────────────────────────────────────────────

TEST("RunningStatistics: merge of two halves equals full") {
    const int N = 100;
    std::vector<Real> data(N);
    for (int i = 0; i < N; ++i) data[i] = static_cast<Real>(i + 1);

    // Full computation
    RunningStatistics full(1);
    for (auto v : data) full.update(v);

    // Split into two halves and merge
    RunningStatistics a(1), b(1);
    for (int i = 0; i < N / 2; ++i) a.update(data[i]);
    for (int i = N / 2; i < N; ++i) b.update(data[i]);
    a.merge(b);

    REQUIRE_CLOSE(a.mean(0), full.mean(0), 1e-10);
    REQUIRE_CLOSE(a.variance(0), full.variance(0), 1e-8);
    REQUIRE(a.count() == full.count());
}
END_TEST

TEST("RunningStatistics: merge with empty is identity") {
    RunningStatistics s(1);
    s.update(1.0); s.update(2.0); s.update(3.0);
    const Real m = s.mean(0);
    const Real v = s.variance(0);

    RunningStatistics empty(1);
    s.merge(empty);

    REQUIRE_CLOSE(s.mean(0), m, 1e-14);
    REQUIRE_CLOSE(s.variance(0), v, 1e-14);
    REQUIRE(s.count() == 3);
}
END_TEST

TEST("RunningStatistics: merge commutes (order invariant)") {
    // a.merge(b) and b.merge(a) should give the same mean/variance
    RunningStatistics a(1), b(1);
    a.update(1.0); a.update(2.0);
    b.update(3.0); b.update(4.0); b.update(5.0);

    RunningStatistics a_copy = a, b_copy = b;
    a.merge(b);
    b_copy.merge(a_copy);

    REQUIRE_CLOSE(a.mean(0), b_copy.mean(0), 1e-13);
    REQUIRE_CLOSE(a.variance(0), b_copy.variance(0), 1e-12);
}
END_TEST

TEST("RunningStatistics: merge of N singletons matches direct update") {
    const int N = 50;
    RunningStatistics direct(1);
    std::vector<RunningStatistics> singles;
    for (int i = 0; i < N; ++i) {
        singles.emplace_back(1);
        singles.back().update(static_cast<Real>(i));
        direct.update(static_cast<Real>(i));
    }

    // Merge all singletons into first
    for (int i = 1; i < N; ++i) singles[0].merge(singles[i]);

    REQUIRE_CLOSE(singles[0].mean(0),     direct.mean(0),     1e-10);
    REQUIRE_CLOSE(singles[0].variance(0), direct.variance(0), 1e-8);
}
END_TEST

// ── Relative SEM ──────────────────────────────────────────────────────────────

TEST("RunningStatistics: relative_sem decreases as 1/sqrt(N)") {
    RunningStatistics s(1);
    // Add N uniform draws in [0,1] — variance = 1/12
    // SEM ∝ 1/sqrt(N), so relative_sem ∝ 1/sqrt(N)
    const int N_small = 100, N_large = 400;

    RunningStatistics small_s(1), large_s(1);
    for (int i = 0; i < N_small; ++i) small_s.update(0.5 + 0.01 * (i % 10));
    for (int i = 0; i < N_large; ++i) large_s.update(0.5 + 0.01 * (i % 10));

    // SEM(large) ≈ SEM(small) / 2  (N_large = 4 * N_small → sqrt(4)=2)
    const Real ratio = large_s.sem(0) / small_s.sem(0);
    REQUIRE_CLOSE(ratio, 0.5, 0.05);  // within 5%
}
END_TEST

// ── Reset ─────────────────────────────────────────────────────────────────────

TEST("RunningStatistics: reset clears all state") {
    RunningStatistics s(1);
    for (int i = 0; i < 10; ++i) s.update(static_cast<Real>(i));
    REQUIRE(s.count() == 10);

    s.reset();
    REQUIRE(s.count() == 0);
    REQUIRE_CLOSE(s.mean(0), 0.0, 1e-14);
}
END_TEST

int main() {
    std::printf("=== RunningStatistics Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
