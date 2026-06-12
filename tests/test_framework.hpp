#pragma once

// tests/test_framework.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Minimal self-contained test framework.
// No dependencies on Catch2, GTest, or any external library.
//
// Usage:
//   TEST("description") { REQUIRE(expr); REQUIRE_CLOSE(a, b, tol); }
//   int main() { return RUN_ALL_TESTS(); }
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <stdexcept>


#include <cstdlib>
#include <exception>
#include <iostream>

#ifdef _WIN32
    #include <windows.h>
#endif

/* =========================================================
   ANSI COLOR SUPPORT
   ========================================================= */

namespace termcolor {

#ifdef _WIN32

inline void enable() {
    static bool initialized = false;

    if (initialized)
        return;

    initialized = true;

    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);

    if (hOut == INVALID_HANDLE_VALUE)
        return;

    DWORD mode = 0;

    if (!GetConsoleMode(hOut, &mode))
        return;

    SetConsoleMode(
        hOut,
        mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING
    );
}

#else

inline void enable() {}

#endif

constexpr const char* reset  = "\x1B[0m";

constexpr const char* red    = "\x1B[31m";
constexpr const char* green  = "\x1B[32m";
constexpr const char* yellow = "\x1B[33m";
constexpr const char* blue   = "\x1B[34m";
constexpr const char* cyan   = "\x1B[36m";

constexpr const char* bold   = "\x1B[1m";

} // namespace termcolor


namespace liquid::test {

struct TestCase {
    std::string              name;
    std::function<void()>    fn;
};

inline std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> reg;
    return reg;
}
/* =========================================================
   TEST RUNNER
   ========================================================= */

inline int run_all_tests() {

    termcolor::enable();

    int passed = 0;
    int failed = 0;

    const auto& tests = test_registry();

    std::printf(
        "\n%s========================================%s\n",
        termcolor::bold,
        termcolor::reset
    );

    std::printf(
        "%sRunning %zu test(s)%s\n",
        termcolor::cyan,
        tests.size(),
        termcolor::reset
    );

    std::printf(
        "%s========================================%s\n\n",
        termcolor::bold,
        termcolor::reset
    );

    for (const auto& tc : tests) {

        std::printf(
            "%s[ RUN  ]%s %s\n",
            termcolor::blue,
            termcolor::reset,
            tc.name.c_str()
        );

        try {

            tc.fn();

            ++passed;

            std::printf(
                "%s[ PASS ]%s %s\n\n",
                termcolor::green,
                termcolor::reset,
                tc.name.c_str()
            );

        } catch (const std::exception& e) {

            ++failed;

            std::printf(
                "%s[ FAIL ]%s %s\n",
                termcolor::red,
                termcolor::reset,
                tc.name.c_str()
            );

            std::printf(
                "         reason: %s\n\n",
                e.what()
            );

        } catch (...) {

            ++failed;

            std::printf(
                "%s[ FAIL ]%s %s\n",
                termcolor::red,
                termcolor::reset,
                tc.name.c_str()
            );

            std::printf(
                "         reason: unknown exception\n\n"
            );
        }
    }

    const int total = passed + failed;

    const double success_rate =
        (total == 0)
            ? 0.0
            : (100.0 * passed / total);

    std::printf(
        "%s========================================%s\n",
        termcolor::bold,
        termcolor::reset
    );

    std::printf(
        "%sTest Summary%s\n",
        termcolor::bold,
        termcolor::reset
    );

    std::printf(
        "  Total   : %d\n",
        total
    );

    std::printf(
        "  Passed  : %s%d%s\n",
        termcolor::green,
        passed,
        termcolor::reset
    );

    std::printf(
        "  Failed  : %s%d%s\n",
        failed > 0 ? termcolor::red : termcolor::green,
        failed,
        termcolor::reset
    );

    std::printf(
        "  Success : %.1f%%\n",
        success_rate
    );

    std::printf(
        "%s========================================%s\n\n",
        termcolor::bold,
        termcolor::reset
    );

    return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

//=========================================================
struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> fn) {
        test_registry().push_back({name, std::move(fn)});
    }
};

} // namespace liquid::test

// ── Assertion macros ──────────────────────────────────────────────────────────

#define REQUIRE(expr) \
    do { \
        if (!(expr)) { \
            throw std::runtime_error( \
                std::string("REQUIRE failed: " #expr \
                    " at " __FILE__ ":") + std::to_string(__LINE__)); \
        } \
    } while(0)

#define REQUIRE_CLOSE(a, b, tol) \
    do { \
        double _a = static_cast<double>(a); \
        double _b = static_cast<double>(b); \
        double _tol = static_cast<double>(tol); \
        if (std::abs(_a - _b) > _tol) { \
            char _buf[256]; \
            std::snprintf(_buf, sizeof(_buf), \
                "REQUIRE_CLOSE failed: |%g - %g| = %g > %g at " __FILE__ ":%d", \
                _a, _b, std::abs(_a - _b), _tol, __LINE__); \
            throw std::runtime_error(_buf); \
        } \
    } while(0)

#define REQUIRE_LT(a, b) \
    do { \
        double _a = static_cast<double>(a); \
        double _b = static_cast<double>(b); \
        if (!(_a < _b)) { \
            char _buf[256]; \
            std::snprintf(_buf, sizeof(_buf), \
                "REQUIRE_LT failed: %g < %g at " __FILE__ ":%d", _a, _b, __LINE__); \
            throw std::runtime_error(_buf); \
        } \
    } while(0)

#define LIQUID_CONCAT_IMPL(a, b) a##b
#define LIQUID_CONCAT(a, b) LIQUID_CONCAT_IMPL(a, b)

#define TEST(name) \
    static liquid::test::TestRegistrar LIQUID_CONCAT(_liq_test_reg_, __LINE__){ \
        name, []()

#define END_TEST };

#define RUN_ALL_TESTS() liquid::test::run_all_tests()
