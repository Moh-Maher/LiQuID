#pragma once

// liquid/core/config.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Plain data configuration structs.
// These are value types: cheap to copy, no invariants to maintain.
// Validation happens at the point of use (propagator construction),
// not in these structs.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include <cstddef>

namespace liquid {

// ── PropagatorConfig ──────────────────────────────────────────────────────────
// Controls the numerical behaviour of MCWFPropagator.
// All fields have physically motivated defaults suitable for
// typical quantum optics problems (γ ~ 1, ω ~ 1 in natural units).

struct PropagatorConfig {
    // ODE stepping
    Real dt_initial      = 1e-3;   // Initial stepsize suggestion
    Real dt_min          = 1e-12;  // Stepsize floor — triggers Failed status
    Real dt_max          = 1e-1;   // Stepsize ceiling

    // Adaptive solver tolerances (Phase 3: RK45)
    // In Phase 1 (fixed-step RK4), these are stored but unused.
    Real atol            = 1e-8;   // Absolute local error tolerance
    Real rtol            = 1e-6;   // Relative local error tolerance

    // Jump detection
    // Acceptable absolute difference |‖ψ‖² - r| at located jump time.
    Real jump_tol        = 1e-6;

    // Maximum bisection iterations for jump location (Phase 3)
    int jump_bisect_max  = 50;

    // Maximum consecutive step rejections before declaring failure (Phase 3)
    int max_step_rejects = 100;
};

// ── StoppingCriteria ──────────────────────────────────────────────────────────
// Controls when the EnsembleManager halts trajectory generation.

struct StoppingCriteria {
    // Statistical convergence: stop when SEM/|mean| < target for all observables
    Real target_rel_sem      = 1e-2;      // 1% relative standard error

    // Hard bounds
    std::size_t min_trajectories = 100;   // Never stop before this count
    std::size_t max_trajectories = 100000;// Always stop at this count

    // Wall-clock budget (0.0 = no limit)
    Real max_wall_seconds    = 0.0;

    // Enforce min_trajectories even after convergence criterion is met
    bool enforce_min         = true;
};

// ── EnsembleConfig ────────────────────────────────────────────────────────────

struct EnsembleConfig {
    std::size_t    num_threads    = 1;
    Seed           global_seed    = 42;
    std::size_t    initial_batch  = 100;
    DiagnosticLevel diag_level   = DiagnosticLevel::Summary;
    PropagatorConfig propagator  = {};
};

} // namespace liquid
