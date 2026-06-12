#pragma once

// liquid/core/types.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Fundamental scalar and index types for LiQuID.
//
// Design decisions (frozen):
//   - Scalar is std::complex<double>. Not templated. Not float.
//     Rationale: quantum amplitudes require double precision. float32 is a
//     future GPU concern and will be handled by a separate policy, not by
//     templating the entire framework.
//   - All dimensions and indices are std::size_t.
//   - All enums are enum class for type safety and scoping.
// ─────────────────────────────────────────────────────────────────────────────

#include <complex>
#include <cstddef>
#include <cstdint>

namespace liquid {

// ── Numeric types ─────────────────────────────────────────────────────────────

using Scalar  = std::complex<double>;   // Complex amplitude
using Real    = double;                 // Real-valued quantity
using Dim     = std::size_t;            // Hilbert space dimension
using Idx     = std::size_t;            // General index
using TrajId  = std::uint64_t;         // Trajectory identifier (never reused)
using Seed    = std::uint64_t;         // RNG seed

// Imaginary unit — use as: liquid::i * 2.5, not std::complex<double>(0,1)*2.5
// This is a named constant, not a macro, and respects the liquid:: namespace.
inline constexpr Scalar i_unit{0.0, 1.0};

// ── Diagnostic level ──────────────────────────────────────────────────────────
//
// Controls what DiagnosticRecord accumulates.
// Higher levels carry higher memory and runtime cost.
//
// None:        Only summary statistics (O(1) space). Zero per-step overhead.
// Summary:     Summary + jump history. O(N_jumps) space. DEFAULT.
// Full:        Summary + jump history + norm samples. O(N_steps) space.
// StepHistory: All of the above + per-step ODE records. EXPENSIVE.

enum class DiagnosticLevel : std::uint8_t {
    None        = 0,
    Summary     = 1,
    Full        = 2,
    StepHistory = 3
};

// ── Trajectory status ─────────────────────────────────────────────────────────
//
// Lifecycle: Initialized → Running → {Completed | Failed}
// The Suspended state is set by the EnsembleManager for adaptive allocation.
// A Suspended trajectory retains its full CoreState and can be resumed exactly.

enum class TrajectoryStatus : std::uint8_t {
    Initialized = 0,  // Created, not yet started
    Running     = 1,  // Active evolution in progress
    Completed   = 2,  // Reached t_final successfully
    Failed      = 3,  // Numerical failure (stepsize collapse, etc.)
    Suspended   = 4   // Paused by EnsembleManager
};

// ── Solver type tags (Phase 1: fixed-step RK4 only) ──────────────────────────

enum class SolverType : std::uint8_t {
    RK4Fixed        = 0,   // Phase 1: fixed-step Runge-Kutta 4
    DormandPrince45 = 1,   // Phase 3: adaptive RK45 (Dormand-Prince)
    RK23Adaptive    = 2    // Phase 3: adaptive RK23 (lighter weight)
};

enum class JumpDetectorType : std::uint8_t {
    Threshold           = 0,  // Phase 1: simple norm < r threshold
    LinearInterpBisect  = 1   // Phase 3: interpolation + bisection
};

// ── Storage tags (for operator type dispatch) ─────────────────────────────────

struct DenseTag  {};   // N×N dense matrix
struct SparseTag {};   // CSR sparse matrix

// ── Time-dependence tags ──────────────────────────────────────────────────────

struct TimeIndependentTag {};
struct TimeDependentTag   {};

} // namespace liquid
