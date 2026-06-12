#pragma once

// liquid/trajectory/trajectory_state.hpp
// ─────────────────────────────────────────────────────────────────────────────
// TrajectoryState: everything needed to describe, resume, and record
// a single MCWF trajectory.
//
// Split into two parts:
//
//   CoreState        — minimal physics state for correctness and resumability
//                      Always present. O(N) memory (N = Hilbert dim).
//
//   DiagnosticRecord — accumulated history for diagnostics and statistics
//                      Optional (controlled by DiagnosticLevel).
//                      When disabled: std::nullopt, zero overhead beyond tag.
//
// INVARIANTS (maintained by MCWFPropagator, assumed by all consumers):
//
//   INV-1 (Normalization):
//     At t=t0 and immediately after every jump:
//         core.psi.norm_sq() ≈ 1.0  (to machine precision)
//     Between jumps:
//         core.r < core.psi.norm_sq() ≤ 1.0  (strictly decreasing)
//
//   INV-2 (Jump threshold freshness):
//     core.r is always a fresh uniform draw from (0, 1].
//     It is redrawn immediately after every jump.
//     It is NEVER stale.
//
//   INV-3 (RNG exclusivity):
//     core.rng is the sole source of randomness for this trajectory.
//     No global, static, or thread-local RNG is accessed during propagation.
//
//   INV-4 (Time monotonicity):
//     core.t is non-decreasing. Never modified except by MCWFPropagator.
//
//   INV-5 (Diagnostic consistency):
//     If diagnostics.has_value():
//         diagnostics->total_jumps == diagnostics->jumps.size()
//         diagnostics->total_steps >= diagnostics->total_jumps
//
//   INV-6 (Identity immutability):
//     core.traj_id is set by make_trajectory_state() and never modified.
//
//   INV-7 (t_final immutability):
//     core.t_final is set by make_trajectory_state() and never modified.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/core/config.hpp"
#include "liquid/core/rng.hpp"
#include "liquid/linalg/dense.hpp"
#include <cassert>
#include <optional>
#include <vector>

namespace liquid::trajectory {

// ─────────────────────────────────────────────────────────────────────────────
// JumpEvent: record of a single quantum jump
// ─────────────────────────────────────────────────────────────────────────────

struct JumpEvent {
    Real        t;            // Time of jump
    std::size_t channel;      // Which Lindblad channel fired (0-indexed)
    Real        norm_before;  // ‖ψ‖² just before the jump
                              // Diagnostic value — records how far the norm
                              // had decayed before the jump occurred.
                              // Redundant with norm_samples if those are stored,
                              // but kept here for O(1) access. 24 bytes/jump.
};

// ─────────────────────────────────────────────────────────────────────────────
// StepRecord: record of one accepted ODE step
// ─────────────────────────────────────────────────────────────────────────────

struct StepRecord {
    Real           t;           // Time at start of step
    Real           dt;          // Accepted stepsize
    Real           local_err;   // Error estimate (0 for fixed-step solvers)
    std::uint32_t  rejections;  // Rejected sub-steps before acceptance
};

// ─────────────────────────────────────────────────────────────────────────────
// DiagnosticRecord: accumulated history (optional, conditionally enabled)
// ─────────────────────────────────────────────────────────────────────────────

struct DiagnosticRecord {
    // ── Jump history (enabled when level >= Summary) ──────────────────────────
    std::vector<JumpEvent> jumps;   // One entry per quantum jump

    // ── Step history (enabled when level >= StepHistory) ─────────────────────
    std::vector<StepRecord> steps;  // One entry per accepted ODE step

    // ── Norm samples (enabled when level >= Full) ─────────────────────────────
    // Subsampled, not every step. Rate controlled by MCWFPropagator config.
    std::vector<std::pair<Real, Real>> norm_samples;  // (t, ‖ψ‖²)

    // ── Summary statistics (always present when diagnostics are enabled) ──────
    // These are O(1) space and always updated regardless of DiagnosticLevel.
    // Layer 4 reads these without needing the full history vectors.
    std::uint64_t total_steps    = 0;   // Total accepted ODE steps
    std::uint64_t rejected_steps = 0;   // Total rejected ODE steps
    std::uint64_t total_jumps    = 0;   // Total quantum jumps (== jumps.size())
    Real          wall_time_s    = 0.0; // Actual compute time (seconds)
    Real          mean_dt        = 0.0; // Running mean accepted stepsize
    Real          min_dt         = 1e300; // Minimum accepted stepsize

    // Diagnostic level that was active when this record was created.
    // Determines which vectors above are populated.
    DiagnosticLevel level = DiagnosticLevel::Summary;

    // ── Convenience accessors ─────────────────────────────────────────────────

    void record_jump(Real t, std::size_t ch, Real norm_sq_before) {
        ++total_jumps;
        if (level >= DiagnosticLevel::Summary) {
            jumps.push_back({t, ch, norm_sq_before});
        }
    }

    void record_step(Real t, Real dt, Real err, std::uint32_t rejects) {
        ++total_steps;
        rejected_steps += rejects;

        // Running mean: mean_n = mean_{n-1} + (dt - mean_{n-1}) / n
        mean_dt += (dt - mean_dt) / static_cast<Real>(total_steps);
        if (dt < min_dt) min_dt = dt;

        if (level >= DiagnosticLevel::StepHistory) {
            steps.push_back({t, dt, err, rejects});
        }
    }

    void record_norm_sample(Real t, Real norm_sq) {
        if (level >= DiagnosticLevel::Full) {
            norm_samples.push_back({t, norm_sq});
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CoreState: the minimal physics state
// ─────────────────────────────────────────────────────────────────────────────

struct CoreState {
    StateVector      psi;       // Current wavefunction (see INV-1)
    Real             t;         // Current simulation time (see INV-4)
    Real             t_final;   // End time — immutable after init (INV-7)
    Real             r;         // Jump threshold, uniform in (0,1] (INV-2)
    Real             dt_last;   // Last accepted stepsize (warm-start hint)
    Real             err_last;  // Last local error estimate (stepsize control)
    RNGState         rng;       // Trajectory-local RNG (INV-3)
    TrajId           traj_id;   // Immutable trajectory index (INV-6)
    TrajectoryStatus status;    // Current lifecycle status
};

// ─────────────────────────────────────────────────────────────────────────────
// TrajectoryState: the complete trajectory record
// ─────────────────────────────────────────────────────────────────────────────

struct TrajectoryState {
    CoreState                    core;
    std::optional<DiagnosticRecord> diagnostics;

    // ── Convenience diagnostics access ────────────────────────────────────────
    // Calling diag() when diagnostics are disabled is a programming error.
    // Asserts in debug; undefined behaviour in release (consistent with
    // the general LiQuID policy: invariant violations are programming errors).

    DiagnosticRecord& diag() noexcept {
        assert(diagnostics.has_value());
        return *diagnostics;
    }

    const DiagnosticRecord& diag() const noexcept {
        assert(diagnostics.has_value());
        return *diagnostics;
    }

    bool has_diagnostics() const noexcept { return diagnostics.has_value(); }

    // ── Status queries ────────────────────────────────────────────────────────

    bool is_running()   const noexcept {
        return core.status == TrajectoryStatus::Running;
    }
    bool is_completed() const noexcept {
        return core.status == TrajectoryStatus::Completed;
    }
    bool is_failed()    const noexcept {
        return core.status == TrajectoryStatus::Failed;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory function: make_trajectory_state
// ─────────────────────────────────────────────────────────────────────────────
//
// This is the ONLY way to create a TrajectoryState.
// Making it a free function (not a constructor or propagator method) keeps
// TrajectoryState as a plain data aggregate and separates initialization
// logic from the physics propagation logic.
//
// Preconditions:
//   psi0.norm() > 0           (must be a valid state)
//   t0 < t_final              (must have a non-trivial time interval)
//   Seeding: seed = hash(global_seed, traj_id) — caller's responsibility
//
// Postconditions:
//   All CoreState invariants hold.
//   psi is normalized.
//   r is drawn from the provided RNG.
//   status = Initialized.

inline TrajectoryState make_trajectory_state(
    StateVector      psi0,
    Real             t0,
    Real             t_final,
    TrajId           traj_id,
    Seed             global_seed,
    DiagnosticLevel  diag_level = DiagnosticLevel::Summary)
{
    assert(t0 < t_final);
    assert(!psi0.empty());
    assert(psi0.norm() > 0.0);

    TrajectoryState ts;

    // Normalize initial state
    psi0.normalize();

    // Seed the trajectory-local RNG
    RNGState rng;
    rng.seed(global_seed, traj_id);

    // Draw initial jump threshold
    const Real r0 = rng.draw_uniform();

    ts.core = CoreState{
        /* psi     */ std::move(psi0),
        /* t       */ t0,
        /* t_final */ t_final,
        /* r       */ r0,
        /* dt_last */ 0.0,   // signals: use PropagatorConfig::dt_initial
        /* err_last*/ 0.0,
        /* rng     */ rng,
        /* traj_id */ traj_id,
        /* status  */ TrajectoryStatus::Initialized
    };

    if (diag_level != DiagnosticLevel::None) {
        DiagnosticRecord rec;
        rec.level = diag_level;
        ts.diagnostics = std::move(rec);
    }

    return ts;
}

} // namespace liquid::trajectory
