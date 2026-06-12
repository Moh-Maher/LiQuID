#pragma once

// liquid/trajectory/mcwf_propagator.hpp
// ─────────────────────────────────────────────────────────────────────────────
// MCWFPropagator: implements the Monte Carlo Wave Function algorithm.
//
// Templated on:
//   SystemType   — OpenSystem<...>, provides apply_Heff, jump_probabilities,
//                  apply_jump, hilbert_dim, num_channels
//   StepperType  — ODE stepper policy (e.g. RK4Stepper)
//
// ALGORITHM (one call to advance()):
//
//   1. Compute suggested dt (from config or warm-start)
//   2. Attempt one ODE step under H_eff: |ψ(t)⟩ → |ψ(t+dt)⟩
//   3. Check: did ‖|ψ(t+dt)⟩‖² drop below r?
//      NO  → accept step, advance t, update diagnostics, continue loop
//      YES → a jump occurred in [t, t+dt]
//             a. Simple jump location: use t+dt as jump time (Phase 1)
//             b. Compute jump channel probabilities pₖ = ‖Lₖ|ψ⟩‖²
//             c. Select channel by weighted draw from rng
//             d. Apply jump: |ψ⟩ → Lₖ|ψ⟩ / ‖Lₖ|ψ⟩‖
//             e. Redraw r from rng
//             f. Record JumpEvent in diagnostics
//             g. Return JumpOccurred to caller
//   4. If t >= t_target: return Advanced
//   5. If t >= t_final:  set status=Completed, return Completed
//   6. If stepsize collapsed: set status=Failed, return Failed
//
// EVENT-DRIVEN ADVANCE MODEL:
//   advance() returns at the FIRST jump, not after all jumps to t_target.
//   The ensemble manager decides whether to continue, inspect, or suspend.
//   run_to_completion() wraps advance() in a loop for simple use cases.
//
// THREAD SAFETY:
//   MCWFPropagator is NOT thread-safe. Each thread owns one instance.
//   The system_ pointer is read-only (safe to share across threads).
//   All mutable state (stepper scratch, prob_scratch_) is per-instance.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/core/config.hpp"
#include "liquid/trajectory/trajectory_state.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <vector>

namespace liquid::trajectory {

// ── StepOutcome ───────────────────────────────────────────────────────────────

enum class StepOutcome : std::uint8_t {
    Advanced,      // t advanced to t_target, no jump
    JumpOccurred,  // A jump occurred; t is at jump time; state is post-jump
    Completed,     // t reached t_final; status set to Completed
    Failed         // Numerical failure; status set to Failed
};

// ── MCWFPropagator ────────────────────────────────────────────────────────────

template<typename SystemType, typename StepperType>
class MCWFPropagator {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // system:  non-owning pointer. System must outlive the propagator.
    //          Multiple propagators (threads) may share one system.
    // stepper: owned by value. One stepper per propagator (per thread).
    // config:  copied by value.
    MCWFPropagator(
        const SystemType*  system,
        StepperType        stepper,
        PropagatorConfig   config = {})
        : system_(system)
        , stepper_(std::move(stepper))
        , config_(config)
        , psi_scratch_(system->hilbert_dim())
        , psi_proposed_(system->hilbert_dim())
        , psi_before_(system->hilbert_dim())
        , accum_tmp_(system->hilbert_dim())
        , prob_scratch_(system->num_channels(), 0.0)
    {
        assert(system_ != nullptr);
    }

    // ── Core operation ────────────────────────────────────────────────────────

    // Advance the trajectory toward t_target.
    //
    // Stops at the FIRST of:
    //   (a) t reaches t_target         → returns Advanced
    //   (b) a quantum jump occurs       → returns JumpOccurred
    //   (c) t reaches core.t_final     → returns Completed
    //   (d) numerical failure           → returns Failed
    //
    // Mutates state in-place.
    // Does NOT reset stepper between calls (warm-start continuity).
    StepOutcome advance(TrajectoryState& ts, Real t_target) {
        CoreState& core = ts.core;

        assert(core.status == TrajectoryStatus::Initialized ||
               core.status == TrajectoryStatus::Running     ||
               core.status == TrajectoryStatus::Suspended);

        core.status = TrajectoryStatus::Running;

        const Real t_stop = std::min(t_target, core.t_final);

        // Determine initial stepsize
        Real dt = (core.dt_last > 0.0) ? core.dt_last : config_.dt_initial;
        dt = std::min(dt, config_.dt_max);
        dt = std::max(dt, config_.dt_min);

        while (core.t < t_stop - 1e-14 * std::abs(t_stop + 1.0)) {
            const Real t_remaining = t_stop - core.t;
            const Real dt_clamped  = std::min(dt, t_remaining);
            if (dt_clamped <= 0.0) break;

            // Save pre-step state for jump processing
            psi_before_.copy_from(core.psi);  // save pre-step wavefunction

            // Attempt one ODE step
            // RHS: d|ψ⟩/dt = -i H_eff |ψ⟩
            // apply_Heff(psi, out) sets out = -i*H_eff*psi (overwrites)
            // The stepper expects: accum += f(t, psi)
            // So we write to accum_tmp_ then add into the stepper's accumulator.
            auto rhs_adapted = [this](Real /*t_rhs*/, const StateVector& psi,
                                      StateVector& accum) {
                system_->apply_Heff(psi, accum_tmp_);
                for (Dim i = 0; i < psi.size(); ++i) {
                    accum[i] += accum_tmp_[i];
                }
            };

            auto result = stepper_.try_step(
                core.t, core.psi, psi_proposed_,
                dt_clamped, t_stop, rhs_adapted);

            // Update diagnostics for this step
            if (ts.has_diagnostics()) {
                ts.diag().record_step(core.t, result.dt_taken, result.local_err, 0);
            }

            core.dt_last  = result.dt_next;
            core.err_last = result.local_err;
            dt            = result.dt_next;

            // Check for quantum jump
            const Real norm_sq_proposed = psi_proposed_.norm_sq();

            if (norm_sq_proposed > core.r) {
                // No jump: accept step
                core.psi.copy_from(psi_proposed_);
                core.t += result.dt_taken;
            } else {
                // Jump detected in [t_before, t_before + dt_taken]
                // Phase 1: simple jump location — use the proposed time
                // (no bisection). The jump is placed at t_before + dt_taken.
                // Phase 3 will replace this with interpolation + bisection.
                const Real t_jump = core.t + result.dt_taken;
                const Real norm_before = core.psi.norm_sq();

                // Use psi_proposed_ as the pre-jump state
                // (it has the lowest norm, closest to the jump moment)
                // In Phase 1 we do not locate the exact jump time.

                // Compute per-channel jump probabilities
                // Use psi_before_ (the state at t_before) for probability
                // weights — this is the pre-step state where we know
                // ‖ψ‖² was still above r.
                const Real p_total = system_->jump_probabilities(
                    core.psi, psi_scratch_, prob_scratch_);

                // Select jump channel
                const std::size_t k = select_channel(core.rng, p_total);

                // Apply jump: |ψ⟩ → Lₖ|ψ⟩ / ‖Lₖ|ψ⟩‖
                system_->apply_jump(k, core.psi, psi_scratch_);

                // Advance time to jump location
                core.t = t_jump;

                // Redraw jump threshold (INV-2: always fresh after jump)
                core.r = core.rng.draw_uniform();

                // Reset stepper after jump (state is discontinuous)
                stepper_.reset(config_.dt_initial);
                dt = config_.dt_initial;

                // Record jump event
                if (ts.has_diagnostics()) {
                    ts.diag().record_jump(t_jump, k, norm_before);
                }

                return StepOutcome::JumpOccurred;
            }

            // Check for stepsize collapse (numerical failure)
            if (result.dt_taken < config_.dt_min && dt_clamped > config_.dt_min) {
                core.status = TrajectoryStatus::Failed;
                return StepOutcome::Failed;
            }
        }

        // Reached t_stop. Determine why.
        const Real eps = 1e-12 * (std::abs(core.t_final) + 1.0);
        if (core.t >= core.t_final - eps) {
            core.status = TrajectoryStatus::Completed;
            return StepOutcome::Completed;
        }

        // Reached t_target (< t_final): normal chunked advance
        return StepOutcome::Advanced;
    }

    // ── Convenience: run to completion ───────────────────────────────────────

    // Equivalent to calling advance(ts, ts.core.t_final) in a loop
    // until Completed or Failed.
    // Returns the final status.
    TrajectoryStatus run_to_completion(TrajectoryState& ts) {
        auto t_start_wall = std::chrono::steady_clock::now();

        StepOutcome outcome = StepOutcome::Advanced;
        while (outcome != StepOutcome::Completed &&
               outcome != StepOutcome::Failed) {
            outcome = advance(ts, ts.core.t_final);
        }

        // Record wall time if diagnostics are enabled
        if (ts.has_diagnostics()) {
            auto t_end_wall = std::chrono::steady_clock::now();
            ts.diag().wall_time_s =
                std::chrono::duration<Real>(t_end_wall - t_start_wall).count();
        }

        return ts.core.status;
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    const PropagatorConfig& config() const noexcept { return config_; }
    const SystemType&       system() const noexcept { return *system_; }
    const StepperType&      stepper() const noexcept { return stepper_; }
    StepperType&            stepper()       noexcept { return stepper_; }

private:
    // ── Channel selection ─────────────────────────────────────────────────────

    // Select a jump channel from prob_scratch_ using weighted random draw.
    // Returns the selected channel index.
    std::size_t select_channel(RNGState& rng, Real p_total) const {
        const std::size_t K = system_->num_channels();

        if (K == 1) return 0;

        // Draw a uniform value in [0, p_total)
        const Real u = rng.draw_uniform() * p_total;

        Real cumulative = 0.0;
        for (std::size_t k = 0; k < K - 1; ++k) {
            cumulative += prob_scratch_[k];
            if (u < cumulative) return k;
        }
        return K - 1;  // last channel (catches floating-point rounding)
    }

    // ── Data ──────────────────────────────────────────────────────────────────

    const SystemType*     system_;       // non-owning, read-only, thread-safe
    StepperType           stepper_;      // owned; NOT shared across threads
    PropagatorConfig      config_;

    // Per-propagator scratch space (no per-step allocation)
    // Sized at construction from system_->hilbert_dim()
    StateVector           psi_scratch_;    // general scratch
    StateVector           psi_proposed_;   // proposed state after ODE step
    StateVector           psi_before_;     // state saved before each step
    StateVector           accum_tmp_;      // RHS accumulation temporary
    mutable std::vector<Real> prob_scratch_;  // per-channel jump probabilities
};

// ── Deduction guide — not needed in C++17 with explicit template args,
//    but provided for convenience when types can be deduced.

} // namespace liquid::trajectory
