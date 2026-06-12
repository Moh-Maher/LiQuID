#pragma once

// liquid/ensemble/convergence_monitor.hpp
// ─────────────────────────────────────────────────────────────────────────────
// ConvergenceMonitor: decides when the ensemble simulation should stop.
//
// Design principles:
//   - Stateless except for a single atomic stop flag (for external interrupts)
//   - evaluate() is a pure function of (statistics, elapsed_time, criteria)
//   - Trivially testable: no simulation needed, just feed statistics
//   - Does NOT manage trajectories or threads — only answers "stop or not?"
//
// Stopping criteria hierarchy (all must be satisfied to stop):
//   1. min_trajectories must have completed (hard floor)
//   2. Statistical criterion: max relative SEM < target_rel_sem
//      OR max_trajectories reached (hard ceiling)
//      OR wall-clock budget exceeded
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/config.hpp"
#include "liquid/ensemble/observable_accumulator.hpp"
#include <atomic>
#include <string>

namespace liquid::ensemble {

enum class StoppingDecision : std::uint8_t {
    Continue,    // Keep running: not converged yet
    Converged,   // Statistical criterion satisfied
    BudgetHit,   // max_trajectories or wall-clock reached
    Forced       // External stop request (SIGINT, user call)
};

struct ConvergenceReport {
    StoppingDecision decision;
    std::size_t      trajectories_completed;
    Real             current_rel_sem;       // worst-case across observables
    Real             wall_time_elapsed;
    std::string      reason;
};

class ConvergenceMonitor {
public:
    explicit ConvergenceMonitor(StoppingCriteria criteria)
        : criteria_(criteria)
        , stop_requested_(false)
    {}

    // ── Primary interface ─────────────────────────────────────────────────────

    // Evaluate stopping criterion given current accumulator state.
    // Pure function: does not mutate any state except stop_requested_ check.
    // Called after each trajectory completes.
    ConvergenceReport evaluate(const ObservableAccumulator& accum,
                                Real wall_time_elapsed) const {
        const std::size_t N = accum.statistics().count();
        const Real rel_sem  = (N >= 2)
                              ? accum.statistics().max_relative_sem()
                              : std::numeric_limits<Real>::infinity();

        // External stop request
        if (stop_requested_.load(std::memory_order_relaxed)) {
            return {StoppingDecision::Forced, N, rel_sem, wall_time_elapsed,
                    "External stop requested"};
        }

        // Hard ceiling
        if (N >= criteria_.max_trajectories) {
            return {StoppingDecision::BudgetHit, N, rel_sem, wall_time_elapsed,
                    "max_trajectories reached (" +
                    std::to_string(criteria_.max_trajectories) + ")"};
        }

        // Wall-clock budget
        if (criteria_.max_wall_seconds > 0.0 &&
            wall_time_elapsed >= criteria_.max_wall_seconds) {
            return {StoppingDecision::BudgetHit, N, rel_sem, wall_time_elapsed,
                    "Wall-clock budget exceeded"};
        }

        // Must have minimum trajectories
        if (criteria_.enforce_min && N < criteria_.min_trajectories) {
            return {StoppingDecision::Continue, N, rel_sem, wall_time_elapsed,
                    "Below minimum trajectory count"};
        }

        // Statistical convergence
        if (N >= 2 && rel_sem <= criteria_.target_rel_sem) {
            return {StoppingDecision::Converged, N, rel_sem, wall_time_elapsed,
                    "Relative SEM " + fmt(rel_sem) +
                    " <= target " + fmt(criteria_.target_rel_sem)};
        }

        return {StoppingDecision::Continue, N, rel_sem, wall_time_elapsed,
                "Rel SEM " + fmt(rel_sem) + " > target " +
                fmt(criteria_.target_rel_sem)};
    }

    // ── External stop ─────────────────────────────────────────────────────────

    // Thread-safe: can be called from a signal handler or another thread.
    void request_stop() noexcept {
        stop_requested_.store(true, std::memory_order_relaxed);
    }

    bool stop_was_requested() const noexcept {
        return stop_requested_.load(std::memory_order_relaxed);
    }

    void clear_stop_request() noexcept {
        stop_requested_.store(false, std::memory_order_relaxed);
    }

    const StoppingCriteria& criteria() const noexcept { return criteria_; }

private:
    static std::string fmt(Real v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.4g", v);
        return buf;
    }

    StoppingCriteria         criteria_;
    mutable std::atomic<bool> stop_requested_;
};

} // namespace liquid::ensemble
