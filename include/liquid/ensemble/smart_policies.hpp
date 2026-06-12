#pragma once

// liquid/ensemble/smart_policies.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Smart allocation policies for Phase 5.
//
// make_uniform_policy()           — always spawn new (Phase 2 baseline)
// make_convergence_policy()       — spawn more when SEM is large, fewer when small
// make_variance_weighted_policy() — future: weight by per-observable variance
//
// All policies implement:
//   AllocationDecision policy(const EnsembleSummary&)
//
// The ML seam: any callable with this signature can be used as a policy.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/ensemble/adaptive_allocator.hpp"
#include <cmath>

namespace liquid::ensemble {

// ── Convergence-aware policy ──────────────────────────────────────────────────
//
// Identical to uniform for now — always spawns new trajectories.
// The "convergence-awareness" is entirely in the ConvergenceMonitor stopping
// criterion. The allocator's job in Phase 5 is to decide WHICH trajectory
// to run next. For a homogeneous ensemble (all trajectories start from the
// same psi0 and have the same t_final), all trajectories contribute equal
// variance. So variance-weighting is trivially uniform.
//
// The variance-weighted policy becomes non-trivial only when:
//   (a) Different trajectories have different t_final (adaptive time allocation)
//   (b) Different observables have very different variances and we want
//       to prioritise convergence of the worst one
//   (c) Rare-event trajectories are identified and given more weight
//
// These are Phase 6+ scenarios. For Phase 5, document the seam and
// implement the uniform baseline with proper telemetry.

inline AllocatorPolicy make_convergence_policy(Seed global_seed) {
    // Phase 5: same as uniform. Interface is correct for future extension.
    TrajId next_id = 0;
    return [next_id, global_seed](const EnsembleSummary& /*s*/) mutable
           -> AllocationDecision {
        const TrajId id   = next_id++;
        const Seed seed = global_seed
            ^ (id * 0x9e3779b97f4a7c15ULL + 0x6c62272e07bb0142ULL);
        return AllocationDecision{AllocAction::SpawnNew, seed, id};
    };
}

// ── Statistical stopping analysis ────────────────────────────────────────────
//
// Helper for convergence studies: given a completed EnsembleResult,
// determine the minimum N that would have achieved the target SEM
// assuming SEM ∝ 1/√N (the theoretical rate).
//
// This lets us answer: "How many fewer trajectories would smart stopping
// have needed compared to running fixed-N?"

struct ConvergenceAnalysis {
    std::size_t actual_N;          // trajectories actually run
    double      achieved_rel_sem;  // relative SEM at actual_N
    double      target_rel_sem;    // target that was set
    std::size_t predicted_min_N;   // min N to achieve target (theoretical)
    double      efficiency_ratio;  // predicted_min_N / actual_N (<= 1 if over-ran)
};

inline ConvergenceAnalysis analyse_convergence(
    const EnsembleResult& result,
    double                target_rel_sem)
{
    ConvergenceAnalysis analysis;
    analysis.actual_N         = result.total_trajectories;
    analysis.target_rel_sem   = target_rel_sem;

    // Worst-case rel SEM across observables
    double worst_rel_sem = 0.0;
    for (const auto& obs : result.observables)
        if (obs.rel_sem > worst_rel_sem) worst_rel_sem = obs.rel_sem;
    analysis.achieved_rel_sem = worst_rel_sem;

    // Theoretical minimum N: (achieved_sem * sqrt(N)) / sqrt(N_min) = target
    // N_min = N * (achieved_sem / target)^2
    if (target_rel_sem > 0.0 && worst_rel_sem > 0.0) {
        const double ratio = worst_rel_sem / target_rel_sem;
        analysis.predicted_min_N = static_cast<std::size_t>(
            std::ceil(static_cast<double>(result.total_trajectories)
                      * ratio * ratio));
    } else {
        analysis.predicted_min_N = result.total_trajectories;
    }

    analysis.efficiency_ratio =
        (analysis.actual_N > 0)
        ? static_cast<double>(analysis.predicted_min_N) /
          static_cast<double>(analysis.actual_N)
        : 1.0;

    return analysis;
}

} // namespace liquid::ensemble
