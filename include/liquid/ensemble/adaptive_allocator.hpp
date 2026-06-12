#pragma once

// liquid/ensemble/adaptive_allocator.hpp
// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveAllocator: decides what trajectory to run next.
//
// This is LiQuID's scientific differentiator and the ML integration seam.
//
// Architecture:
//   The allocation policy is a std::function<AllocationDecision(EnsembleSummary)>.
//   The default (Phase 2/5) is UniformPolicy: always spawn a new trajectory.
//   A future ML policy replaces this one callable without touching anything else.
//
// Why std::function here?
//   The allocator is called once per trajectory completion — not in the hot
//   ODE loop. The indirect call overhead (~10ns) is completely negligible.
//   The flexibility to swap policies at runtime justifies it.
//
// TrajectorySummary:
//   The compact per-trajectory record the allocator sees. Deliberately does
//   NOT contain wavefunctions (too large). Contains only O(1) scalars.
//   This is also the primary training feature vector for future ML models.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/core/config.hpp"
#include <functional>
#include <vector>

namespace liquid::ensemble {

// ── TrajectorySummary ─────────────────────────────────────────────────────────
// O(1) per-trajectory record. Used by allocator and as ML training features.

struct TrajectorySummary {
    TrajId           traj_id;
    TrajectoryStatus status;
    std::uint64_t    total_jumps;
    std::uint64_t    total_steps;
    Real             wall_time_seconds;
    Real             last_t;
    Real             t_final;
    Real             mean_dt;
    Real             min_dt;
};

// ── EnsembleSummary ───────────────────────────────────────────────────────────
// What the allocator sees when making a decision.
// Passed by const reference — the allocator reads only, never writes.

struct EnsembleSummary {
    std::size_t  trajectories_completed;
    std::size_t  trajectories_running;
    std::size_t  trajectories_failed;
    Real         current_rel_sem;
    Real         wall_time_elapsed;

    // Per-trajectory summaries (does NOT contain wavefunctions)
    // Points into EnsembleManager's storage — valid only during the call.
    const std::vector<TrajectorySummary>* completed_summaries = nullptr;
};

// ── AllocationDecision ────────────────────────────────────────────────────────

enum class AllocAction : std::uint8_t {
    SpawnNew,  // Create and run a new trajectory
    Idle       // No action (ensemble is done or waiting)
};

struct AllocationDecision {
    AllocAction   action;
    Seed          new_seed;    // for SpawnNew: seed for the new trajectory
    TrajId        new_traj_id; // for SpawnNew: id of the new trajectory
};

// ── AllocatorPolicy ───────────────────────────────────────────────────────────
// THE ML SEAM.
// A policy is any callable matching this signature.
// The default policy is make_uniform_policy().
// An ML model is a callable object stored here.

using AllocatorPolicy =
    std::function<AllocationDecision(const EnsembleSummary&)>;

// ── Built-in policies ─────────────────────────────────────────────────────────

// Phase 2/5 default: always spawn a new trajectory with sequential IDs.
// Simple, correct, and the right baseline for benchmarking smart policies.
inline AllocatorPolicy make_uniform_policy() {
    // Captures mutable counter for sequential traj_id assignment.
    // The EnsembleManager provides the seed; the allocator provides the id.
    TrajId next_id = 0;
    Seed   base_seed = 0;

    return [next_id, base_seed](const EnsembleSummary& /*summary*/)
            mutable -> AllocationDecision {
        const TrajId id   = next_id++;
        // Derive seed: combine base_seed and id via simple hash
        // Same approach as RNGState::seed — independent streams per trajectory
        const Seed seed = base_seed ^ (id * 0x9e3779b97f4a7c15ULL + 0x6c62272e07bb0142ULL);
        return AllocationDecision{AllocAction::SpawnNew, seed, id};
    };
}

// Initialise the policy with a specific global seed.
// Must be called before the first decide() call.
inline AllocatorPolicy make_uniform_policy(Seed global_seed) {
    TrajId next_id = 0;

    return [next_id, global_seed](const EnsembleSummary& /*summary*/)
            mutable -> AllocationDecision {
        const TrajId id   = next_id++;
        const Seed seed = global_seed ^ (id * 0x9e3779b97f4a7c15ULL + 0x6c62272e07bb0142ULL);
        return AllocationDecision{AllocAction::SpawnNew, seed, id};
    };
}

// ── AdaptiveAllocator ─────────────────────────────────────────────────────────

class AdaptiveAllocator {
public:
    explicit AdaptiveAllocator(AllocatorPolicy policy)
        : policy_(std::move(policy))
    {}

    AllocationDecision decide(const EnsembleSummary& summary) const {
        return policy_(summary);
    }

    // Replace policy at runtime.
    // Used for: switching from uniform to ML policy mid-run,
    //           A/B testing allocation strategies.
    void set_policy(AllocatorPolicy policy) {
        policy_ = std::move(policy);
    }

private:
    mutable AllocatorPolicy policy_;
};

} // namespace liquid::ensemble
