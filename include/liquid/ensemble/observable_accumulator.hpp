#pragma once

// liquid/ensemble/observable_accumulator.hpp
// ─────────────────────────────────────────────────────────────────────────────
// ObservableAccumulator: bridges TrajectoryState (physics) and
// RunningStatistics (mathematics).
//
// Responsibilities:
//   1. Hold a list of named observable definitions (type-erased callables)
//   2. Evaluate all observables on a given wavefunction
//   3. Feed results into RunningStatistics
//   4. Support parallel merge (each thread owns one instance)
//
// Type erasure boundary:
//   The hot path (ODE steps, jump detection) uses concrete templated types
//   with zero virtual dispatch. Observable evaluation is the boundary where
//   we deliberately cross into type erasure via std::function.
//
//   Cost: one indirect call per observable per trajectory completion.
//   For 10 observables and 10,000 trajectories: ~100,000 indirect calls.
//   At ~10ns each: ~1ms total. Negligible vs trajectory simulation time.
//
// Observable formula (MCWF):
//   For an observable O and a (possibly unnormalized) MCWF state |ψ⟩:
//       <O> = <ψ|O|ψ> / <ψ|ψ>
//   The ObservableFn must implement this normalization internally.
//   Reason: the accumulator does not know whether |ψ⟩ is normalized.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/linalg/dense.hpp"
#include "liquid/ensemble/running_statistics.hpp"
#include <cassert>
#include <functional>
#include <string>
#include <vector>

namespace liquid::ensemble {

// ── Observable definition ─────────────────────────────────────────────────────

// An observable is a callable: (StateVector) → Real
// The function receives the (possibly unnormalized) MCWF state.
// It is responsible for normalization before computing <O>.
//
// Convenience constructors are provided in SimulationBuilder for
// common cases (DenseOperator expectation value).
using ObservableFn = std::function<Real(const StateVector&)>;

struct ObservableDef {
    std::string  name;  // for labeling output
    ObservableFn eval;  // the computation
};

// ── ObservableAccumulator ─────────────────────────────────────────────────────

class ObservableAccumulator {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    explicit ObservableAccumulator(std::vector<ObservableDef> observables)
        : observables_(std::move(observables))
        , stats_(observables_.size())
        , eval_scratch_(observables_.size(), 0.0)
    {
        assert(!observables_.empty());
    }

    // ── Core operation ────────────────────────────────────────────────────────

    // Evaluate all observables on psi and update statistics.
    // Called once per trajectory at the accumulation point (t_final).
    // psi may be unnormalized — each ObservableFn handles normalization.
    void accumulate(const StateVector& psi) {
        for (std::size_t i = 0; i < observables_.size(); ++i) {
            eval_scratch_[i] = observables_[i].eval(psi);
        }
        stats_.update(eval_scratch_);
    }

    // ── Parallel merge ────────────────────────────────────────────────────────

    // Merge another accumulator into this one.
    // Used for OpenMP parallel reduction.
    // Precondition: other has the same observables (same count).
    void merge(const ObservableAccumulator& other) {
        assert(other.observables_.size() == observables_.size());
        stats_.merge(other.stats_);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    std::size_t num_observables() const noexcept {
        return observables_.size();
    }

    const std::string& name(std::size_t i) const noexcept {
        assert(i < observables_.size());
        return observables_[i].name;
    }

    const RunningStatistics& statistics() const noexcept {
        return stats_;
    }

    void reset() {
        stats_.reset();
    }

private:
    std::vector<ObservableDef> observables_;
    RunningStatistics          stats_;
    std::vector<Real>          eval_scratch_;  // avoids per-accumulate alloc
};

// ── Free function: make observable from DenseOperator ─────────────────────────

// Convenience: create an ObservableDef that computes <ψ|O|ψ> / <ψ|ψ>
// for a given DenseOperator O.
// Takes O by value (copied into the closure).
inline ObservableDef make_operator_observable(std::string name,
                                               DenseOperator O) {
    return ObservableDef{
        std::move(name),
        [op = std::move(O)](const StateVector& psi) -> Real {
            const Real norm_sq = psi.norm_sq();
            if (norm_sq < 1e-300) return 0.0;
            // <O> = Re(<ψ|O|ψ>) / <ψ|ψ>
            // For Hermitian O, the imaginary part is zero by construction.
            StateVector scratch(psi.size());
            op.apply_add(psi, Scalar{1.0, 0.0}, scratch);
            const Scalar raw = StateVector::inner(psi, scratch);
            return raw.real() / norm_sq;
        }
    };
}

} // namespace liquid::ensemble
