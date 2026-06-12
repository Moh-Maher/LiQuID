#pragma once

// liquid/system/lindblad.hpp
// ─────────────────────────────────────────────────────────────────────────────
// LindbladSet: stores a collection of jump operators {Lₖ} and their
// precomputed aggregate Γ = Σₖ Lₖ†Lₖ.
//
// INVARIANT (established at construction, never violated):
//   gamma_ == Σₖ ops_[k].adjoint() * ops_[k]
//
// INVARIANT (immutability):
//   After construction, LindbladSet is immutable.
//   No operator can be added, removed, or modified.
//   This eliminates an entire class of bugs where gamma_ becomes stale.
//
// Why precompute Γ?
//   The non-Hermitian effective Hamiltonian is:
//       H_eff = H - (i/2) Γ
//   The term -(i/2)Γ|ψ⟩ is evaluated at EVERY ODE sub-step.
//   Without precomputation: cost = O(K · N²) per step (K operators, dim N)
//   With precomputation:    cost = O(N²) per step (single matrix-vector)
//   For K=5, N=100: 5× reduction in the dominant inner-loop cost.
//
// Access patterns:
//   HOT PATH  (every ODE step):   apply_decay() — uses precomputed gamma_
//   COLD PATH (at each jump):     apply_channel() — uses individual ops_[k]
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/linalg/dense.hpp"
#include <cassert>
#include <stdexcept>
#include <vector>

namespace liquid {

template<typename StorageTag>
class LindbladSet;

// ─────────────────────────────────────────────────────────────────────────────
// Dense specialization (Phase 1)
// ─────────────────────────────────────────────────────────────────────────────

template<>
class LindbladSet<DenseTag> {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // Construct from a list of jump operators.
    // Precomputes gamma_ = Σₖ Lₖ†Lₖ immediately.
    // Precondition: all operators must have the same dimension.
    // Precondition: operators must not be empty.
    explicit LindbladSet(std::vector<DenseOperator> ops) {
        if (ops.empty()) {
            throw std::invalid_argument(
                "LindbladSet: must have at least one operator. "
                "For closed systems, use a Hamiltonian without LindbladSet.");
        }

        dim_ = ops[0].size();
        for (std::size_t k = 1; k < ops.size(); ++k) {
            if (ops[k].size() != dim_) {
                throw std::invalid_argument(
                    "LindbladSet: all operators must have the same dimension. "
                    "Operator 0 has dimension " + std::to_string(dim_) +
                    " but operator " + std::to_string(k) +
                    " has dimension " + std::to_string(ops[k].size()) + ".");
            }
        }

        ops_ = std::move(ops);
        precompute_gamma();
    }

    // No copy (potentially large; force explicit clone if needed)
    // Move is allowed
    LindbladSet(const LindbladSet&)            = delete;
    LindbladSet& operator=(const LindbladSet&) = delete;
    LindbladSet(LindbladSet&&)                 = default;
    LindbladSet& operator=(LindbladSet&&)      = default;

    // ── HOT PATH ──────────────────────────────────────────────────────────────

    // Accumulate decay contribution into out:
    //   out += -(i/2) * Γ * psi
    // where Γ = Σₖ Lₖ†Lₖ (precomputed at construction).
    //
    // This is called at every ODE sub-step. Must be as fast as possible.
    // Preconditions: psi.size() == dim_, out.size() == dim_
    void apply_decay(const StateVector& psi, StateVector& out) const noexcept {
        // -(i/2) = i_unit * (-0.5) = Scalar{0, -0.5}
        // We want: out += -i/2 * gamma_ * psi
        // Using apply_add with alpha = -i/2
        constexpr Scalar neg_i_half{0.0, -0.5};
        gamma_.apply_add(psi, neg_i_half, out);
    }

    // ── COLD PATH ─────────────────────────────────────────────────────────────

    // Number of jump channels
    std::size_t num_channels() const noexcept { return ops_.size(); }

    // Hilbert space dimension
    Dim hilbert_dim() const noexcept { return dim_; }

    // Access individual operator Lₖ (read-only)
    const DenseOperator& channel_op(std::size_t k) const noexcept {
        assert(k < ops_.size());
        return ops_[k];
    }

    // Compute out = Lₖ * psi  (for jump probability and post-jump state)
    // Preconditions: k < num_channels(), psi.size() == dim_, out.size() == dim_
    void apply_channel(std::size_t k,
                       const StateVector& psi,
                       StateVector& out) const {
        assert(k < ops_.size());
        out.set_zero();
        ops_[k].apply_add(psi, Scalar{1.0, 0.0}, out);
    }

    // Compute jump probability for channel k: pₖ = ‖Lₖ|ψ⟩‖²
    // The caller is responsible for providing scratch space.
    // Preconditions: k < num_channels(), psi.size() == dim_, scratch.size() == dim_
    Real channel_probability(std::size_t k,
                              const StateVector& psi,
                              StateVector& scratch) const {
        apply_channel(k, psi, scratch);
        return scratch.norm_sq();
    }

    // Compute all jump probabilities at once.
    // probs_out must have size >= num_channels().
    // scratch must have size == dim_.
    // Returns total probability Σₖ pₖ (= 1 - ‖|ψ(t+dt)⟩‖² for MCWF norm)
    Real all_probabilities(const StateVector& psi,
                            StateVector& scratch,
                            std::vector<Real>& probs_out) const {
        assert(probs_out.size() >= ops_.size());
        Real total = 0.0;
        for (std::size_t k = 0; k < ops_.size(); ++k) {
            probs_out[k] = channel_probability(k, psi, scratch);
            total += probs_out[k];
        }
        return total;
    }

    // ── Introspection ──────────────────────────────────────────────────────────

    // Access precomputed Γ (for diagnostics, not for physics in the hot path)
    const DenseOperator& decay_operator() const noexcept { return gamma_; }

private:
    void precompute_gamma() {
        gamma_ = DenseOperator(dim_);  // zero-initialized
        for (const auto& L : ops_) {
            // gamma_ += L†L
            DenseOperator LdagL = adjoint_times(L);
            gamma_.add_scaled(LdagL, Scalar{1.0, 0.0});
        }
    }

    std::vector<DenseOperator> ops_;   // {Lₖ} — individual jump operators
    DenseOperator              gamma_; // Σₖ Lₖ†Lₖ — precomputed, immutable
    Dim                        dim_{0};
};

} // namespace liquid
