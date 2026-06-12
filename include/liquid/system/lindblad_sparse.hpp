#pragma once

// liquid/system/lindblad_sparse.hpp
// ─────────────────────────────────────────────────────────────────────────────
// LindbladSet<SparseTag>: sparse specialization.
// Mirrors LindbladSet<DenseTag> exactly — same interface, sparse storage.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/linalg/dense.hpp"
#include "liquid/linalg/sparse.hpp"
#include <cassert>
#include <stdexcept>
#include <vector>

namespace liquid {

template<>
class LindbladSet<SparseTag> {
public:
    explicit LindbladSet(std::vector<SparseOperator> ops) {
        if (ops.empty())
            throw std::invalid_argument("LindbladSet: must have at least one operator.");

        dim_ = ops[0].size();
        for (std::size_t k = 1; k < ops.size(); ++k)
            if (ops[k].size() != dim_)
                throw std::invalid_argument(
                    "LindbladSet: all operators must have the same dimension.");

        ops_ = std::move(ops);
        precompute_gamma();
    }

    LindbladSet(const LindbladSet&)            = delete;
    LindbladSet& operator=(const LindbladSet&) = delete;
    LindbladSet(LindbladSet&&)                 = default;
    LindbladSet& operator=(LindbladSet&&)      = default;

    // HOT PATH
    void apply_decay(const StateVector& psi, StateVector& out) const noexcept {
        constexpr Scalar neg_i_half{0.0, -0.5};
        gamma_.apply_add(psi, neg_i_half, out);
    }

    // COLD PATH
    std::size_t num_channels()  const noexcept { return ops_.size(); }
    Dim         hilbert_dim()   const noexcept { return dim_; }

    const SparseOperator& channel_op(std::size_t k) const noexcept {
        assert(k < ops_.size()); return ops_[k];
    }

    void apply_channel(std::size_t k,
                       const StateVector& psi,
                       StateVector& out) const {
        assert(k < ops_.size());
        out.set_zero();
        ops_[k].apply_add(psi, Scalar{1.0, 0.0}, out);
    }

    Real channel_probability(std::size_t k,
                              const StateVector& psi,
                              StateVector& scratch) const {
        apply_channel(k, psi, scratch);
        return scratch.norm_sq();
    }

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

    const SparseOperator& decay_operator() const noexcept { return gamma_; }

private:
    void precompute_gamma() {
        gamma_ = SparseOperator(dim_);
        for (const auto& L : ops_) {
            SparseOperator LdagL = sparse_adjoint_times(L);
            gamma_.add_scaled(LdagL, Scalar{1.0, 0.0});
        }
    }

    std::vector<SparseOperator> ops_;
    SparseOperator              gamma_;
    Dim                         dim_{0};
};

} // namespace liquid
