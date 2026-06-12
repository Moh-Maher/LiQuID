#pragma once

// liquid/linalg/sparse.hpp
// ─────────────────────────────────────────────────────────────────────────────
// SparseOperator: Compressed Sparse Row (CSR) complex sparse matrix.
//
// Storage layout:
//   values_[k]      — complex value of the k-th stored element
//   col_indices_[k] — column index of the k-th stored element
//   row_ptr_[i]     — index into values_/col_indices_ where row i begins
//   row_ptr_[dim_]  — total number of stored elements (nnz)
//
// This is the standard CSR format. Row i contains elements:
//   values_[row_ptr_[i] .. row_ptr_[i+1]-1]
//   at columns col_indices_[row_ptr_[i] .. row_ptr_[i+1]-1]
//
// Design decisions (frozen):
//   - Square matrices only (quantum operators are always N×N)
//   - Row-major access is O(1) per row (good for matvec)
//   - No dynamic insertion after construction
//   - Construction from triplet list (COO format) then sorted/compressed
//   - apply_add() is the hot-path interface — no allocation
//   - adjoint() allocates a new SparseOperator (cold path)
//
// Thread safety: read-only after construction — safe for concurrent matvec.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace liquid {

// ── Triplet: (row, col, value) for construction ───────────────────────────────

struct Triplet {
    Idx    row;
    Idx    col;
    Scalar value;
};

// ─────────────────────────────────────────────────────────────────────────────
// SparseOperator
// ─────────────────────────────────────────────────────────────────────────────

class SparseOperator {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    SparseOperator() = default;

    // Construct from a list of (row, col, value) triplets.
    // Duplicate (row,col) entries are summed.
    // Precondition: all row, col indices < dim.
    SparseOperator(Dim dim, std::vector<Triplet> triplets)
        : dim_(dim)
    {
        build_from_triplets(std::move(triplets));
    }

    // Construct an N×N zero matrix (no stored elements)
    explicit SparseOperator(Dim dim)
        : dim_(dim)
        , row_ptr_(dim + 1, 0)
    {}

    // Copy and move
    SparseOperator(const SparseOperator&)            = default;
    SparseOperator& operator=(const SparseOperator&) = default;
    SparseOperator(SparseOperator&&)                 = default;
    SparseOperator& operator=(SparseOperator&&)      = default;

    // ── Dimensions ────────────────────────────────────────────────────────────

    Dim         size() const noexcept { return dim_; }
    std::size_t nnz()  const noexcept { return values_.size(); }
    bool        empty()const noexcept { return dim_ == 0; }

    // Sparsity: fraction of non-zero entries
    double sparsity() const noexcept {
        if (dim_ == 0) return 0.0;
        return static_cast<double>(nnz()) / static_cast<double>(dim_ * dim_);
    }

    // ── HOT PATH: matrix-vector product ──────────────────────────────────────

    // out += alpha * A * psi  (accumulates, no allocation)
    // Preconditions: psi.size() == dim_, out.size() == dim_
    void apply_add(const StateVector& psi,
                   Scalar             alpha,
                   StateVector&       out) const noexcept {
        assert(psi.size() == dim_);
        assert(out.size() == dim_);

        for (Idx i = 0; i < dim_; ++i) {
            Scalar acc{0.0, 0.0};
            const Idx start = row_ptr_[i];
            const Idx end   = row_ptr_[i + 1];
            for (Idx k = start; k < end; ++k) {
                acc += values_[k] * psi[col_indices_[k]];
            }
            out[i] += alpha * acc;
        }
    }

    // out = A * psi  (allocates output vector)
    StateVector apply(const StateVector& psi) const {
        assert(psi.size() == dim_);
        StateVector out(dim_);
        apply_add(psi, Scalar{1.0, 0.0}, out);
        return out;
    }

    // ── COLD PATH: algebraic operations ──────────────────────────────────────

    // Hermitian conjugate: (A†)ᵢⱼ = conj(Aⱼᵢ)
    // Allocates a new SparseOperator.
    SparseOperator adjoint() const {
        // Transpose + conjugate: for each stored (i,j,v), add triplet (j,i,conj(v))
        std::vector<Triplet> trips;
        trips.reserve(nnz());
        for (Idx i = 0; i < dim_; ++i) {
            for (Idx k = row_ptr_[i]; k < row_ptr_[i+1]; ++k) {
                trips.push_back({col_indices_[k], i, std::conj(values_[k])});
            }
        }
        return SparseOperator(dim_, std::move(trips));
    }

    // In-place addition: A += alpha * B
    // Precondition: B.size() == this->size()
    // NOTE: this rebuilds the CSR structure — use only for precomputation.
    void add_scaled(const SparseOperator& B, Scalar alpha) {
        assert(B.dim_ == dim_);
        // Collect all triplets from this + alpha*B, then rebuild
        std::vector<Triplet> trips;
        trips.reserve(nnz() + B.nnz());

        for (Idx i = 0; i < dim_; ++i) {
            for (Idx k = row_ptr_[i]; k < row_ptr_[i+1]; ++k)
                trips.push_back({i, col_indices_[k], values_[k]});
        }
        for (Idx i = 0; i < dim_; ++i) {
            for (Idx k = B.row_ptr_[i]; k < B.row_ptr_[i+1]; ++k)
                trips.push_back({i, B.col_indices_[k], alpha * B.values_[k]});
        }
        build_from_triplets(std::move(trips));
    }

    // Sparse matrix-matrix product: C = A * B
    // Returns a new SparseOperator. Used for precomputation (cold path).
    SparseOperator matmul(const SparseOperator& B) const {
        assert(B.dim_ == dim_);
        std::vector<Triplet> trips;

        for (Idx i = 0; i < dim_; ++i) {
            for (Idx ka = row_ptr_[i]; ka < row_ptr_[i+1]; ++ka) {
                const Idx  a_col = col_indices_[ka];
                const Scalar a_val = values_[ka];
                for (Idx kb = B.row_ptr_[a_col]; kb < B.row_ptr_[a_col+1]; ++kb) {
                    trips.push_back({i, B.col_indices_[kb], a_val * B.values_[kb]});
                }
            }
        }
        return SparseOperator(dim_, std::move(trips));
    }

    // Element access (for diagnostics/testing — not for hot path)
    Scalar operator()(Idx row, Idx col) const noexcept {
        assert(row < dim_ && col < dim_);
        for (Idx k = row_ptr_[row]; k < row_ptr_[row+1]; ++k) {
            if (col_indices_[k] == col) return values_[k];
        }
        return Scalar{0.0, 0.0};
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    // Export as dense operator (for testing and small-system use)
    // Allocates an N×N dense matrix.
    DenseOperator to_dense() const;  // defined after DenseOperator include

private:
    void build_from_triplets(std::vector<Triplet> trips) {
        // Sort by (row, col)
        std::sort(trips.begin(), trips.end(), [](const Triplet& a, const Triplet& b) {
            return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
        });

        // Validate indices and sum duplicates
        std::vector<Triplet> merged;
        merged.reserve(trips.size());
        for (auto& t : trips) {
            if (t.row >= dim_ || t.col >= dim_) {
                throw std::out_of_range(
                    "SparseOperator: triplet index out of range");
            }
            if (!merged.empty() &&
                merged.back().row == t.row &&
                merged.back().col == t.col) {
                merged.back().value += t.value;  // sum duplicates
            } else {
                merged.push_back(t);
            }
        }

        // Remove numerical zeros (drop elements below threshold)
        constexpr double zero_tol = 1e-300;
        merged.erase(
            std::remove_if(merged.begin(), merged.end(),
                [](const Triplet& t) { return std::abs(t.value) < zero_tol; }),
            merged.end());

        // Build CSR arrays
        const std::size_t nnz = merged.size();
        values_.resize(nnz);
        col_indices_.resize(nnz);
        row_ptr_.assign(dim_ + 1, 0);

        // Count entries per row
        for (auto& t : merged) ++row_ptr_[t.row + 1];

        // Prefix sum
        for (Dim i = 0; i < dim_; ++i)
            row_ptr_[i + 1] += row_ptr_[i];

        // Fill values and column indices
        std::vector<Idx> cursor(row_ptr_.begin(), row_ptr_.end() - 1);
        for (auto& t : merged) {
            const Idx pos = cursor[t.row]++;
            values_[pos]      = t.value;
            col_indices_[pos] = t.col;
        }
    }

    Dim              dim_{0};
    std::vector<Scalar> values_;      // non-zero values
    std::vector<Idx>    col_indices_; // column of each value
    std::vector<Idx>    row_ptr_;     // row_ptr_[i]: start of row i in values_
};

// ── Free functions ────────────────────────────────────────────────────────────

// Create sparse identity matrix
inline SparseOperator make_sparse_identity(Dim n) {
    std::vector<Triplet> trips(n);
    for (Idx i = 0; i < n; ++i)
        trips[i] = {i, i, Scalar{1.0, 0.0}};
    return SparseOperator(n, std::move(trips));
}

// Compute A†A for a sparse operator (used for precomputing Lₖ†Lₖ)
inline SparseOperator sparse_adjoint_times(const SparseOperator& A) {
    return A.adjoint().matmul(A);
}

// Expectation value ⟨ψ|O|ψ⟩ for sparse operator
inline Scalar sparse_expectation(const SparseOperator& O,
                                  const StateVector& psi) {
    assert(O.size() == psi.size());
    StateVector Opsi(psi.size());
    O.apply_add(psi, Scalar{1.0, 0.0}, Opsi);
    return StateVector::inner(psi, Opsi);
}

// Convert DenseOperator to SparseOperator
inline SparseOperator to_sparse(const DenseOperator& A,
                                  double zero_tol = 1e-14) {
    const Dim n = A.size();
    std::vector<Triplet> trips;
    for (Idx i = 0; i < n; ++i)
        for (Idx j = 0; j < n; ++j)
            if (std::abs(A(i,j)) > zero_tol)
                trips.push_back({i, j, A(i,j)});
    return SparseOperator(n, std::move(trips));
}

// SparseOperator::to_dense() implementation (needs DenseOperator)
inline DenseOperator SparseOperator::to_dense() const {
    DenseOperator D(dim_);
    for (Idx i = 0; i < dim_; ++i)
        for (Idx k = row_ptr_[i]; k < row_ptr_[i+1]; ++k)
            D(i, col_indices_[k]) = values_[k];
    return D;
}

} // namespace liquid
