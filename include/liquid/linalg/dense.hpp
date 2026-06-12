#pragma once

// liquid/linalg/dense.hpp
// ─────────────────────────────────────────────────────────────────────────────
// StateVector and DenseOperator: the core linear algebra objects for Phase 1.
//
// Implementation note:
//   This is a direct implementation using std::vector<Scalar>.
//   It does NOT depend on Eigen. When Eigen becomes available, these types
//   will be reimplemented as thin wrappers around Eigen::VectorXcd and
//   Eigen::MatrixXcd. The interface below is the stable contract;
//   the storage is an implementation detail.
//
// Design decisions:
//   - Row-major storage for DenseOperator (matches C memory order, cache-
//     friendly for row-wise access patterns in apply()).
//   - No expression templates. Operations are eager. This is correct for
//     our access patterns: we always need the result immediately.
//   - No BLAS dependency in Phase 1. Manual loops are sufficient for
//     N ≤ 100 (Phase 1 target). BLAS integration is a Phase 3 concern.
//   - All operations that could allocate are clearly marked.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace liquid {

// ─────────────────────────────────────────────────────────────────────────────
// StateVector: a heap-allocated complex vector of dimension N.
// Represents a quantum state |ψ⟩ ∈ ℂ^N.
// ─────────────────────────────────────────────────────────────────────────────

class StateVector {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    StateVector() = default;

    explicit StateVector(Dim n, Scalar fill = Scalar{0.0, 0.0})
        : data_(n, fill), dim_(n) {}

    // Copy and move — standard value semantics
    StateVector(const StateVector&)            = default;
    StateVector& operator=(const StateVector&) = default;
    StateVector(StateVector&&)                 = default;
    StateVector& operator=(StateVector&&)      = default;

    // ── Element access ────────────────────────────────────────────────────────

    Scalar& operator[](Idx i) noexcept {
        assert(i < dim_);
        return data_[i];
    }

    const Scalar& operator[](Idx i) const noexcept {
        assert(i < dim_);
        return data_[i];
    }

    // ── Dimensions ────────────────────────────────────────────────────────────

    Dim size() const noexcept { return dim_; }
    bool empty() const noexcept { return dim_ == 0; }

    // ── Raw data access ───────────────────────────────────────────────────────

    Scalar*       data()       noexcept { return data_.data(); }
    const Scalar* data() const noexcept { return data_.data(); }

    // ── Mathematical operations ───────────────────────────────────────────────

    // Squared L2 norm: ⟨ψ|ψ⟩ = Σᵢ |ψᵢ|²
    // Returns Real (always non-negative by construction)
    Real norm_sq() const noexcept {
        Real acc = 0.0;
        for (Idx i = 0; i < dim_; ++i) {
            acc += std::norm(data_[i]);  // std::norm = |z|²
        }
        return acc;
    }

    // L2 norm: ‖|ψ⟩‖ = √⟨ψ|ψ⟩
    Real norm() const noexcept {
        return std::sqrt(norm_sq());
    }

    // In-place normalization: |ψ⟩ → |ψ⟩ / ‖|ψ⟩‖
    // Precondition: norm() > 0 (checked in debug mode)
    void normalize() {
        const Real n = norm();
        assert(n > 0.0);
        const Real inv_n = 1.0 / n;
        for (Idx i = 0; i < dim_; ++i) {
            data_[i] *= inv_n;
        }
    }

    // Scale in-place: |ψ⟩ → α|ψ⟩
    void scale(Scalar alpha) noexcept {
        for (Idx i = 0; i < dim_; ++i) {
            data_[i] *= alpha;
        }
    }

    // In-place addition: |ψ⟩ += α|φ⟩
    // Precondition: other.size() == this->size()
    void add_scaled(const StateVector& other, Scalar alpha) noexcept {
        assert(other.dim_ == dim_);
        for (Idx i = 0; i < dim_; ++i) {
            data_[i] += alpha * other.data_[i];
        }
    }

    // Zero all elements
    void set_zero() noexcept {
        for (auto& x : data_) x = Scalar{0.0, 0.0};
    }

    // Copy from another vector (must be same dimension)
    void copy_from(const StateVector& other) noexcept {
        assert(other.dim_ == dim_);
        data_ = other.data_;
    }

    // Inner product ⟨φ|ψ⟩ = Σᵢ conj(φᵢ) ψᵢ
    static Scalar inner(const StateVector& phi, const StateVector& psi) noexcept {
        assert(phi.dim_ == psi.dim_);
        Scalar acc{0.0, 0.0};
        for (Idx i = 0; i < phi.dim_; ++i) {
            acc += std::conj(phi.data_[i]) * psi.data_[i];
        }
        return acc;
    }

private:
    std::vector<Scalar> data_;
    Dim                 dim_{0};
};


// ─────────────────────────────────────────────────────────────────────────────
// DenseOperator: an N×N complex matrix.
// Represents a quantum operator O ∈ ℂ^{N×N}.
//
// Storage: row-major (row i, col j → data_[i * dim_ + j]).
// ─────────────────────────────────────────────────────────────────────────────

class DenseOperator {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    DenseOperator() = default;

    explicit DenseOperator(Dim n, Scalar fill = Scalar{0.0, 0.0})
        : data_(n * n, fill), dim_(n) {}

    // Construct from flat row-major data
    DenseOperator(Dim n, std::vector<Scalar> data)
        : data_(std::move(data)), dim_(n) {
        assert(data_.size() == n * n);
    }

    // Copy and move — standard value semantics
    DenseOperator(const DenseOperator&)            = default;
    DenseOperator& operator=(const DenseOperator&) = default;
    DenseOperator(DenseOperator&&)                 = default;
    DenseOperator& operator=(DenseOperator&&)      = default;

    // ── Element access (row i, col j) ─────────────────────────────────────────

    Scalar& operator()(Idx i, Idx j) noexcept {
        assert(i < dim_ && j < dim_);
        return data_[i * dim_ + j];
    }

    const Scalar& operator()(Idx i, Idx j) const noexcept {
        assert(i < dim_ && j < dim_);
        return data_[i * dim_ + j];
    }

    // ── Dimensions ────────────────────────────────────────────────────────────

    Dim size() const noexcept { return dim_; }

    // ── Mathematical operations ───────────────────────────────────────────────

    // Matrix-vector product: out = A * psi
    // ALLOCATES: creates output vector.
    // For hot-path use, prefer apply_add() below.
    StateVector apply(const StateVector& psi) const {
        assert(psi.size() == dim_);
        StateVector out(dim_);
        apply_add(psi, Scalar{1.0, 0.0}, out);
        return out;
    }

    // out += alpha * A * psi  (no allocation)
    // This is the hot-path interface used by ODE solvers.
    // Preconditions: psi.size() == dim_, out.size() == dim_
    void apply_add(const StateVector& psi,
                   Scalar alpha,
                   StateVector& out) const noexcept {
        assert(psi.size() == dim_);
        assert(out.size() == dim_);
        for (Idx i = 0; i < dim_; ++i) {
            Scalar acc{0.0, 0.0};
            for (Idx j = 0; j < dim_; ++j) {
                acc += data_[i * dim_ + j] * psi[j];
            }
            out[i] += alpha * acc;
        }
    }

    // Hermitian conjugate: (A†)ᵢⱼ = conj(Aⱼᵢ)
    // ALLOCATES.
    DenseOperator adjoint() const {
        DenseOperator result(dim_);
        for (Idx i = 0; i < dim_; ++i) {
            for (Idx j = 0; j < dim_; ++j) {
                result(i, j) = std::conj((*this)(j, i));
            }
        }
        return result;
    }

    // In-place addition: A += alpha * B
    // Precondition: B.size() == this->size()
    void add_scaled(const DenseOperator& B, Scalar alpha) noexcept {
        assert(B.dim_ == dim_);
        for (std::size_t k = 0; k < data_.size(); ++k) {
            data_[k] += alpha * B.data_[k];
        }
    }

    // Matrix-matrix product: C = A * B
    // ALLOCATES. Used for precomputation, not hot path.
    DenseOperator matmul(const DenseOperator& B) const {
        assert(B.dim_ == dim_);
        DenseOperator C(dim_);
        for (Idx i = 0; i < dim_; ++i) {
            for (Idx k = 0; k < dim_; ++k) {
                const Scalar aik = (*this)(i, k);
                if (aik == Scalar{0.0, 0.0}) continue;  // sparse shortcut
                for (Idx j = 0; j < dim_; ++j) {
                    C(i, j) += aik * B(k, j);
                }
            }
        }
        return C;
    }

    // Zero all elements
    void set_zero() noexcept {
        for (auto& x : data_) x = Scalar{0.0, 0.0};
    }

    // Raw data access
    Scalar*       data()       noexcept { return data_.data(); }
    const Scalar* data() const noexcept { return data_.data(); }

private:
    std::vector<Scalar> data_;
    Dim                 dim_{0};
};

// ─────────────────────────────────────────────────────────────────────────────
// Free functions — operator construction helpers
// ─────────────────────────────────────────────────────────────────────────────

// Create N×N identity matrix
inline DenseOperator make_identity(Dim n) {
    DenseOperator I(n);
    for (Idx i = 0; i < n; ++i) I(i, i) = Scalar{1.0, 0.0};
    return I;
}

// Create N×N zero matrix
inline DenseOperator make_zero_operator(Dim n) {
    return DenseOperator(n);
}

// Compute A†A (used for precomputing Lₖ†Lₖ in LindbladSet)
// ALLOCATES.
inline DenseOperator adjoint_times(const DenseOperator& A) {
    return A.adjoint().matmul(A);
}

// Compute expectation value ⟨ψ|O|ψ⟩
// Returns the full complex value (caller takes real part if needed)
inline Scalar expectation(const DenseOperator& O, const StateVector& psi) {
    assert(O.size() == psi.size());
    StateVector Opsi(psi.size());
    O.apply_add(psi, Scalar{1.0, 0.0}, Opsi);
    return StateVector::inner(psi, Opsi);
}

} // namespace liquid
