#pragma once

// liquid/ensemble/running_statistics.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Online computation of mean, variance, and standard error for a vector
// of real-valued observables.
//
// Algorithm: Welford (1962) — numerically stable, O(1) per update.
//   Reference: B. P. Welford, "Note on a method for calculating corrected
//   sums of squares and products", Technometrics 4(3):419–420, 1962.
//
// Parallel merge: Chan, Golub, LeVeque (1979) — exact two-pass equivalent.
//   Reference: T. F. Chan et al., "Updating formulae and a pairwise algorithm
//   for computing sample variances", COMPSTAT 1979.
//
// Design:
//   - Vector-valued: one mean/variance/SEM per observable dimension.
//   - No dynamic allocation after construction.
//   - merge() enables lock-free parallel accumulation:
//       each thread owns a local instance → merge under mutex at end.
//   - All outputs lazily computed from M1_ and M2_ on request.
//   - Thread safety: NOT thread-safe. Each thread owns one instance.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace liquid::ensemble {

class RunningStatistics {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    explicit RunningStatistics(std::size_t dim)
        : dim_(dim)
        , n_(0)
        , mean_(dim, 0.0)
        , M2_(dim, 0.0)
        , var_cache_(dim, 0.0)
        , sem_cache_(dim, 0.0)
        , cache_valid_(false)
    {
        assert(dim > 0);
    }

    // ── Update ────────────────────────────────────────────────────────────────

    // Incorporate one new sample vector.
    // Precondition: sample.size() == dim()
    // Amortized O(dim) — no allocation.
    void update(const std::vector<Real>& sample) {
        assert(sample.size() == dim_);
        ++n_;
        cache_valid_ = false;

        // Welford's online algorithm:
        //   delta  = x - mean_{n-1}
        //   mean_n = mean_{n-1} + delta / n
        //   delta2 = x - mean_n
        //   M2_n   = M2_{n-1} + delta * delta2
        for (std::size_t i = 0; i < dim_; ++i) {
            const Real delta  = sample[i] - mean_[i];
            mean_[i] += delta / static_cast<Real>(n_);
            const Real delta2 = sample[i] - mean_[i];
            M2_[i]   += delta * delta2;
        }
    }

    // Single-observable overload (dim==1 convenience)
    void update(Real scalar_sample) {
        assert(dim_ == 1);
        tmp_scalar_[0] = scalar_sample;
        update(tmp_scalar_);
    }

    // ── Parallel merge ────────────────────────────────────────────────────────

    // Merge another RunningStatistics into this one.
    // After merge, *this contains the combined statistics of both.
    // Precondition: other.dim() == this->dim()
    //
    // Chan et al. parallel Welford:
    //   n_ab   = n_a + n_b
    //   delta  = mean_b - mean_a
    //   mean_ab = mean_a + delta * n_b / n_ab
    //   M2_ab  = M2_a + M2_b + delta^2 * n_a * n_b / n_ab
    void merge(const RunningStatistics& other) {
        assert(other.dim_ == dim_);
        if (other.n_ == 0) return;
        if (n_ == 0) { *this = other; return; }

        cache_valid_ = false;
        const std::size_t n_ab = n_ + other.n_;
        const Real inv_n_ab = 1.0 / static_cast<Real>(n_ab);

        for (std::size_t i = 0; i < dim_; ++i) {
            const Real delta = other.mean_[i] - mean_[i];
            mean_[i] += delta * static_cast<Real>(other.n_) * inv_n_ab;
            M2_[i]   += other.M2_[i]
                      + delta * delta
                        * static_cast<Real>(n_) * static_cast<Real>(other.n_)
                        * inv_n_ab;
        }
        n_ = n_ab;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    std::size_t count() const noexcept { return n_; }
    std::size_t dim()   const noexcept { return dim_; }

    // Mean of observable i. Valid for n >= 1.
    Real mean(std::size_t i) const noexcept {
        assert(i < dim_);
        return mean_[i];
    }

    // Unbiased sample variance of observable i. Valid for n >= 2.
    // Returns 0 for n < 2.
    Real variance(std::size_t i) const noexcept {
        assert(i < dim_);
        if (n_ < 2) return 0.0;
        return M2_[i] / static_cast<Real>(n_ - 1);
    }

    // Standard error of the mean for observable i. Valid for n >= 2.
    Real sem(std::size_t i) const noexcept {
        assert(i < dim_);
        if (n_ < 2) return std::numeric_limits<Real>::infinity();
        return std::sqrt(variance(i) / static_cast<Real>(n_));
    }

    // Relative SEM: sem(i) / |mean(i)|
    // Returns infinity if mean is zero.
    Real relative_sem(std::size_t i) const noexcept {
        assert(i < dim_);
        const Real m = std::abs(mean_[i]);
        if (m < 1e-300) return std::numeric_limits<Real>::infinity();
        return sem(i) / m;
    }

    // Worst-case relative SEM across all observables.
    // Primary convergence signal for ConvergenceMonitor.
    Real max_relative_sem() const noexcept {
        Real worst = 0.0;
        for (std::size_t i = 0; i < dim_; ++i) {
            const Real r = relative_sem(i);
            if (r > worst) worst = r;
        }
        return worst;
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    void reset() noexcept {
        n_ = 0;
        std::fill(mean_.begin(), mean_.end(), 0.0);
        std::fill(M2_.begin(),   M2_.end(),   0.0);
        cache_valid_ = false;
    }

    // Raw mean vector access (for output)
    const std::vector<Real>& mean_vec() const noexcept { return mean_; }

private:
    std::size_t       dim_;
    std::size_t       n_;
    std::vector<Real> mean_;        // M1: running mean
    std::vector<Real> M2_;          // sum of squared deviations from mean
    mutable std::vector<Real> var_cache_;
    mutable std::vector<Real> sem_cache_;
    mutable bool      cache_valid_;
    std::vector<Real> tmp_scalar_ = std::vector<Real>(1, 0.0);  // avoids alloc in scalar overload

    static constexpr Real inf_ = std::numeric_limits<Real>::infinity();
};

} // namespace liquid::ensemble
