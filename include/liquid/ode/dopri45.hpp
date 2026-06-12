#pragma once

// liquid/ode/dopri45.hpp
// Dormand-Prince 4(5) adaptive Runge-Kutta integrator with PI stepsize control.
// Reference: Dormand & Prince (1980); Gustafsson PI controller (1991).

#include "liquid/core/types.hpp"
#include "liquid/linalg/dense.hpp"
#include "liquid/ode/rk4.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace liquid::ode {

// Config must be defined outside DormandPrince45 so it can be used
// as a default argument without triggering the C++ restriction on
// nested types with default member initializers.
struct DormandPrince45Config {
    Real atol        = 1e-8;
    Real rtol        = 1e-6;
    Real dt_initial  = 1e-3;
    Real dt_min      = 1e-12;
    Real dt_max      = 1e-1;
    Real safety      = 0.9;
    Real alpha       = 0.7 / 5.0;
    Real beta        = 0.4 / 5.0;
    int  max_rejects = 100;
};

class DormandPrince45 {
public:
    using Config = DormandPrince45Config;

    explicit DormandPrince45(Config cfg = Config{})
        : cfg_(cfg)
        , dt_(cfg.dt_initial)
        , err_prev_(1.0)
        , fsal_valid_(false)
        , rhs_count_(0)
        , reject_count_(0)
    {}

    template<typename RHSFn>
    StepResult try_step(
        Real               t,
        const StateVector& psi_in,
        StateVector&       psi_out,
        Real               dt_suggested,
        Real               t_max,
        RHSFn&&            rhs_fn)
    {
        assert(psi_in.size() == psi_out.size());
        assert(t_max >= t);

        const Dim n = psi_in.size();
        ensure_scratch(n);

        dt_ = std::min({dt_suggested, dt_, cfg_.dt_max, t_max - t});
        if (dt_ <= 0.0) {
            psi_out.copy_from(psi_in);
            return StepResult{true, 0.0, dt_, 0.0};
        }

        int rejects = 0;
        while (true) {
            const Real h = std::min(dt_, t_max - t);
            if (h <= 0.0) {
                psi_out.copy_from(psi_in);
                return StepResult{true, 0.0, dt_, 0.0};
            }

            compute_stages(t, h, psi_in, rhs_fn);
            const Real err = compute_error(psi_in, h);

            if (err <= 1.0) {
                psi_out.copy_from(psi5_);

                Real factor = cfg_.safety
                    * std::pow(1.0 / std::max(err, 1e-10), cfg_.alpha)
                    * std::pow(std::max(err_prev_, 1e-10), cfg_.beta);
                factor = std::min(std::max(factor, 0.2), 10.0);

                const Real dt_taken = h;
                dt_        = std::min(std::max(h * factor, cfg_.dt_min), cfg_.dt_max);
                err_prev_  = std::max(err, 1e-4);
                k1_.copy_from(k7_);
                fsal_valid_ = true;

                return StepResult{true, dt_taken, dt_, err};
            } else {
                ++rejects; ++reject_count_;
                Real factor = cfg_.safety
                    * std::pow(1.0 / std::max(err, 1e-10), cfg_.alpha);
                factor = std::max(factor, 0.1);
                dt_ = std::max(h * factor, cfg_.dt_min);
                if (rejects > cfg_.max_rejects || dt_ <= cfg_.dt_min) {
                    psi_out.copy_from(psi_in);
                    return StepResult{false, dt_, dt_, err};
                }
            }
        }
    }

    void reset(Real dt_initial) noexcept {
        dt_          = dt_initial;
        err_prev_    = 1.0;
        fsal_valid_  = false;
        reject_count_ = 0;
    }

    std::size_t rhs_count()    const noexcept { return rhs_count_; }
    std::size_t reject_count() const noexcept { return reject_count_; }
    Real        current_dt()   const noexcept { return dt_; }
    void reset_counters() noexcept { rhs_count_ = 0; reject_count_ = 0; }

private:
    // DOPRI Butcher tableau (Hairer, Nørsett, Wanner — Table 5.2)
    static constexpr Real c2=1./5., c3=3./10., c4=4./5., c5=8./9.;
    static constexpr Real a21=1./5.;
    static constexpr Real a31=3./40.,      a32=9./40.;
    static constexpr Real a41=44./45.,     a42=-56./15.,    a43=32./9.;
    static constexpr Real a51=19372./6561.,a52=-25360./2187.,
                          a53=64448./6561.,a54=-212./729.;
    static constexpr Real a61=9017./3168., a62=-355./33.,
                          a63=46732./5247.,a64=49./176.,
                          a65=-5103./18656.;
    static constexpr Real b1=35./384.,  b3=500./1113.,
                          b4=125./192., b5=-2187./6784., b6=11./84.;
    static constexpr Real e1= 71./57600.,  e3=-71./16695.,
                          e4= 71./1920.,   e5=-17253./339200.,
                          e6= 22./525.,    e7=-1./40.;

    template<typename RHSFn>
    void compute_stages(Real t, Real h, const StateVector& y0, RHSFn& rhs_fn) {
        if (!fsal_valid_) {
            k1_.set_zero(); rhs_fn(t, y0, k1_); ++rhs_count_;
        }
        auto stage = [&](Real tc, Real a1, Real a2, Real a3,
                          Real a4, Real a5, StateVector& ki, StateVector* k_extra=nullptr, Real ae=-1) {
            (void)k_extra; (void)ae;
            tmp_.copy_from(y0);
            if (a1!=0.0) tmp_.add_scaled(k1_, Scalar{h*a1,0});
            if (a2!=0.0) tmp_.add_scaled(k2_, Scalar{h*a2,0});
            if (a3!=0.0) tmp_.add_scaled(k3_, Scalar{h*a3,0});
            if (a4!=0.0) tmp_.add_scaled(k4_, Scalar{h*a4,0});
            if (a5!=0.0) tmp_.add_scaled(k5_, Scalar{h*a5,0});
            ki.set_zero(); rhs_fn(t+tc*h, tmp_, ki); ++rhs_count_;
        };
        stage(c2, a21, 0,   0,   0,   0,   k2_);
        stage(c3, a31, a32, 0,   0,   0,   k3_);
        stage(c4, a41, a42, a43, 0,   0,   k4_);
        stage(c5, a51, a52, a53, a54, 0,   k5_);

        tmp_.copy_from(y0);
        tmp_.add_scaled(k1_, Scalar{h*a61,0}); tmp_.add_scaled(k2_, Scalar{h*a62,0});
        tmp_.add_scaled(k3_, Scalar{h*a63,0}); tmp_.add_scaled(k4_, Scalar{h*a64,0});
        tmp_.add_scaled(k5_, Scalar{h*a65,0});
        k6_.set_zero(); rhs_fn(t+h, tmp_, k6_); ++rhs_count_;

        psi5_.copy_from(y0);
        psi5_.add_scaled(k1_, Scalar{h*b1,0}); psi5_.add_scaled(k3_, Scalar{h*b3,0});
        psi5_.add_scaled(k4_, Scalar{h*b4,0}); psi5_.add_scaled(k5_, Scalar{h*b5,0});
        psi5_.add_scaled(k6_, Scalar{h*b6,0});

        k7_.set_zero(); rhs_fn(t+h, psi5_, k7_); ++rhs_count_;
    }

    Real compute_error(const StateVector& y0, Real h) const {
        const Dim n = y0.size();
        Real sum_sq = 0.0;
        for (Dim i = 0; i < n; ++i) {
            const Scalar err_i = Scalar{h,0} * (
                Scalar{e1,0}*k1_[i] + Scalar{e3,0}*k3_[i] +
                Scalar{e4,0}*k4_[i] + Scalar{e5,0}*k5_[i] +
                Scalar{e6,0}*k6_[i] + Scalar{e7,0}*k7_[i]);
            const Real sc = cfg_.atol + cfg_.rtol *
                std::max(std::abs(y0[i]), std::abs(psi5_[i]));
            const Real s = std::abs(err_i) / sc;
            sum_sq += s * s;
        }
        return std::sqrt(sum_sq / static_cast<Real>(n));
    }

    void ensure_scratch(Dim n) {
        if (k1_.size() == n) return;
        k1_=k2_=k3_=k4_=k5_=k6_=k7_=StateVector(n);
        tmp_=psi5_=StateVector(n);
        fsal_valid_ = false;
    }

    Config      cfg_;
    Real        dt_, err_prev_;
    bool        fsal_valid_;
    std::size_t rhs_count_, reject_count_;
    StateVector k1_,k2_,k3_,k4_,k5_,k6_,k7_,tmp_,psi5_;
};

} // namespace liquid::ode
