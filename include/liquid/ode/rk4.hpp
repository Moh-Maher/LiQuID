#pragma once

// liquid/ode/rk4.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Fixed-step 4th-order Runge-Kutta integrator.
//
// Integrates: d|ψ⟩/dt = f(t, |ψ⟩)
// where f is provided as a callable at call time (not stored).
//
// This is the Phase 1 reference solver. It is:
//   - Unconditionally stable for this equation class
//   - Simple to verify correctness
//   - 4th order accurate in dt
//   - Not adaptive (fixed stepsize)
//
// Interface contract (matches the Stepper policy concept):
//
//   StepResult try_step(t, psi_in, psi_out, dt, t_max, rhs_fn)
//   void reset(dt_initial)
//
// The 'try_step' name is used even for the fixed-step solver to maintain
// interface uniformity with the future adaptive solver (DormandPrince45).
// Fixed-step RK4 always accepts its step (never rejects).
//
// RHS callable signature:
//   rhs_fn(t, psi_in, accumulator)
//   Semantics: accumulator += f(t, psi_in)
//   The accumulator pattern avoids allocation in the inner loop.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/linalg/dense.hpp"
#include <algorithm>
#include <cassert>

namespace liquid::ode {

struct StepResult {
    bool  accepted;     // Always true for fixed-step RK4
    Real  dt_taken;     // Actual dt used (may be clamped by t_max)
    Real  dt_next;      // Suggested dt for next step (same as dt_taken for RK4)
    Real  local_err;    // Estimated local error (0.0 for fixed-step)
};

class RK4Stepper {
public:
    explicit RK4Stepper(Real dt_fixed) : dt_(dt_fixed) {
        assert(dt_fixed > 0.0);
    }

    // Default construction uses dt=1e-3; caller sets via reset()
    RK4Stepper() : dt_(1e-3) {}

    // ── Stepper policy interface ──────────────────────────────────────────────

    // Advance one RK4 step from (t, psi_in) to (t + dt, psi_out).
    //
    // rhs_fn: callable with signature
    //     void rhs_fn(Real t, const StateVector& psi, StateVector& accum)
    // Semantics: accum += f(t, psi)  (accumulates into accum, does not zero it)
    //
    // t_max: hard upper bound on t_new. If t + dt > t_max, the step is
    //        clamped to exactly t_max.
    //
    // psi_in and psi_out may NOT alias.
    //
    // Preconditions:
    //   psi_in.size() == psi_out.size()
    //   t_max >= t
    template<typename RHSFn>
    StepResult try_step(
        Real              t,
        const StateVector& psi_in,
        StateVector&       psi_out,
        Real               dt_suggested,
        Real               t_max,
        RHSFn&&            rhs_fn)
    {
        assert(psi_in.size() == psi_out.size());
        assert(t_max >= t);

        // Clamp stepsize to t_max
        const Real dt = std::min(dt_suggested, t_max - t);

        // For fixed-step RK4, if t is already at t_max, return zero step
        if (dt <= 0.0) {
            psi_out.copy_from(psi_in);
            return StepResult{true, 0.0, dt_, 0.0};
        }

        const Dim n = psi_in.size();
        ensure_scratch(n);

        // ── RK4 stages ────────────────────────────────────────────────────────
        //
        // k1 = f(t,          ψ)
        // k2 = f(t + dt/2,   ψ + dt/2 * k1)
        // k3 = f(t + dt/2,   ψ + dt/2 * k2)
        // k4 = f(t + dt,     ψ + dt   * k3)
        // ψ_new = ψ + (dt/6)(k1 + 2k2 + 2k3 + k4)
        //
        // Memory layout:
        //   k1_, k2_, k3_, k4_  — stage derivatives (pre-allocated in scratch)
        //   tmp_                — intermediate state

        const Real half_dt   = 0.5 * dt;
        const Scalar s_half  = Scalar{half_dt, 0.0};
        const Scalar s_full  = Scalar{dt, 0.0};

        // k1 = f(t, psi_in)
        k1_.set_zero();
        rhs_fn(t, psi_in, k1_);

        // tmp = psi_in + dt/2 * k1
        tmp_.copy_from(psi_in);
        tmp_.add_scaled(k1_, s_half);

        // k2 = f(t + dt/2, tmp)
        k2_.set_zero();
        rhs_fn(t + half_dt, tmp_, k2_);

        // tmp = psi_in + dt/2 * k2
        tmp_.copy_from(psi_in);
        tmp_.add_scaled(k2_, s_half);

        // k3 = f(t + dt/2, tmp)
        k3_.set_zero();
        rhs_fn(t + half_dt, tmp_, k3_);

        // tmp = psi_in + dt * k3
        tmp_.copy_from(psi_in);
        tmp_.add_scaled(k3_, s_full);

        // k4 = f(t + dt, tmp)
        k4_.set_zero();
        rhs_fn(t + dt, tmp_, k4_);

        // psi_out = psi_in + (dt/6)(k1 + 2k2 + 2k3 + k4)
        const Real sixth_dt   = dt / 6.0;
        const Real third_dt   = dt / 3.0;

        psi_out.copy_from(psi_in);
        psi_out.add_scaled(k1_, Scalar{sixth_dt, 0.0});
        psi_out.add_scaled(k2_, Scalar{third_dt, 0.0});
        psi_out.add_scaled(k3_, Scalar{third_dt, 0.0});
        psi_out.add_scaled(k4_, Scalar{sixth_dt, 0.0});

        return StepResult{true, dt, dt_, 0.0};
    }

    // Reset stepper state after a quantum jump.
    // For fixed-step RK4, this sets the suggested dt.
    void reset(Real dt_initial) noexcept {
        dt_ = dt_initial;
    }

    Real current_dt() const noexcept { return dt_; }

private:
    void ensure_scratch(Dim n) {
        // Allocate scratch vectors on first use or if dimension changes.
        // Dimension changes should never happen (OpenSystem is immutable),
        // but the check is cheap and guards against programming errors.
        if (k1_.size() != n) {
            k1_ = StateVector(n);
            k2_ = StateVector(n);
            k3_ = StateVector(n);
            k4_ = StateVector(n);
            tmp_ = StateVector(n);
        }
    }

    Real dt_;   // Fixed stepsize (or current suggestion)

    // Scratch vectors — reused across steps (no per-step allocation)
    // This makes RK4Stepper NOT thread-safe. Each thread owns its instance.
    StateVector k1_, k2_, k3_, k4_;
    StateVector tmp_;
};

} // namespace liquid::ode
