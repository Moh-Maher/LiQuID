#pragma once

// liquid/system/open_system_ext.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Extended OpenSystem specializations:
//   - OpenSystem<TimeIndependentTag, SparseTag>  (Phase 4)
//   - OpenSystem<TimeDependentTag,   DenseTag>   (Phase 4)
//   - OpenSystem<TimeDependentTag,   SparseTag>  (Phase 4)
//
// All follow the same interface contract as the dense time-independent case:
//   apply_Heff()         — HOT PATH
//   jump_probabilities() — COLD PATH
//   apply_jump()         — COLD PATH
//
// Time-dependent Hamiltonian representation:
//   H(t) = H_static + Σⱼ fⱼ(t) · H_driveⱼ
//
//   DriveTerm<StorageTag> = { ScalarFn f; Operator H_drive; }
//   ScalarFn: std::function<Real(Real)> — envelope function fⱼ(t)
//
//   apply_Heff(t, psi, out) computes:
//     out = -i * [H_static + Σⱼ fⱼ(t)·H_driveⱼ - (i/2)·Γ] * psi
//         = -i·H_static·psi + (-i)·Σⱼ fⱼ(t)·H_driveⱼ·psi - (i/2)·Γ·psi
//   where Γ is precomputed at construction.
//   Only the scalar f(t) evaluation and scaling are done at runtime.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/linalg/dense.hpp"
#include "liquid/linalg/sparse.hpp"
#include "liquid/system/lindblad.hpp"
#include "liquid/system/lindblad_sparse.hpp"
#include "liquid/system/open_system.hpp"
#include <functional>
#include <stdexcept>
#include <vector>

namespace liquid {

// ── DriveTerm: one term in a time-dependent Hamiltonian ──────────────────────

template<typename StorageTag>
struct DriveTerm;

template<>
struct DriveTerm<DenseTag> {
    std::function<Real(Real)> envelope;  // fⱼ(t)
    DenseOperator             H_drive;   // H_driveⱼ
};

template<>
struct DriveTerm<SparseTag> {
    std::function<Real(Real)> envelope;
    SparseOperator            H_drive;
};

// ── Helper: select operator type from tag ────────────────────────────────────

template<typename StorageTag> struct OperatorType;
template<> struct OperatorType<DenseTag>  { using type = DenseOperator; };
template<> struct OperatorType<SparseTag> { using type = SparseOperator; };

// ─────────────────────────────────────────────────────────────────────────────
// OpenSystem<TimeIndependentTag, SparseTag>
// Sparse time-independent: same design as dense, sparse storage.
// ─────────────────────────────────────────────────────────────────────────────

template<>
class OpenSystem<TimeIndependentTag, SparseTag> {
public:
    OpenSystem(SparseOperator H, LindbladSet<SparseTag> L)
        : H_(std::move(H))
        , L_(std::move(L))
        , H_eff_(H_.size())
    {
        if (H_.size() != L_.hilbert_dim())
            throw std::invalid_argument(
                "OpenSystem: Hamiltonian and Lindblad dimension mismatch.");
        precompute_H_eff();
    }

    OpenSystem(const OpenSystem&)            = delete;
    OpenSystem& operator=(const OpenSystem&) = delete;
    OpenSystem(OpenSystem&&)                 = default;
    OpenSystem& operator=(OpenSystem&&)      = default;

    // HOT PATH
    void apply_Heff(const StateVector& psi, StateVector& out) const noexcept {
        out.set_zero();
        H_eff_.apply_add(psi, -i_unit, out);
    }

    // Overload accepting t (ignored for time-independent systems).
    void apply_Heff(Real /*t*/,
                    const StateVector& psi,
                    StateVector& out) const noexcept {
        apply_Heff(psi, out);
    }

    // COLD PATH
    Real jump_probabilities(const StateVector& psi,
                             StateVector& scratch,
                             std::vector<Real>& probs_out) const {
        return L_.all_probabilities(psi, scratch, probs_out);
    }

    void apply_jump(std::size_t k, StateVector& psi, StateVector& scratch) const {
        L_.apply_channel(k, psi, scratch);
        scratch.normalize();
        psi.copy_from(scratch);
    }

    Dim         hilbert_dim()  const noexcept { return H_.size(); }
    std::size_t num_channels() const noexcept { return L_.num_channels(); }

    const SparseOperator&       H_eff()    const noexcept { return H_eff_; }
    const SparseOperator&       hamiltonian() const noexcept { return H_; }
    const LindbladSet<SparseTag>& lindblad() const noexcept { return L_; }

private:
    void precompute_H_eff() {
        H_eff_ = H_;
        constexpr Scalar neg_i_half{0.0, -0.5};
        H_eff_.add_scaled(L_.decay_operator(), neg_i_half);
    }

    SparseOperator         H_;
    LindbladSet<SparseTag> L_;
    SparseOperator         H_eff_;
};

// ─────────────────────────────────────────────────────────────────────────────
// OpenSystem<TimeDependentTag, StorageTag>
// Template for both dense and sparse time-dependent systems.
// ─────────────────────────────────────────────────────────────────────────────

template<typename StorageTag>
class OpenSystem<TimeDependentTag, StorageTag> {
public:
    using Op = typename OperatorType<StorageTag>::type;

    // H(t) = H_static + Σⱼ fⱼ(t)·H_driveⱼ
    OpenSystem(Op                                 H_static,
               std::vector<DriveTerm<StorageTag>> drive_terms,
               LindbladSet<StorageTag>            L)
        : H_static_(std::move(H_static))
        , drive_terms_(std::move(drive_terms))
        , L_(std::move(L))
        , scratch_drive_(H_static_.size())
    {
        if (H_static_.size() != L_.hilbert_dim())
            throw std::invalid_argument(
                "OpenSystem: Hamiltonian and Lindblad dimension mismatch.");
    }

    OpenSystem(const OpenSystem&)            = delete;
    OpenSystem& operator=(const OpenSystem&) = delete;
    OpenSystem(OpenSystem&&)                 = default;
    OpenSystem& operator=(OpenSystem&&)      = default;

    // HOT PATH — time-dependent version takes t explicitly
    // out = -i·[H(t) - (i/2)·Γ]·psi
    //      = -i·H_static·psi - i·Σⱼ fⱼ(t)·H_driveⱼ·psi - (1/2)·Γ·psi
    //
    // NOTE: The decay contribution is -(1/2)·Γ·psi (real, negative).
    // L_.apply_decay() adds -(i/2)·Γ·psi, so we must NOT use it here.
    // Instead we apply -(1/2)·Γ directly.
    void apply_Heff(Real t, const StateVector& psi, StateVector& out) const noexcept {
        out.set_zero();
        // Static Hamiltonian term: -i·H_static·psi
        H_static_.apply_add(psi, -i_unit, out);
        // Drive terms: -i·fⱼ(t)·H_driveⱼ·psi
        for (const auto& term : drive_terms_) {
            const Real ft = term.envelope(t);
            if (ft == 0.0) continue;
            term.H_drive.apply_add(psi, Scalar{0.0, -ft}, out);
        }
        // Decay term: -i·(-(i/2))·Γ·psi = -(1/2)·Γ·psi
        // Apply_decay adds -(i/2)*Γ*psi, but we need -i times that = -(1/2)*Γ*psi
        // So apply Γ with alpha = -0.5 (real)
        L_.decay_operator().apply_add(psi, Scalar{-0.5, 0.0}, out);
    }

    // Overload without t for interface compatibility with MCWFPropagator.
    // The propagator RHS lambda passes t — this is handled by the lambda
    // in MCWFPropagator. For time-independent systems apply_Heff(psi,out)
    // is used; for time-dependent the lambda captures the time.
    // This overload is provided for any code that treats both uniformly.
    void apply_Heff(const StateVector& psi, StateVector& out) const noexcept {
        apply_Heff(0.0, psi, out);
    }

    // COLD PATH
    Real jump_probabilities(const StateVector& psi,
                             StateVector& scratch,
                             std::vector<Real>& probs_out) const {
        return L_.all_probabilities(psi, scratch, probs_out);
    }

    void apply_jump(std::size_t k, StateVector& psi, StateVector& scratch) const {
        L_.apply_channel(k, psi, scratch);
        scratch.normalize();
        psi.copy_from(scratch);
    }

    Dim         hilbert_dim()  const noexcept { return H_static_.size(); }
    std::size_t num_channels() const noexcept { return L_.num_channels(); }

private:
    Op                                H_static_;
    std::vector<DriveTerm<StorageTag>> drive_terms_;
    LindbladSet<StorageTag>           L_;
    mutable StateVector               scratch_drive_;
};

// ── Convenience aliases ───────────────────────────────────────────────────────

using SparseOpenSystem    = OpenSystem<TimeIndependentTag, SparseTag>;
using DenseTDOpenSystem   = OpenSystem<TimeDependentTag,   DenseTag>;
using SparseTDOpenSystem  = OpenSystem<TimeDependentTag,   SparseTag>;

} // namespace liquid
