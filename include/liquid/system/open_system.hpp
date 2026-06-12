#pragma once

// liquid/system/open_system.hpp
// ─────────────────────────────────────────────────────────────────────────────
// OpenSystem: the complete physics description of an open quantum system.
//
// Encapsulates:
//   H       — system Hamiltonian (time-independent, Phase 1)
//   {Lₖ}   — Lindblad jump operators (via LindbladSet)
//   H_eff   — precomputed effective non-Hermitian Hamiltonian
//
// Precomputed at construction:
//   H_eff = H - (i/2) Σₖ Lₖ†Lₖ
//
// Why precompute H_eff?
//   The MCWF algorithm evolves |ψ⟩ under H_eff at every ODE step.
//   For time-independent H, H_eff is constant — compute once, use forever.
//   Not precomputing it would be a correctness-preserving performance error.
//
// Access pattern contract:
//   apply_Heff()         — HOT PATH. Called O(N_steps) times per trajectory.
//   jump_probabilities() — COLD PATH. Called O(N_jumps) times per trajectory.
//   apply_jump()         — COLD PATH. Called O(N_jumps) times per trajectory.
//
// INVARIANT: OpenSystem is immutable after construction.
//            No method mutates any member.
//            Parameter sweeps construct new instances.
//
// Thread safety: OpenSystem is read-only after construction.
//                Multiple threads may call apply_Heff() concurrently
//                on the same instance without synchronization.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/linalg/dense.hpp"
#include "liquid/system/lindblad.hpp"
#include <cassert>
#include <stdexcept>
#include <vector>

namespace liquid {

// ─────────────────────────────────────────────────────────────────────────────
// OpenSystem<TimeIndependentTag, DenseTag> — Phase 1 concrete type
// ─────────────────────────────────────────────────────────────────────────────

template<typename TimeTag, typename StorageTag>
class OpenSystem;

template<>
class OpenSystem<TimeIndependentTag, DenseTag> {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // Construct from Hamiltonian and Lindblad operators.
    // Precomputes H_eff = H - (i/2) * Gamma immediately.
    //
    // Preconditions:
    //   H.size() == L.hilbert_dim()
    //   H and all Lₖ operators have the same dimension
    OpenSystem(DenseOperator H, LindbladSet<DenseTag> L)
        : H_(std::move(H))
        , L_(std::move(L))
        , H_eff_(H_.size())
    {
        if (H_.size() != L_.hilbert_dim()) {
            throw std::invalid_argument(
                "OpenSystem: Hamiltonian dimension (" +
                std::to_string(H_.size()) +
                ") does not match Lindblad operator dimension (" +
                std::to_string(L_.hilbert_dim()) + ").");
        }

        precompute_H_eff();
    }

    // No copy (large objects; prevents accidental copying in hot code)
    // Move is allowed
    OpenSystem(const OpenSystem&)            = delete;
    OpenSystem& operator=(const OpenSystem&) = delete;
    OpenSystem(OpenSystem&&)                 = default;
    OpenSystem& operator=(OpenSystem&&)      = default;

    // ── HOT PATH ──────────────────────────────────────────────────────────────

    // Compute: out = H_eff * psi
    //
    // H_eff = H - (i/2) Σₖ Lₖ†Lₖ  (precomputed, stored in H_eff_)
    //
    // This is the RHS of the non-Hermitian Schrödinger equation:
    //   d|ψ⟩/dt = -i H_eff |ψ⟩
    //
    // The ODE solver calls this function at every sub-step.
    // Preconditions: psi.size() == dim(), out.size() == dim()
    // Postcondition: out is overwritten (not accumulated)
    void apply_Heff(const StateVector& psi, StateVector& out) const noexcept {
        out.set_zero();
        // out += -i * H_eff * psi
        // The full non-Hermitian Schrödinger equation RHS is -i * H_eff * psi
        H_eff_.apply_add(psi, -i_unit, out);
    }

    // ── COLD PATH ─────────────────────────────────────────────────────────────

    // Compute per-channel jump probabilities at the moment of a jump.
    //
    // pₖ = ‖Lₖ|ψ⟩‖²  (unnormalized; proportional to jump rate for channel k)
    //
    // probs_out must have size >= num_channels()
    // scratch must have size == dim()
    // Returns sum Σₖ pₖ (useful for normalization check)
    Real jump_probabilities(const StateVector& psi,
                             StateVector& scratch,
                             std::vector<Real>& probs_out) const {
        return L_.all_probabilities(psi, scratch, probs_out);
    }

    // Apply jump for channel k and renormalize:
    //   |ψ⟩ → Lₖ|ψ⟩ / ‖Lₖ|ψ⟩‖
    //
    // Modifies psi in-place.
    // scratch is used as a temporary (size == dim())
    // Preconditions: k < num_channels(), ‖Lₖ|ψ⟩‖ > 0
    void apply_jump(std::size_t k,
                    StateVector& psi,
                    StateVector& scratch) const {
        L_.apply_channel(k, psi, scratch);
        scratch.normalize();
        psi.copy_from(scratch);
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    Dim         hilbert_dim()   const noexcept { return H_.size(); }
    std::size_t num_channels()  const noexcept { return L_.num_channels(); }

    const DenseOperator&     hamiltonian()    const noexcept { return H_; }
    const LindbladSet<DenseTag>& lindblad()  const noexcept { return L_; }
    const DenseOperator&     H_eff()         const noexcept { return H_eff_; }

private:
    void precompute_H_eff() {
        // H_eff = H - (i/2) * Gamma
        // Start with H_eff_ = H_
        H_eff_ = H_;

        // Add -(i/2) * Gamma = Gamma * (-i/2)
        // Scalar(-i/2) = Scalar{0, -0.5}
        constexpr Scalar neg_i_half{0.0, -0.5};
        H_eff_.add_scaled(L_.decay_operator(), neg_i_half);
    }

    DenseOperator         H_;      // Physical Hamiltonian (kept for diagnostics)
    LindbladSet<DenseTag> L_;      // Jump operators + precomputed Gamma
    DenseOperator         H_eff_;  // H - (i/2)*Gamma — precomputed, immutable
};

// ─────────────────────────────────────────────────────────────────────────────
// Convenience alias for Phase 1
// ─────────────────────────────────────────────────────────────────────────────

using DenseOpenSystem = OpenSystem<TimeIndependentTag, DenseTag>;

} // namespace liquid
