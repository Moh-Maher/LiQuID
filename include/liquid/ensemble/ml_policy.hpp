#pragma once

// liquid/ensemble/ml_policy.hpp
// ─────────────────────────────────────────────────────────────────────────────
// ML-assisted trajectory allocation policy.
//
// Architecture:
//   Input layer  : feature vector extracted from EnsembleSummary
//   Hidden layer 1: N_hidden neurons, sigmoid activation
//   Hidden layer 2: N_hidden neurons, sigmoid activation
//   Output layer : 1 neuron, sigmoid activation → allocation weight in (0,1)
//
// Training:
//   Online SGD with reward = -delta_rel_sem / delta_N
//   (SEM reduction per new trajectory — we want to maximise this)
//   Each completed batch provides one training example.
//
// Policy behaviour:
//   The output weight w ∈ (0,1) modulates the seed selection.
//   For Phase 6: the policy always spawns one new trajectory (same as uniform).
//   The ML contribution is in WHICH seed to use — currently a placeholder
//   for future rare-event biasing. The primary research value in Phase 6 is
//   the training loop and feature extraction infrastructure.
//
// Self-contained: zero external dependencies. All linear algebra is scalar
// operations on std::vector<Real>. The network is intentionally tiny (input
// dim ~10, hidden dim 8) — this is a proof-of-concept, not production ML.
//
// Integration point:
//   AllocatorPolicy policy = make_ml_policy(global_seed);
//   // Use exactly like make_uniform_policy — drop-in replacement.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/ensemble/adaptive_allocator.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <vector>

namespace liquid::ensemble::ml {

// ── Feature extraction ────────────────────────────────────────────────────────

// Extract a fixed-size feature vector from an EnsembleSummary.
// All features are normalised to approximately [0, 1].
//
// Feature index | Meaning
// 0             | log(1 + N_completed) / 10           (trajectory count)
// 1             | min(rel_sem, 1.0)                   (current convergence)
// 2             | N_failed / max(N_completed, 1)       (failure rate)
// 3             | wall_time / 60.0 (clamped to 1)     (time budget pressure)
// 4             | N_completed / 1000.0 (clamped to 1) (normalised count)

static constexpr std::size_t FEATURE_DIM = 5;

inline std::vector<Real> extract_features(const EnsembleSummary& s) {
    std::vector<Real> f(FEATURE_DIM, 0.0);

    const Real N = static_cast<Real>(s.trajectories_completed);

    f[0] = std::log(1.0 + N) / 10.0;
    f[1] = std::min(s.current_rel_sem, 1.0);
    f[2] = (N > 0.0)
           ? static_cast<Real>(s.trajectories_failed) / N
           : 0.0;
    f[3] = std::min(s.wall_time_elapsed / 60.0, 1.0);
    f[4] = std::min(N / 1000.0, 1.0);

    return f;
}

// ── Tiny feedforward network ──────────────────────────────────────────────────

class MLP {
public:
    MLP(std::size_t input_dim,
        std::size_t hidden_dim,
        Real        learning_rate = 0.01)
        : in_(input_dim)
        , hid_(hidden_dim)
        , lr_(learning_rate)
    {
        // Xavier initialisation: weights ~ Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
        // Use a simple deterministic initialisation for reproducibility.
        init_weights(W1_, b1_, input_dim,  hidden_dim, 42);
        init_weights(W2_, b2_, hidden_dim, hidden_dim, 137);
        init_weights(W3_, b3_, hidden_dim, 1,          271);
    }

    // Forward pass: returns scalar output in (0, 1)
    Real forward(const std::vector<Real>& x) const {
        assert(x.size() == in_);

        h1_ = linear_then_sigmoid(x,  W1_, b1_, in_,  hid_);
        h2_ = linear_then_sigmoid(h1_, W2_, b2_, hid_, hid_);

        // Output layer: single neuron
        Real z = b3_[0];
        for (std::size_t j = 0; j < hid_; ++j)
            z += W3_[j] * h2_[j];
        last_out_ = sigmoid(z);
        last_z3_  = z;
        return last_out_;
    }

    // Backward pass: update weights given scalar reward signal.
    // reward > 0: this output was good (increase it if x was the input).
    // reward < 0: this output was bad.
    // Uses simple SGD with the reward directly as gradient signal.
    void backward(const std::vector<Real>& x, Real reward) {
        // Re-run forward to populate activations (in case called independently)
        forward(x);

        // Output layer gradient
        const Real d_out = reward * sigmoid_prime(last_z3_);

        // Gradient for W3, b3
        for (std::size_t j = 0; j < hid_; ++j)
            W3_[j] += lr_ * d_out * h2_[j];
        b3_[0] += lr_ * d_out;

        // Backprop through h2
        std::vector<Real> d_h2(hid_, 0.0);
        for (std::size_t j = 0; j < hid_; ++j)
            d_h2[j] = d_out * W3_[j];

        // Gradient for W2, b2
        for (std::size_t i = 0; i < hid_; ++i) {
            const Real d = d_h2[i] * sigmoid_prime_from_act(h2_[i]);
            for (std::size_t j = 0; j < hid_; ++j)
                W2_[i * hid_ + j] += lr_ * d * h1_[j];
            b2_[i] += lr_ * d;
        }

        // Backprop through h1
        std::vector<Real> d_h1(hid_, 0.0);
        for (std::size_t j = 0; j < hid_; ++j) {
            for (std::size_t i = 0; i < hid_; ++i)
                d_h1[j] += d_h2[i] * sigmoid_prime_from_act(h2_[i]) * W2_[i * hid_ + j];
        }

        // Gradient for W1, b1
        for (std::size_t i = 0; i < hid_; ++i) {
            const Real d = d_h1[i] * sigmoid_prime_from_act(h1_[i]);
            for (std::size_t j = 0; j < in_; ++j)
                W1_[i * in_ + j] += lr_ * d * x[j];
            b1_[i] += lr_ * d;
        }
    }

    // Diagnostics
    std::size_t input_dim()  const noexcept { return in_; }
    std::size_t hidden_dim() const noexcept { return hid_; }
    Real        last_output()const noexcept { return last_out_; }

private:
    static Real sigmoid(Real z) noexcept {
        return 1.0 / (1.0 + std::exp(-z));
    }

    static Real sigmoid_prime(Real z) noexcept {
        const Real s = sigmoid(z);
        return s * (1.0 - s);
    }

    // sigmoid'(a) where a = sigmoid(z) — avoids recomputing z
    static Real sigmoid_prime_from_act(Real a) noexcept {
        return a * (1.0 - a);
    }

    static std::vector<Real> linear_then_sigmoid(
        const std::vector<Real>& x,
        const std::vector<Real>& W,
        const std::vector<Real>& b,
        std::size_t              fan_in,
        std::size_t              fan_out)
    {
        std::vector<Real> h(fan_out, 0.0);
        for (std::size_t i = 0; i < fan_out; ++i) {
            Real z = b[i];
            for (std::size_t j = 0; j < fan_in; ++j)
                z += W[i * fan_in + j] * x[j];
            h[i] = sigmoid(z);
        }
        return h;
    }

    // Deterministic pseudo-random initialisation from an integer seed.
    static void init_weights(std::vector<Real>& W,
                              std::vector<Real>& b,
                              std::size_t fan_in,
                              std::size_t fan_out,
                              std::uint64_t seed)
    {
        W.resize(fan_out * fan_in);
        b.resize(fan_out, 0.0);

        const Real scale = 1.0 / std::sqrt(static_cast<Real>(fan_in));

        // LCG for deterministic weight initialisation
        std::uint64_t state = seed;
        auto lcg = [&]() -> Real {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            // Map to [-scale, scale]
            const Real u = static_cast<Real>(state >> 11)
                         / static_cast<Real>(1ULL << 53);
            return (2.0 * u - 1.0) * scale;
        };

        for (auto& w : W) w = lcg();
        for (auto& bi : b) bi = lcg() * 0.1;
    }

    std::size_t in_, hid_;
    Real        lr_;

    // Weights: W[i*fan_in + j] = weight from neuron j to neuron i
    std::vector<Real> W1_, b1_;  // in_  → hid_
    std::vector<Real> W2_, b2_;  // hid_ → hid_
    std::vector<Real> W3_, b3_;  // hid_ → 1

    // Cached activations for backprop
    mutable std::vector<Real> h1_, h2_;
    mutable Real              last_out_{0.5};
    mutable Real              last_z3_{0.0};
};

// ── MLPolicy: stateful policy object ─────────────────────────────────────────

class MLPolicy {
public:
    explicit MLPolicy(Seed global_seed,
                      std::size_t hidden_dim    = 8,
                      Real        learning_rate = 0.01)
        : global_seed_(global_seed)
        , next_id_(0)
        , net_(FEATURE_DIM, hidden_dim, learning_rate)
        , prev_rel_sem_(1.0)
        , prev_N_(0)
        , calls_(0)
        , updates_(0)
    {}

    AllocationDecision operator()(const EnsembleSummary& summary) {
        ++calls_;

        // Extract features
        auto features = extract_features(summary);

        // Online training: compute reward from last step
        // Reward = improvement in rel_sem per new trajectory
        if (calls_ > 1 && summary.trajectories_completed > prev_N_) {
            const Real delta_N   = static_cast<Real>(
                summary.trajectories_completed - prev_N_);
            const Real delta_sem = prev_rel_sem_ - summary.current_rel_sem;
            // Positive reward when SEM decreases efficiently
            const Real reward = delta_sem / (delta_N + 1e-8);

            net_.backward(prev_features_, reward);
            ++updates_;
        }

        // Forward pass — output modulates seed perturbation
        const Real weight = net_.forward(features);

        // Store state for next training step
        prev_features_ = features;
        prev_rel_sem_  = summary.current_rel_sem;
        prev_N_        = summary.trajectories_completed;

        // Allocation decision: always spawn new (same as uniform)
        // Weight influences seed selection — perturbed for exploration
        const TrajId id = next_id_++;
        const Seed base = global_seed_ ^ (id * 0x9e3779b97f4a7c15ULL);
        // Use weight to select from two candidate seeds
        const Seed seed_a = base + 0x6c62272e07bb0142ULL;
        const Seed seed_b = base + 0xdeadbeefcafeULL;
        const Seed seed   = (weight > 0.5) ? seed_a : seed_b;

        return AllocationDecision{AllocAction::SpawnNew, seed, id};
    }

    // Diagnostics
    std::size_t calls()   const noexcept { return calls_; }
    std::size_t updates() const noexcept { return updates_; }
    Real        last_output() const noexcept { return net_.last_output(); }
    const MLP&  network() const noexcept { return net_; }

private:
    Seed             global_seed_;
    TrajId           next_id_;
    MLP              net_;
    Real             prev_rel_sem_;
    std::size_t      prev_N_;
    std::size_t      calls_;
    std::size_t      updates_;
    std::vector<Real> prev_features_;
};

// ── Factory function ──────────────────────────────────────────────────────────

// Creates an AllocatorPolicy backed by an online-learning MLP.
// The returned policy is stateful — the MLP trains as trajectories complete.
//
// Parameters:
//   global_seed   — same seed as EnsembleConfig::global_seed
//   hidden_dim    — neurons per hidden layer (default: 8)
//   learning_rate — SGD step size (default: 0.01)
//
// Usage:
//   auto policy = make_ml_policy(42);
//   SimulationBuilder{}.allocator_policy(std::move(policy))...build();

inline AllocatorPolicy make_ml_policy(
    Seed        global_seed,
    std::size_t hidden_dim    = 8,
    Real        learning_rate = 0.01)
{
    // MLPolicy is a stateful callable — wrap in shared_ptr so the
    // std::function copy semantics don't slice the state.
    auto policy_ptr = std::make_shared<MLPolicy>(
        global_seed, hidden_dim, learning_rate);

    return [policy_ptr](const EnsembleSummary& s) -> AllocationDecision {
        return (*policy_ptr)(s);
    };
}

} // namespace liquid::ensemble::ml
