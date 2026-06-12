#pragma once

// liquid/core/rng.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Trajectory-local reproducible random number generation.
//
// Algorithm: xoshiro256** by Blackman and Vigna (2018).
//   - 256-bit state (32 bytes)
//   - Period: 2^256 - 1
//   - Passes all PractRand and TestU01 tests
//   - ~4× faster than std::mt19937_64 with 78× smaller state
//   - No dynamic allocation
//
// INVARIANT: Each trajectory owns exactly one RNGState.
//            No global or thread-local RNG is used anywhere in LiQuID.
//            This guarantees reproducibility regardless of thread scheduling.
//
// Seeding protocol:
//   seed(global_seed, traj_id) uses splitmix64 to derive independent streams.
//   Different (global_seed, traj_id) pairs are guaranteed to produce
//   statistically independent sequences.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include <array>
#include <cstdint>
#include <iosfwd>
#include <limits>

namespace liquid {

class RNGState {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    RNGState() noexcept = default;

    // Primary seeding interface.
    // Combines global_seed and traj_id via splitmix64 to produce an
    // independent stream for each trajectory.
    // Precondition: none (any values are valid)
    void seed(Seed global_seed, TrajId traj_id) noexcept;

    // ── Random number generation ──────────────────────────────────────────────

    // Raw 64-bit output
    std::uint64_t next_uint64() noexcept;

    // Uniform draw in (0, 1].
    // NOTE: returns in (0, 1], not [0, 1). Zero is excluded.
    // This is intentional: r is used as a jump threshold where r=0
    // would cause an immediate jump at t=0, which is physically wrong
    // for any finite Lindblad rate.
    Real draw_uniform() noexcept;

    // Draw integer in [0, n-1] with uniform distribution.
    // Used for channel selection.
    // Precondition: n > 0
    std::size_t draw_int(std::size_t n) noexcept;

    // ── State access (for checkpointing and testing) ──────────────────────────

    const std::array<std::uint64_t, 4>& state() const noexcept { return s_; }

    void set_state(const std::array<std::uint64_t, 4>& s) noexcept { s_ = s; }

    // ── Serialization ─────────────────────────────────────────────────────────

    void serialize(std::ostream& os) const;
    static RNGState deserialize(std::istream& is);

private:
    std::array<std::uint64_t, 4> s_{1, 2, 3, 4};  // Must not be all-zero

    // xoshiro256** core rotation
    static std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    // splitmix64 — used for seeding only, not simulation
    static std::uint64_t splitmix64(std::uint64_t& x) noexcept {
        x += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

} // namespace liquid
