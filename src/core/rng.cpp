// liquid/core/rng.cpp
// ─────────────────────────────────────────────────────────────────────────────
// xoshiro256** implementation.
// Reference: https://prng.di.unimi.it/xoshiro256starstar.c
// Blackman and Vigna, 2018.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/rng.hpp"

#include <cassert>
#include <istream>
#include <ostream>

namespace liquid {

void RNGState::seed(Seed global_seed, TrajId traj_id) noexcept {
    // Combine global_seed and traj_id into a single 64-bit value.
    // XOR with a large prime to ensure (0, 0) doesn't produce all-zero state.
    // We then use splitmix64 to expand into the full 256-bit state.
    //
    // This construction guarantees:
    //   - Different traj_ids → different streams (splitmix64 is bijective)
    //   - Different global_seeds → different streams
    //   - The state is never all-zero (splitmix64 output is never all-zero
    //     for a non-degenerate input sequence)

    std::uint64_t x = global_seed ^ (traj_id * 0x9e3779b97f4a7c15ULL);

    // Advance x enough to separate streams even for nearby traj_ids
    x ^= 0x6c62272e07bb0142ULL;  // arbitrary decorrelation constant

    s_[0] = splitmix64(x);
    s_[1] = splitmix64(x);
    s_[2] = splitmix64(x);
    s_[3] = splitmix64(x);

    // Paranoia: state must not be all-zero (xoshiro256** would get stuck)
    // splitmix64 never produces all-zero output, so this is always satisfied.
    // Assert in debug mode as a sanity check.
    assert(s_[0] != 0 || s_[1] != 0 || s_[2] != 0 || s_[3] != 0);
}

std::uint64_t RNGState::next_uint64() noexcept {
    // xoshiro256** core
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;

    const std::uint64_t t = s_[1] << 17;

    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];

    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);

    return result;
}

Real RNGState::draw_uniform() noexcept {
    // Convert 64-bit integer to double in (0, 1].
    //
    // Method: use the upper 53 bits (double mantissa) and divide by 2^53.
    // Adding 1 before division shifts the range from [0, 1) to (0, 1].
    //
    // Why (0, 1] not [0, 1)?
    // The jump threshold r is compared against ||psi||^2 which starts at 1.0.
    // If r=0, the condition ||psi||^2 < r is never true at t=0 (since ||psi||=1),
    // but conceptually r=0 means "jump immediately". We exclude 0 to avoid
    // this degenerate case and keep the physics clean.

    const std::uint64_t raw = next_uint64();
    // Take upper 53 bits, map to [1, 2^53], divide by 2^53 → (0, 1]
    const std::uint64_t mantissa = (raw >> 11) + 1ULL;  // in [1, 2^53]
    constexpr double scale = 1.0 / static_cast<double>(1ULL << 53);
    return static_cast<double>(mantissa) * scale;
}

std::size_t RNGState::draw_int(std::size_t n) noexcept {
    assert(n > 0);
    if (n == 1) return 0;

    // Rejection sampling for unbiased uniform integer in [0, n-1].
    // The threshold ensures we reject the last incomplete bucket.
    const std::uint64_t threshold = (-static_cast<std::uint64_t>(n)) % n;

    while (true) {
        const std::uint64_t r = next_uint64();
        if (r >= threshold) {
            return static_cast<std::size_t>(r % n);
        }
        // Rejection: redraw. Expected iterations: < 2 for any n.
    }
}

void RNGState::serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(s_.data()),
             static_cast<std::streamsize>(s_.size() * sizeof(std::uint64_t)));
}

RNGState RNGState::deserialize(std::istream& is) {
    RNGState rng;
    is.read(reinterpret_cast<char*>(rng.s_.data()),
            static_cast<std::streamsize>(rng.s_.size() * sizeof(std::uint64_t)));
    return rng;
}

} // namespace liquid
