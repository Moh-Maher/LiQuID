// tests/unit/test_opensystem.cpp
// Unit tests for LindbladSet and OpenSystem

#include "liquid/system/open_system.hpp"
#include "../test_framework.hpp"

#include <cmath>

using namespace liquid;

// ── Helper: two-level system operators ───────────────────────────────────────

static DenseOperator make_sigma_z() {
    DenseOperator sz(2);
    sz(0, 0) = Scalar{ 1.0, 0.0};
    sz(1, 1) = Scalar{-1.0, 0.0};
    return sz;
}

static DenseOperator make_sigma_minus(double gamma = 1.0) {
    // L = sqrt(gamma) * sigma_minus
    DenseOperator sm(2);
    sm(1, 0) = Scalar{std::sqrt(gamma), 0.0};
    return sm;
}

// ─────────────────────────────────────────────────────────────────────────────
// LindbladSet tests
// ─────────────────────────────────────────────────────────────────────────────

TEST("LindbladSet: precomputed gamma = L†L for single channel") {
    // L = sqrt(gamma) sigma_minus
    // L†L = gamma * sigma_plus * sigma_minus = gamma * |e><e|
    const double gamma = 2.0;
    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(gamma));

    LindbladSet<DenseTag> L(std::move(ops));

    const DenseOperator& Gamma = L.decay_operator();

    // Gamma should be gamma * |e><e| = [[gamma, 0], [0, 0]]
    REQUIRE_CLOSE(Gamma(0, 0).real(), gamma, 1e-13);
    REQUIRE_CLOSE(Gamma(0, 0).imag(), 0.0,   1e-13);
    REQUIRE_CLOSE(Gamma(0, 1).real(), 0.0,   1e-13);
    REQUIRE_CLOSE(Gamma(1, 0).real(), 0.0,   1e-13);
    REQUIRE_CLOSE(Gamma(1, 1).real(), 0.0,   1e-13);
}
END_TEST

TEST("LindbladSet: apply_decay on excited state gives correct contribution") {
    // L = sqrt(gamma) sigma_minus, gamma = 1
    // Gamma = |e><e| = [[1,0],[0,0]]
    // apply_decay: out += -(i/2) * Gamma * psi
    // For psi = |e> = [1,0]:
    //   -(i/2) * Gamma * |e> = -(i/2) * [1,0] = [0,-0.5]*i ... = [0, -i/2]
    //   i.e., component 0: Scalar{0, -0.5}
    const double gamma = 1.0;
    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(gamma));
    LindbladSet<DenseTag> L(std::move(ops));

    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};  // excited state

    StateVector out(2);
    out.set_zero();
    L.apply_decay(psi, out);

    // out = -(i/2) * [[1,0],[0,0]] * [1,0]^T = -(i/2)*[1,0]^T
    // = [Scalar{0,-0.5}, 0]
    REQUIRE_CLOSE(out[0].real(),  0.0,  1e-14);
    REQUIRE_CLOSE(out[0].imag(), -0.5,  1e-14);
    REQUIRE_CLOSE(out[1].real(),  0.0,  1e-14);
    REQUIRE_CLOSE(out[1].imag(),  0.0,  1e-14);
}
END_TEST

TEST("LindbladSet: apply_decay on ground state gives zero") {
    // Gamma = |e><e|, so Gamma * |g> = 0
    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(1.0));
    LindbladSet<DenseTag> L(std::move(ops));

    StateVector psi(2);
    psi[1] = Scalar{1.0, 0.0};  // ground state

    StateVector out(2);
    out.set_zero();
    L.apply_decay(psi, out);

    REQUIRE_CLOSE(out[0].real(), 0.0, 1e-14);
    REQUIRE_CLOSE(out[1].real(), 0.0, 1e-14);
}
END_TEST

TEST("LindbladSet: channel_probability of excited state = gamma") {
    // pₖ = ||Lₖ|e>||² = ||sqrt(gamma)|g>||² = gamma
    const double gamma = 3.0;
    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(gamma));
    LindbladSet<DenseTag> L(std::move(ops));

    StateVector psi(2);
    psi[0] = Scalar{1.0, 0.0};  // excited state

    StateVector scratch(2);
    const Real p = L.channel_probability(0, psi, scratch);
    REQUIRE_CLOSE(p, gamma, 1e-13);
}
END_TEST

TEST("LindbladSet: channel_probability of ground state = 0") {
    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(1.0));
    LindbladSet<DenseTag> L(std::move(ops));

    StateVector psi(2);
    psi[1] = Scalar{1.0, 0.0};  // ground state

    StateVector scratch(2);
    const Real p = L.channel_probability(0, psi, scratch);
    REQUIRE_CLOSE(p, 0.0, 1e-14);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// OpenSystem tests
// ─────────────────────────────────────────────────────────────────────────────

TEST("OpenSystem: H_eff for two-level decay") {
    // H = (omega/2) sigma_z,  L = sqrt(gamma) sigma_minus
    // H_eff = H - (i/2) * gamma * |e><e|
    //
    // H_eff matrix:
    //   [omega/2 - i*gamma/2,   0             ]
    //   [0,                    -omega/2        ]
    const double omega = 2.0;
    const double gamma = 1.0;

    DenseOperator H(2);
    H(0, 0) = Scalar{omega / 2.0, 0.0};
    H(1, 1) = Scalar{-omega / 2.0, 0.0};

    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(gamma));
    LindbladSet<DenseTag> L(std::move(ops));

    DenseOpenSystem sys(std::move(H), std::move(L));

    const DenseOperator& H_eff = sys.H_eff();

    // (0,0): omega/2 - i*gamma/2
    REQUIRE_CLOSE(H_eff(0, 0).real(),  omega / 2.0,   1e-13);
    REQUIRE_CLOSE(H_eff(0, 0).imag(), -gamma / 2.0,   1e-13);

    // (1,1): -omega/2 (ground state: no decay contribution)
    REQUIRE_CLOSE(H_eff(1, 1).real(), -omega / 2.0,  1e-13);
    REQUIRE_CLOSE(H_eff(1, 1).imag(),  0.0,           1e-13);

    // Off-diagonals: zero
    REQUIRE_CLOSE(H_eff(0, 1).real(), 0.0, 1e-13);
    REQUIRE_CLOSE(H_eff(1, 0).real(), 0.0, 1e-13);
}
END_TEST

TEST("OpenSystem: apply_Heff on excited state") {
    // d|ψ>/dt = -i H_eff |ψ>
    // For |e> = [1,0]:
    //   H_eff |e> = [omega/2 - i*gamma/2, 0]
    //   -i * H_eff |e> = -i * [omega/2 - i*gamma/2, 0]
    //                  = [-i*omega/2 - gamma/2, 0]
    //                  = Scalar{-gamma/2, -omega/2}
    const double omega = 2.0, gamma = 1.0;

    DenseOperator H(2);
    H(0, 0) = Scalar{omega / 2.0, 0.0};
    H(1, 1) = Scalar{-omega / 2.0, 0.0};

    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(gamma));
    LindbladSet<DenseTag> L(std::move(ops));
    DenseOpenSystem sys(std::move(H), std::move(L));

    StateVector psi(2), out(2);
    psi[0] = Scalar{1.0, 0.0};
    sys.apply_Heff(psi, out);

    REQUIRE_CLOSE(out[0].real(), -gamma / 2.0, 1e-13);
    REQUIRE_CLOSE(out[0].imag(), -omega / 2.0, 1e-13);
    REQUIRE_CLOSE(out[1].real(), 0.0,          1e-13);
    REQUIRE_CLOSE(out[1].imag(), 0.0,          1e-13);
}
END_TEST

TEST("OpenSystem: apply_jump produces normalized state") {
    // Jumping from |e> via sigma_minus should yield |g> = [0,1]
    DenseOperator H(2);
    H(0, 0) = Scalar{1.0, 0.0};
    H(1, 1) = Scalar{-1.0, 0.0};

    std::vector<DenseOperator> ops;
    ops.push_back(make_sigma_minus(1.0));
    LindbladSet<DenseTag> L(std::move(ops));
    DenseOpenSystem sys(std::move(H), std::move(L));

    StateVector psi(2), scratch(2);
    psi[0] = Scalar{1.0, 0.0};  // excited

    sys.apply_jump(0, psi, scratch);

    // After jump: psi should be |g> = [0, 1]
    REQUIRE_CLOSE(psi.norm_sq(), 1.0, 1e-14);  // normalized
    REQUIRE_CLOSE(psi[0].real(), 0.0, 1e-14);
    REQUIRE_CLOSE(psi[1].real(), 1.0, 1e-14);
}
END_TEST

int main() {
    std::printf("=== OpenSystem Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
