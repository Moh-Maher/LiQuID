// tests/unit/test_linalg.cpp
// Unit tests for liquid::StateVector and liquid::DenseOperator

#include "liquid/linalg/dense.hpp"
#include "../test_framework.hpp"

#include <cmath>

using namespace liquid;

// ─────────────────────────────────────────────────────────────────────────────
// StateVector tests
// ─────────────────────────────────────────────────────────────────────────────

TEST("StateVector: construction and size") {
    StateVector v(4);
    REQUIRE(v.size() == 4);
}
END_TEST

TEST("StateVector: zero initialization") {
    StateVector v(3);
    for (Idx i = 0; i < 3; ++i) {
        REQUIRE_CLOSE(v[i].real(), 0.0, 1e-15);
        REQUIRE_CLOSE(v[i].imag(), 0.0, 1e-15);
    }
}
END_TEST

TEST("StateVector: norm_sq of |0> + |1> / sqrt(2)") {
    StateVector v(2);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    v[0] = Scalar{inv_sqrt2, 0.0};
    v[1] = Scalar{inv_sqrt2, 0.0};
    REQUIRE_CLOSE(v.norm_sq(), 1.0, 1e-14);
    REQUIRE_CLOSE(v.norm(), 1.0, 1e-14);
}
END_TEST

TEST("StateVector: norm_sq of unnormalized vector") {
    StateVector v(2);
    v[0] = Scalar{3.0, 0.0};
    v[1] = Scalar{4.0, 0.0};
    REQUIRE_CLOSE(v.norm_sq(), 25.0, 1e-13);
    REQUIRE_CLOSE(v.norm(), 5.0, 1e-13);
}
END_TEST

TEST("StateVector: normalize") {
    StateVector v(2);
    v[0] = Scalar{3.0, 0.0};
    v[1] = Scalar{4.0, 0.0};
    v.normalize();
    REQUIRE_CLOSE(v.norm_sq(), 1.0, 1e-14);
    REQUIRE_CLOSE(v[0].real(), 0.6, 1e-14);
    REQUIRE_CLOSE(v[1].real(), 0.8, 1e-14);
}
END_TEST

TEST("StateVector: add_scaled") {
    StateVector a(2), b(2);
    a[0] = Scalar{1.0, 0.0};
    a[1] = Scalar{2.0, 0.0};
    b[0] = Scalar{3.0, 0.0};
    b[1] = Scalar{4.0, 0.0};
    // a += 2.0 * b
    a.add_scaled(b, Scalar{2.0, 0.0});
    REQUIRE_CLOSE(a[0].real(), 7.0, 1e-14);
    REQUIRE_CLOSE(a[1].real(), 10.0, 1e-14);
}
END_TEST

TEST("StateVector: inner product <phi|psi>") {
    // <e|g> = 0 for orthogonal basis states
    StateVector e(2), g(2);
    e[0] = Scalar{1.0, 0.0};
    g[1] = Scalar{1.0, 0.0};
    const Scalar ip = StateVector::inner(e, g);
    REQUIRE_CLOSE(ip.real(), 0.0, 1e-15);
    REQUIRE_CLOSE(ip.imag(), 0.0, 1e-15);
}
END_TEST

TEST("StateVector: inner product is conjugate-linear in first argument") {
    // ⟨iψ|φ⟩ = -i ⟨ψ|φ⟩
    StateVector psi(2), phi(2);
    psi[0] = Scalar{1.0, 0.0};
    psi[1] = Scalar{0.0, 1.0};  // psi = |0> + i|1>
    phi[0] = Scalar{1.0, 0.0};
    phi[1] = Scalar{1.0, 0.0};  // phi = |0> + |1>

    const Scalar ip1 = StateVector::inner(psi, phi);

    // i*psi
    StateVector ipsi(2);
    ipsi[0] = Scalar{0.0, 1.0};   // i * psi[0]
    ipsi[1] = Scalar{-1.0, 0.0};  // i * psi[1] = i*(0+i) = -1

    const Scalar ip2 = StateVector::inner(ipsi, phi);

    // ip2 should == -i * ip1
    const Scalar expected = Scalar{0.0, -1.0} * ip1;
    REQUIRE_CLOSE(ip2.real(), expected.real(), 1e-14);
    REQUIRE_CLOSE(ip2.imag(), expected.imag(), 1e-14);
}
END_TEST

// ─────────────────────────────────────────────────────────────────────────────
// DenseOperator tests
// ─────────────────────────────────────────────────────────────────────────────

TEST("DenseOperator: construction and element access") {
    DenseOperator A(3);
    A(0, 1) = Scalar{2.0, 0.0};
    A(1, 2) = Scalar{0.0, 3.0};
    REQUIRE_CLOSE(A(0, 1).real(), 2.0, 1e-15);
    REQUIRE_CLOSE(A(1, 2).imag(), 3.0, 1e-15);
    REQUIRE_CLOSE(A(0, 0).real(), 0.0, 1e-15);
}
END_TEST

TEST("DenseOperator: identity matrix-vector product") {
    const DenseOperator I = make_identity(3);
    StateVector v(3);
    v[0] = Scalar{1.0, 2.0};
    v[1] = Scalar{3.0, 4.0};
    v[2] = Scalar{5.0, 6.0};

    const StateVector Iv = I.apply(v);
    for (Idx i = 0; i < 3; ++i) {
        REQUIRE_CLOSE(Iv[i].real(), v[i].real(), 1e-14);
        REQUIRE_CLOSE(Iv[i].imag(), v[i].imag(), 1e-14);
    }
}
END_TEST

TEST("DenseOperator: sigma_z * excited_state = +excited_state") {
    // σ_z = [[1, 0], [0, -1]]
    // σ_z |e> = |e>  where |e> = [1, 0]^T
    DenseOperator sigma_z(2);
    sigma_z(0, 0) = Scalar{ 1.0, 0.0};
    sigma_z(1, 1) = Scalar{-1.0, 0.0};

    StateVector excited(2);
    excited[0] = Scalar{1.0, 0.0};

    const StateVector result = sigma_z.apply(excited);
    REQUIRE_CLOSE(result[0].real(),  1.0, 1e-14);
    REQUIRE_CLOSE(result[1].real(), -0.0, 1e-14);  // σ_z |e> = +|e>
}
END_TEST

TEST("DenseOperator: sigma_z * ground_state = -ground_state") {
    DenseOperator sigma_z(2);
    sigma_z(0, 0) = Scalar{ 1.0, 0.0};
    sigma_z(1, 1) = Scalar{-1.0, 0.0};

    StateVector ground(2);
    ground[1] = Scalar{1.0, 0.0};

    const StateVector result = sigma_z.apply(ground);
    REQUIRE_CLOSE(result[1].real(), -1.0, 1e-14);  // σ_z |g> = -|g>
}
END_TEST

TEST("DenseOperator: adjoint of sigma_minus is sigma_plus") {
    // σ_- = [[0,0],[1,0]], σ_+ = [[0,1],[0,0]]
    DenseOperator sigma_minus(2);
    sigma_minus(1, 0) = Scalar{1.0, 0.0};

    const DenseOperator sigma_plus = sigma_minus.adjoint();
    REQUIRE_CLOSE(sigma_plus(0, 1).real(), 1.0, 1e-15);
    REQUIRE_CLOSE(sigma_plus(0, 0).real(), 0.0, 1e-15);
    REQUIRE_CLOSE(sigma_plus(1, 0).real(), 0.0, 1e-15);
    REQUIRE_CLOSE(sigma_plus(1, 1).real(), 0.0, 1e-15);
}
END_TEST

TEST("DenseOperator: adjoint_times gives sigma_plus * sigma_minus = |e><e|") {
    // σ_-†σ_- = σ_+σ_- = [[1,0],[0,0]] = |e><e|
    DenseOperator sigma_minus(2);
    sigma_minus(1, 0) = Scalar{1.0, 0.0};

    const DenseOperator result = adjoint_times(sigma_minus);
    REQUIRE_CLOSE(result(0, 0).real(), 1.0, 1e-14);
    REQUIRE_CLOSE(result(0, 1).real(), 0.0, 1e-14);
    REQUIRE_CLOSE(result(1, 0).real(), 0.0, 1e-14);
    REQUIRE_CLOSE(result(1, 1).real(), 0.0, 1e-14);
}
END_TEST

TEST("DenseOperator: expectation value <e|sigma_z|e> = +1") {
    DenseOperator sigma_z(2);
    sigma_z(0, 0) = Scalar{ 1.0, 0.0};
    sigma_z(1, 1) = Scalar{-1.0, 0.0};

    StateVector excited(2);
    excited[0] = Scalar{1.0, 0.0};

    const Scalar ev = expectation(sigma_z, excited);
    REQUIRE_CLOSE(ev.real(), 1.0, 1e-14);
    REQUIRE_CLOSE(ev.imag(), 0.0, 1e-14);
}
END_TEST

TEST("DenseOperator: expectation value <g|sigma_z|g> = -1") {
    DenseOperator sigma_z(2);
    sigma_z(0, 0) = Scalar{ 1.0, 0.0};
    sigma_z(1, 1) = Scalar{-1.0, 0.0};

    StateVector ground(2);
    ground[1] = Scalar{1.0, 0.0};

    const Scalar ev = expectation(sigma_z, ground);
    REQUIRE_CLOSE(ev.real(), -1.0, 1e-14);
    REQUIRE_CLOSE(ev.imag(), 0.0, 1e-14);
}
END_TEST

int main() {
    std::printf("=== Linear Algebra Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
