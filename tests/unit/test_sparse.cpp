// tests/unit/test_sparse.cpp

#include "liquid/liquid.hpp"
#include "../test_framework.hpp"
#include <cmath>

using namespace liquid;

// ── Construction ──────────────────────────────────────────────────────────────

TEST("SparseOperator: construction from triplets") {
    std::vector<Triplet> trips = {
        {0, 0, Scalar{ 1.0, 0.0}},
        {1, 1, Scalar{-1.0, 0.0}},
    };
    SparseOperator sz(2, std::move(trips));
    REQUIRE(sz.size() == 2);
    REQUIRE(sz.nnz()  == 2);
    REQUIRE_CLOSE(sz(0,0).real(),  1.0, 1e-15);
    REQUIRE_CLOSE(sz(1,1).real(), -1.0, 1e-15);
    REQUIRE_CLOSE(sz(0,1).real(),  0.0, 1e-15);
}
END_TEST

TEST("SparseOperator: duplicate triplets are summed") {
    std::vector<Triplet> trips = {
        {0, 0, Scalar{1.0, 0.0}},
        {0, 0, Scalar{2.0, 0.0}},  // duplicate — should sum to 3
    };
    SparseOperator A(2, std::move(trips));
    REQUIRE_CLOSE(A(0,0).real(), 3.0, 1e-14);
    REQUIRE(A.nnz() == 1);
}
END_TEST

TEST("SparseOperator: sparsity for sigma_minus") {
    std::vector<Triplet> trips = {{1, 0, Scalar{1.0, 0.0}}};
    SparseOperator sm(2, std::move(trips));
    // 1 element out of 4 = 25%
    REQUIRE_CLOSE(sm.sparsity(), 0.25, 1e-14);
}
END_TEST

// ── Matrix-vector product ─────────────────────────────────────────────────────

TEST("SparseOperator: apply_add sigma_z * |e> = +|e>") {
    std::vector<Triplet> trips = {
        {0, 0, Scalar{ 1.0, 0.0}},
        {1, 1, Scalar{-1.0, 0.0}},
    };
    SparseOperator sz(2, std::move(trips));

    StateVector psi(2), out(2);
    psi[0] = Scalar{1.0, 0.0};  // excited
    sz.apply_add(psi, Scalar{1.0, 0.0}, out);

    REQUIRE_CLOSE(out[0].real(),  1.0, 1e-14);
    REQUIRE_CLOSE(out[1].real(),  0.0, 1e-14);
}
END_TEST

TEST("SparseOperator: apply_add sigma_minus * |e> = |g>") {
    std::vector<Triplet> trips = {{1, 0, Scalar{1.0, 0.0}}};
    SparseOperator sm(2, std::move(trips));

    StateVector psi(2), out(2);
    psi[0] = Scalar{1.0, 0.0};  // excited
    sm.apply_add(psi, Scalar{1.0, 0.0}, out);

    REQUIRE_CLOSE(out[0].real(), 0.0, 1e-14);
    REQUIRE_CLOSE(out[1].real(), 1.0, 1e-14);
}
END_TEST

TEST("SparseOperator: apply_add accumulates correctly") {
    // out starts at [1, 2], add 2 * I * [1, 0] = [2, 0], result = [3, 2]
    SparseOperator I = make_sparse_identity(2);
    StateVector psi(2), out(2);
    psi[0] = Scalar{1.0, 0.0};
    out[0] = Scalar{1.0, 0.0};
    out[1] = Scalar{2.0, 0.0};
    I.apply_add(psi, Scalar{2.0, 0.0}, out);
    REQUIRE_CLOSE(out[0].real(), 3.0, 1e-14);
    REQUIRE_CLOSE(out[1].real(), 2.0, 1e-14);
}
END_TEST

// ── Adjoint ───────────────────────────────────────────────────────────────────

TEST("SparseOperator: adjoint of sigma_minus is sigma_plus") {
    std::vector<Triplet> trips = {{1, 0, Scalar{1.0, 0.0}}};
    SparseOperator sm(2, std::move(trips));
    SparseOperator sp = sm.adjoint();

    REQUIRE_CLOSE(sp(0,1).real(), 1.0, 1e-14);  // sigma_+[0,1] = 1
    REQUIRE_CLOSE(sp(1,0).real(), 0.0, 1e-14);
}
END_TEST

TEST("SparseOperator: adjoint of complex operator conjugates values") {
    std::vector<Triplet> trips = {{0, 1, Scalar{0.0, 1.0}}};  // i * |0><1|
    SparseOperator A(2, std::move(trips));
    SparseOperator Adag = A.adjoint();

    // (A†)[1,0] = conj(A[0,1]) = conj(i) = -i
    REQUIRE_CLOSE(Adag(1,0).real(),  0.0, 1e-14);
    REQUIRE_CLOSE(Adag(1,0).imag(), -1.0, 1e-14);
}
END_TEST

// ── sparse_adjoint_times ──────────────────────────────────────────────────────

TEST("SparseOperator: sparse_adjoint_times gives L†L = |e><e|") {
    // L = sigma_minus = [[0,0],[1,0]]
    // L†L = sigma_plus * sigma_minus = [[1,0],[0,0]] = |e><e|
    std::vector<Triplet> trips = {{1, 0, Scalar{1.0, 0.0}}};
    SparseOperator sm(2, std::move(trips));
    SparseOperator LdagL = sparse_adjoint_times(sm);

    REQUIRE_CLOSE(LdagL(0,0).real(), 1.0, 1e-14);
    REQUIRE_CLOSE(LdagL(1,1).real(), 0.0, 1e-14);
    REQUIRE_CLOSE(LdagL(0,1).real(), 0.0, 1e-14);
    REQUIRE_CLOSE(LdagL(1,0).real(), 0.0, 1e-14);
}
END_TEST

// ── to_sparse / to_dense round-trip ──────────────────────────────────────────

TEST("SparseOperator: dense->sparse->dense round-trip") {
    DenseOperator H(2);
    H(0,0) = Scalar{ 0.5, 0.0};
    H(1,1) = Scalar{-0.5, 0.0};

    SparseOperator sH = to_sparse(H);
    DenseOperator  dH = sH.to_dense();

    REQUIRE_CLOSE(dH(0,0).real(),  0.5, 1e-14);
    REQUIRE_CLOSE(dH(1,1).real(), -0.5, 1e-14);
    REQUIRE_CLOSE(dH(0,1).real(),  0.0, 1e-14);
}
END_TEST

// ── SparseOpenSystem ──────────────────────────────────────────────────────────

TEST("SparseOpenSystem: H_eff precomputed correctly") {
    const double omega = 2.0, gamma = 1.0;

    std::vector<Triplet> H_trips = {
        {0, 0, Scalar{ omega/2.0, 0.0}},
        {1, 1, Scalar{-omega/2.0, 0.0}},
    };
    SparseOperator H(2, std::move(H_trips));

    std::vector<Triplet> L_trips = {{1, 0, Scalar{std::sqrt(gamma), 0.0}}};
    std::vector<SparseOperator> ops;
    ops.emplace_back(2, std::move(L_trips));
    LindbladSet<SparseTag> L(std::move(ops));

    SparseOpenSystem sys(std::move(H), std::move(L));

    // H_eff[0,0] = omega/2 - i*gamma/2
    const SparseOperator& Heff = sys.H_eff();
    REQUIRE_CLOSE(Heff(0,0).real(),  omega/2.0,  1e-13);
    REQUIRE_CLOSE(Heff(0,0).imag(), -gamma/2.0,  1e-13);
    REQUIRE_CLOSE(Heff(1,1).real(), -omega/2.0,  1e-13);
    REQUIRE_CLOSE(Heff(1,1).imag(),  0.0,         1e-13);
}
END_TEST

TEST("SparseOpenSystem: apply_Heff matches DenseOpenSystem") {
    const double omega = 1.0, gamma = 1.0;

    // Build both sparse and dense systems
    DenseOperator Hd(2);
    Hd(0,0)=Scalar{omega/2.0,0}; Hd(1,1)=Scalar{-omega/2.0,0};
    DenseOperator Ld(2); Ld(1,0)=Scalar{std::sqrt(gamma),0};
    std::vector<DenseOperator> dops; dops.push_back(std::move(Ld));
    DenseOpenSystem dense_sys(std::move(Hd), LindbladSet<DenseTag>(std::move(dops)));

    std::vector<Triplet> Ht = {{0,0,Scalar{omega/2.0,0}},{1,1,Scalar{-omega/2.0,0}}};
    std::vector<Triplet> Lt = {{1,0,Scalar{std::sqrt(gamma),0}}};
    std::vector<SparseOperator> sops; sops.emplace_back(2, std::move(Lt));
    SparseOpenSystem sparse_sys(SparseOperator(2, std::move(Ht)),
                                LindbladSet<SparseTag>(std::move(sops)));

    StateVector psi(2), out_d(2), out_s(2);
    psi[0] = Scalar{1.0, 0.0};  // excited state

    dense_sys.apply_Heff(psi, out_d);
    sparse_sys.apply_Heff(psi, out_s);

    for (Idx i = 0; i < 2; ++i) {
        REQUIRE_CLOSE(out_s[i].real(), out_d[i].real(), 1e-13);
        REQUIRE_CLOSE(out_s[i].imag(), out_d[i].imag(), 1e-13);
    }
}
END_TEST

// ── Time-dependent system ─────────────────────────────────────────────────────

TEST("TimeDependentOpenSystem: apply_Heff at t=0 matches static") {
    // H(t) = H_static + f(t)*H_drive, with f(t) = sin(t)
    // At t=0: f(0)=0, so H(0) = H_static
    const double omega = 1.0, gamma = 1.0, Omega = 0.5;

    DenseOperator H_static(2);
    // H_static is zero (rotating frame) — drive term carries all the coupling
    (void)omega;

    DenseOperator H_drive(2);
    H_drive(0,1)=Scalar{Omega,0}; H_drive(1,0)=Scalar{Omega,0};

    DenseOperator L(2); L(1,0)=Scalar{std::sqrt(gamma),0};
    std::vector<DenseOperator> ops; ops.push_back(std::move(L));
    LindbladSet<DenseTag> lb(std::move(ops));

    std::vector<DriveTerm<DenseTag>> drives;
    drives.push_back({[](Real t){ return std::sin(t); }, std::move(H_drive)});

    DenseTDOpenSystem td_sys(std::move(H_static), std::move(drives), std::move(lb));

    // Reference: static system without drive (H=0 to match TD at t=0 with f(0)=0)
    DenseOperator H_ref(2);
    // H_ref is zero (same as H_static in TD system)
    DenseOperator L_ref(2); L_ref(1,0)=Scalar{std::sqrt(gamma),0};
    std::vector<DenseOperator> ops_ref; ops_ref.push_back(std::move(L_ref));
    DenseOpenSystem static_sys(std::move(H_ref),
                               LindbladSet<DenseTag>(std::move(ops_ref)));

    StateVector psi(2), out_td(2), out_static(2);
    psi[0] = Scalar{1.0, 0.0};

    td_sys.apply_Heff(0.0, psi, out_td);    // f(0) = sin(0) = 0
    static_sys.apply_Heff(psi, out_static);

    for (Idx i = 0; i < 2; ++i) {
        REQUIRE_CLOSE(out_td[i].real(), out_static[i].real(), 1e-13);
        REQUIRE_CLOSE(out_td[i].imag(), out_static[i].imag(), 1e-13);
    }
}
END_TEST

int main() {
    std::printf("=== Sparse Operator Unit Tests ===\n");
    return RUN_ALL_TESTS();
}
