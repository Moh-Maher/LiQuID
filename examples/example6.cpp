#include "liquid/liquid.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace liquid;
using namespace liquid::ensemble;

// Build the annihilation operator a for an N_fock-dimensional Fock space.
// a|n> = sqrt(n)|n-1>  for n >= 1; a|0> = 0
static SparseOperator make_a_fock(Dim N_fock) {
    std::vector<Triplet> trips;
    trips.reserve(N_fock - 1);
    for (Dim n = 1; n < N_fock; ++n) {
        trips.push_back({
            n - 1,                            // row (output state index)
            n,                                // col (input state index)
            Scalar{std::sqrt(static_cast<double>(n)), 0.0}});
    }
    return SparseOperator(N_fock, std::move(trips));
}

// Driven harmonic oscillator Hamiltonian:
// H = Delta * adaga + epsilon * (a + adag)
static SparseOperator make_H_harmonic(Dim N_fock, double Delta, double eps) {
    std::vector<Triplet> trips;
    // Diagonal: Delta * n
    for (Dim n = 0; n < N_fock; ++n)
        trips.push_back({n, n, Scalar{Delta * static_cast<double>(n), 0.0}});
    // Drive: epsilon * a  (lower triangular)
    for (Dim n = 1; n < N_fock; ++n)
        trips.push_back({n-1, n, Scalar{eps * std::sqrt(static_cast<double>(n)), 0.0}});
    // Drive: epsilon * adag  (upper triangular)
    for (Dim n = 1; n < N_fock; ++n)
        trips.push_back({n, n-1, Scalar{eps * std::sqrt(static_cast<double>(n)), 0.0}});
    return SparseOperator(N_fock, std::move(trips));
}

// Cavity decay: L = sqrt(kappa) * a
static SparseOperator make_L_harmonic(Dim N_fock, double kappa) {
    std::vector<Triplet> trips;
    for (Dim n = 1; n < N_fock; ++n)
        trips.push_back({
            n - 1, n,
            Scalar{std::sqrt(kappa * static_cast<double>(n)), 0.0}});
    return SparseOperator(N_fock, std::move(trips));
}

// Photon number observable: adaga = diag(0, 1, 2, ..., N_fock-1)
static ObservableDef make_n_obs(Dim N_fock) {
    std::vector<Triplet> trips;
    for (Dim n = 0; n < N_fock; ++n)
        trips.push_back({n, n, Scalar{static_cast<double>(n), 0.0}});
    SparseOperator n_op(N_fock, std::move(trips));
    return {"photon_n",
            [op = std::move(n_op)](const StateVector& psi) -> Real {
                const Real ns = psi.norm_sq();
                return ns > 1e-30
                    ? sparse_expectation(op, psi).real() / ns
                    : 0.0;
            }};
}

int main() {
    const Dim    N_fock = 40;    // truncation: must satisfy N >> <n>_ss
    const double Delta  = 0.0;   // on resonance with drive
    const double kappa  = 1.0;   // cavity decay rate
    const double eps    = 0.2;   // drive amplitude (weak: eps << kappa)
    const double T_ss   = 15.0;  // run until steady state

    // Exact steady-state photon number (linear cavity)
    const double exact_n_ss = (eps * eps) / (kappa * kappa);

    std::printf("N_fock=%zu  Delta=%.1f  kappa=%.1f  eps=%.2f\n",
        N_fock, Delta, kappa, eps);
    std::printf("Exact <n>_ss = eps^2/kappa^2 = %.4f\n\n", exact_n_ss);

    // Verify truncation is adequate: <n>_ss << N_fock
    if (exact_n_ss > static_cast<double>(N_fock) / 2.0) {
        std::fprintf(stderr,
            "WARNING: N_fock=%zu may be too small for <n>_ss=%.1f\n",
            N_fock, exact_n_ss);
    }

    // Build sparse system
    SparseOperator H  = make_H_harmonic(N_fock, Delta, eps);
    SparseOperator L  = make_L_harmonic(N_fock, kappa);

    std::vector<SparseOperator> ops;
    ops.push_back(std::move(L));
    SparseOpenSystem sys(std::move(H), LindbladSet<SparseTag>(std::move(ops)));

    // Report sparsity advantage over dense
    const double nnz      = static_cast<double>(sys.H_eff().nnz());
    const double dense_sq = static_cast<double>(N_fock * N_fock);
    std::printf("H_eff sparsity: %.2f%%  (%g non-zeros vs %g dense)\n\n",
        100.0 * nnz / dense_sq, nnz, dense_sq);

    std::vector<ObservableDef> obs_defs;
    obs_defs.push_back(make_n_obs(N_fock));

    StoppingCriteria sc;
    sc.min_trajectories = 300;
    sc.max_trajectories = 5000;
    sc.target_rel_sem   = 0.02;

    EnsembleConfig ec;
    ec.global_seed           = 42;
    ec.diag_level            = DiagnosticLevel::None;
    ec.propagator.dt_initial = 5e-3;

    // Factory for SparseOpenSystem + DOPRI45
    Simulation sim = Simulation::make_sparse_dopri45(
        std::move(sys), std::move(obs_defs), sc, ec);

    // Initial state: vacuum |0>
    StateVector psi0(N_fock);
    psi0[0] = Scalar{1.0, 0.0};

    auto result = sim.run(psi0, 0.0, T_ss);

    const auto& obs = result.observables[0];
    std::printf("=== Driven harmonic oscillator (N_fock=%zu) ===\n", N_fock);
    std::printf("  <n>(MCWF)  = %.4f +/- %.4f  (N=%zu)\n",
        obs.mean, obs.sem, result.total_trajectories);
    std::printf("  <n>(exact) = %.4f\n", exact_n_ss);
    std::printf("  |error|    = %.4f  (%.1f sigma)\n",
        std::abs(obs.mean - exact_n_ss),
        std::abs(obs.mean - exact_n_ss) / obs.sem);

    return 0;
}
