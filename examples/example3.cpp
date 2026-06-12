#include "liquid/liquid.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace liquid;
using namespace liquid::trajectory;
using namespace liquid::ode;
using namespace liquid::ensemble;

// Basis: |n, s> -> index 2*n + s  (s=0: excited, s=1: ground)
static Idx jc_idx(int n, int s) { return static_cast<Idx>(2 * n + s); }

// Cavity photon loss: L1 = sqrt(kappa) * a
// a|n,s> = sqrt(n)|n-1,s>
static SparseOperator make_L_cavity(int N_fock, double kappa) {
    const Dim dim = 2 * N_fock;
    std::vector<Triplet> trips;
    for (int n = 1; n < N_fock; ++n) {
        const double v = std::sqrt(kappa * static_cast<double>(n));
        trips.push_back({jc_idx(n-1, 0), jc_idx(n, 0), Scalar{v, 0.0}});
        trips.push_back({jc_idx(n-1, 1), jc_idx(n, 1), Scalar{v, 0.0}});
    }
    return SparseOperator(dim, std::move(trips));
}

// Atomic decay: L2 = sqrt(gamma_a) * sigma_minus
// sigma_minus|n,e> = |n,g>
static SparseOperator make_L_atomic(int N_fock, double gamma_a) {
    const Dim dim = 2 * N_fock;
    std::vector<Triplet> trips;
    const double v = std::sqrt(gamma_a);
    for (int n = 0; n < N_fock; ++n)
        trips.push_back({jc_idx(n, 1), jc_idx(n, 0), Scalar{v, 0.0}});
    return SparseOperator(dim, std::move(trips));
}

// JC Hamiltonian: omega*adaga + (omega/2)*sigma_z + g*(adagsigma_- + a*sigma_+)
// Coupling term: adagsigma_-|n,e> = sqrt(n+1)|n+1,g>
static SparseOperator make_H_jc(int N_fock, double omega, double g) {
    const Dim dim = 2 * N_fock;
    std::vector<Triplet> trips;

    // Free cavity: omega * adaga diagonal
    for (int n = 0; n < N_fock; ++n) {
        const double diag = omega * static_cast<double>(n);
        trips.push_back({jc_idx(n,0), jc_idx(n,0), Scalar{diag, 0.0}});
        trips.push_back({jc_idx(n,1), jc_idx(n,1), Scalar{diag, 0.0}});
    }

    // Free atom: (omega/2)*sigma_z diagonal
    for (int n = 0; n < N_fock; ++n) {
        trips.push_back({jc_idx(n,0), jc_idx(n,0), Scalar{ omega/2.0, 0.0}});
        trips.push_back({jc_idx(n,1), jc_idx(n,1), Scalar{-omega/2.0, 0.0}});
    }

    // JC interaction: adagsigma_- couples |n,e> -> |n+1,g>
    for (int n = 0; n < N_fock - 1; ++n) {
        const double v = g * std::sqrt(static_cast<double>(n + 1));
        trips.push_back({jc_idx(n+1, 1), jc_idx(n, 0), Scalar{v, 0.0}});  // adagsigma_-
        trips.push_back({jc_idx(n, 0), jc_idx(n+1, 1), Scalar{v, 0.0}});  // h.c. a*sigma_+
    }
    return SparseOperator(dim, std::move(trips));
}

// Photon number observable: n_hat = sum_n n * (|n,e><n,e| + |n,g><n,g|)
static ObservableDef make_photon_number_obs(int N_fock) {
    const Dim dim = 2 * N_fock;
    std::vector<Triplet> trips;
    for (int n = 0; n < N_fock; ++n) {
        const double val = static_cast<double>(n);
        trips.push_back({jc_idx(n,0), jc_idx(n,0), Scalar{val, 0.0}});
        trips.push_back({jc_idx(n,1), jc_idx(n,1), Scalar{val, 0.0}});
    }
    SparseOperator n_op(dim, std::move(trips));
    return {"photon_number",
            [op = std::move(n_op)](const StateVector& psi) -> Real {
                const Real ns = psi.norm_sq();
                return ns > 1e-30
                    ? sparse_expectation(op, psi).real() / ns
                    : 0.0;
            }};
}

// Excited-state population observable
static ObservableDef make_excited_pop_obs(int N_fock) {
    const Dim dim = 2 * N_fock;
    std::vector<Triplet> trips;
    for (int n = 0; n < N_fock; ++n)
        trips.push_back({jc_idx(n,0), jc_idx(n,0), Scalar{1.0, 0.0}});
    SparseOperator pe_op(dim, std::move(trips));
    return {"excited_pop",
            [op = std::move(pe_op)](const StateVector& psi) -> Real {
                const Real ns = psi.norm_sq();
                return ns > 1e-30
                    ? sparse_expectation(op, psi).real() / ns
                    : 0.0;
            }};
}

int main() {
    const int    N_fock   = 12;    // Fock space truncation
    const double omega    = 1.0;   // cavity / atom frequency
    const double g        = 0.5;   // vacuum Rabi coupling
    const double kappa    = 0.2;   // cavity photon loss rate
    const double gamma_a  = 0.05;  // atomic decay rate
    const double T        = 5.0;   // simulation time

    // Build sparse JC system
    SparseOperator H   = make_H_jc(N_fock, omega, g);
    SparseOperator L1  = make_L_cavity(N_fock, kappa);
    SparseOperator L2  = make_L_atomic(N_fock, gamma_a);

    std::vector<SparseOperator> ops;
    ops.push_back(std::move(L1));
    ops.push_back(std::move(L2));

    SparseOpenSystem sys(
        std::move(H),
        LindbladSet<SparseTag>(std::move(ops)));

    // Observables
    std::vector<ObservableDef> obs_defs;
    obs_defs.push_back(make_photon_number_obs(N_fock));
    obs_defs.push_back(make_excited_pop_obs(N_fock));

    // Stopping criteria
    StoppingCriteria sc;
    sc.min_trajectories = 500;
    sc.max_trajectories = 5000;
    sc.target_rel_sem   = 0.03;

    EnsembleConfig ec;
    ec.global_seed           = 42;
    ec.diag_level            = DiagnosticLevel::None;
    ec.propagator.dt_initial = 5e-3;

    // Use the sparse+DOPRI45 factory (SparseOpenSystem requires
    // Simulation::make_sparse_dopri45; SimulationBuilder supports
    // DenseOperator only in v0.7.0)
    Simulation sim = Simulation::make_sparse_dopri45(
        std::move(sys), std::move(obs_defs), sc, ec);

    // Initial state: |n=1, e> -- one photon, atom excited
    StateVector psi0(2 * N_fock);
    psi0[jc_idx(1, 0)] = Scalar{1.0, 0.0};

    auto result = sim.run(psi0, 0.0, T);

    std::printf("=== Jaynes-Cummings (N_fock=%d, g/kappa=%.1f) ===\n",
        N_fock, g / kappa);
    std::printf("  <n_photon>  = %.4f +/- %.4f  (N=%zu)\n",
        result.observables[0].mean, result.observables[0].sem,
        result.total_trajectories);
    std::printf("  <rho_e>     = %.4f +/- %.4f\n",
        result.observables[1].mean, result.observables[1].sem);
    std::printf("  Mean jumps  = %.2f per trajectory\n",
        result.mean_jumps_per_trajectory);
    std::printf("  Convergence = %s\n",
        result.convergence.reason.c_str());

    return 0;
}

