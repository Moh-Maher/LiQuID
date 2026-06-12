#include "liquid/liquid.hpp"
#include <cmath>
#include <cstdio>
using namespace liquid;
using namespace liquid::ensemble;

int main() {
    const double omega = 1.0;
    const double gamma = 1.0;
    const double T     = 5.0;

    // Hamiltonian
    DenseOperator H(2);
    H(0,0) = Scalar{ omega/2, 0}; H(1,1) = Scalar{-omega/2, 0};

    // Jump operator: sigma_minus = sqrt(gamma) |g><e|
    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0};

    // Observable: sigma_z = |e><e| - |g><g|
    DenseOperator sz(2);
    sz(0,0) = Scalar{1, 0}; sz(1,1) = Scalar{-1, 0};

    // Build and run
    Simulation sim = SimulationBuilder{}
        .hamiltonian(std::move(H))
        .collapse_operator(std::move(L))
        .observe("sigma_z", std::move(sz))
        .seed(42)
        .dt(1e-3)
        .stop_at_sem(0.05)          // 0.5% SEM target
        .min_trajectories(200)
        .max_trajectories(20000)
        .build();

    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};  // excited state

    auto result = sim.run(psi0, 0.0, T);

    const double exact = 2.0 * std::exp(-gamma * T) - 1.0;
    const double error = std::abs(result.observables[0].mean - exact);
    const double nsig  = error / result.observables[0].sem;

    std::printf("=== Two-Level Spontaneous Decay ===\n");
    std::printf("  <sigma_z>  MCWF : %+.6f +/- %.6f\n",
        result.observables[0].mean,
        result.observables[0].sem);
    std::printf("  <sigma_z>  Exact: %+.6f\n", exact);
    std::printf("  Deviation        : %.2f sigma\n", nsig);
    std::printf("  Trajectories     : %zu\n", result.total_trajectories);
    std::printf("  Mean jumps/traj  : %.2f\n",
        result.mean_jumps_per_trajectory);

    // Validation: result must be within 3 sigma of exact
    return (nsig < 3.0) ? 0 : 1;
}
