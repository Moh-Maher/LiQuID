#include "liquid/liquid.hpp"
#include <cmath>
#include <cstdio>
using namespace liquid;
using namespace liquid::ensemble;

static DenseOperator make_H(double omega) {
    DenseOperator H(2);
    H(0,0) = Scalar{ omega/2.0, 0.0};
    H(1,1) = Scalar{-omega/2.0, 0.0};
    return H;
}
static DenseOperator make_L(double gamma) {
    DenseOperator L(2);
    L(1,0) = Scalar{std::sqrt(gamma), 0.0};
    return L;
}
static DenseOperator make_sz() {
    DenseOperator sz(2);
    sz(0,0) = Scalar{1.0, 0.0};
    sz(1,1) = Scalar{-1.0, 0.0};
    return sz;
}

int main() {
    const double omega = 1.0;
    const double gamma = 1.0;
    const double T     = 1.0;

    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};

    const double exact = 2.0 * std::exp(-gamma * T) - 1.0;

    // -- 1. SEM-based stopping (1% relative SEM target) ---------------------
    {
        Simulation sim = SimulationBuilder{}
            .hamiltonian(make_H(omega))
            .collapse_operator(make_L(gamma))
            .observe("sz", make_sz())
            .seed(42)
            .dt(1e-3)
            .stop_at_sem(0.01)
            .min_trajectories(50)
            .max_trajectories(50000)
            .build();

        auto result = sim.run(psi0, 0.0, T);
        const auto& obs = result.observables[0];

        std::printf("=== SEM-based stopping ===\n");
        std::printf("  <sz> = %.4f +/- %.4f  rel_sem = %.4f  N = %zu\n",
            obs.mean, obs.sem, obs.rel_sem, result.total_trajectories);
        std::printf("  Exact = %.4f  |error| = %.4f sigma\n",
            exact, std::abs(obs.mean - exact) / obs.sem);
        std::printf("  Decision: %s\n\n",
            result.convergence.reason.c_str());
    }

    // -- 2. Wall-clock budget (0.3 s limit, impossibly tight SEM) -----------
    {
        Simulation sim = SimulationBuilder{}
            .hamiltonian(make_H(omega))
            .collapse_operator(make_L(gamma))
            .observe("sz", make_sz())
            .seed(999)
            .dt(1e-3)
            .stop_at_sem(1e-9)          // impossible: triggers budget
            .min_trajectories(10)
            .max_trajectories(1000000)
            .max_wall_time(0.3)         // 300 ms hard limit
            .build();

        auto result = sim.run(psi0, 0.0, T);

        std::printf("=== Wall-clock budget ===\n");
        std::printf("  Decision  : %s\n",
            result.convergence.reason.c_str());
        std::printf("  N achieved: %zu  wall = %.3f s\n\n",
            result.total_trajectories,
            result.total_wall_time_seconds);
    }

    // -- 3. SEM scales as 1/sqrt(N): verify with three ensemble sizes --------
    {
        const std::size_t sizes[3] = {100, 400, 1600};
        double sems[3];

        std::printf("=== SEM scaling: expected SEM ~ 1/sqrt(N) ===\n");
        for (int k = 0; k < 3; ++k) {
            Simulation sim = SimulationBuilder{}
                .hamiltonian(make_H(omega))
                .collapse_operator(make_L(gamma))
                .observe("sz", make_sz())
                .seed(static_cast<Seed>(k * 777 + 42))
                .dt(1e-3)
                .min_trajectories(sizes[k])
                .max_trajectories(sizes[k])
                .build();

            auto result  = sim.run(psi0, 0.0, T);
            sems[k]      = result.observables[0].sem;
            std::printf("  N = %4zu  SEM = %.5f\n", sizes[k], sems[k]);
        }

        // Ratios should be ~2 (since N quadruples each time)
        std::printf("  SEM ratio N=100/400  = %.3f  (expected ~2.0)\n",
            sems[0] / sems[1]);
        std::printf("  SEM ratio N=400/1600 = %.3f  (expected ~2.0)\n",
            sems[1] / sems[2]);
    }

    return 0;
}

