#include "liquid/liquid.hpp"
#include <cmath>
#include <cstdio>
using namespace liquid;
using namespace liquid::ensemble;

int main() {
    const double gamma = 1.0;
    const double T_ss  = 15.0;  // sufficient time to reach steady state

    // Initial state: excited |e>
    StateVector psi0(2);
    psi0[0] = Scalar{1.0, 0.0};

    // Analytic steady-state formula
    auto exact_ss = [&](double Omega) {
        const double r = Omega / gamma;
        return 4.0 * r * r / (1.0 + 8.0 * r * r);
    };

    // Define the sweep over Rabi frequency Omega
    auto sweep = ParameterSweepBuilder{}
        .parameter("Omega", {0.1, 0.5, 1.0, 2.0})
        .simulation_factory([gamma](double Omega) {
            // H = Omega * sigma_x  (resonant drive)
            DenseOperator H(2);
            H(0, 1) = Scalar{Omega, 0.0};
            H(1, 0) = Scalar{Omega, 0.0};

            // L = sqrt(gamma) * sigma_minus
            DenseOperator L(2);
            L(1, 0) = Scalar{std::sqrt(gamma), 0.0};

            // Observable: rho_ee = |e><e|
            DenseOperator rho_ee_op(2);
            rho_ee_op(0, 0) = Scalar{1.0, 0.0};

            return SimulationBuilder{}
                .hamiltonian(std::move(H))
                .collapse_operator(std::move(L))
                .observe("rho_ee", std::move(rho_ee_op))
                .seed(42)
                .dt(5e-4)
                .stop_at_sem(0.02)          // 2% relative SEM at each point
                .min_trajectories(500)
                .max_trajectories(5000)
                .build();
        })
        .initial_state(psi0)
        .time_interval(0.0, T_ss)
        .build();

    SweepResult result = sweep.run();

    // Print table with comparison to analytic formula
    std::printf("%-10s  %-12s  %-12s  %-10s  %-8s  %s\n",
        "Omega", "rho_ee(MCWF)", "rho_ee(exact)", "rel_err(%)", "N", "converged");
    std::printf("%s\n", std::string(72, '-').c_str());

    for (const auto& pt : result.points) {
        const auto& obs   = pt.result.observables[0];
        const double exact = exact_ss(pt.param_value);
        const double rel_err = 100.0 * std::abs(obs.mean - exact) / exact;

        std::printf("%-10.4f  %-12.4f  %-12.4f  %-10.2f  %-8zu  %s\n",
            pt.param_value,
            obs.mean,
            exact,
            rel_err,
            pt.result.total_trajectories,
            pt.result.convergence.reason.c_str());
    }

    // Save full results (CSV columns: param,obs,mean,sem,rel_sem,N,...)
    result.save_csv("sweep_Omega.csv");
    std::printf("\nSaved: sweep_Omega.csv\n");

    return 0;
}

