#include "liquid/liquid.hpp"
#include <cmath>
#include <cstdio>
using namespace liquid;
using namespace liquid::ensemble;

// Analytic steady-state excited population for H = Omega*sigma_x,
// L = sqrt(gamma)*sigma_minus.
static double analytic_rho_ee_ss(double Omega, double gamma) {
    const double r = Omega / gamma;
    return 4.0 * r * r / (1.0 + 8.0 * r * r);
}

// Build the rotating-frame driven open system
static DenseOpenSystem make_driven_system(double Omega, double gamma) {
    // H = Omega * sigma_x  (resonant drive, rotating frame)
    DenseOperator H(2);
    H(0, 1) = Scalar{Omega, 0.0};
    H(1, 0) = Scalar{Omega, 0.0};

    // L = sqrt(gamma) * sigma_minus
    DenseOperator L(2);
    L(1, 0) = Scalar{std::sqrt(gamma), 0.0};

    std::vector<DenseOperator> ops;
    ops.push_back(std::move(L));
    return DenseOpenSystem(std::move(H), LindbladSet<DenseTag>(std::move(ops)));
}

int main() {
    const double gamma  = 1.0;
    const double T_ss   = 20.0;  // long enough to reach steady state
    const int    N_traj = 3000;

    // -- Regime 1: weak drive (Omega = 0.1*gamma) --------------------------
    {
        const double Omega     = 0.1 * gamma;
        const double exact_ss  = analytic_rho_ee_ss(Omega, gamma);

        auto sys = make_driven_system(Omega, gamma);

        PropagatorConfig cfg;
        cfg.dt_initial = 5e-4;  // small dt: Rabi period = pi/(Omega) ~ 31

        double sum_rho = 0.0, sum_rho2 = 0.0;
        StateVector psi0(2);
        psi0[0] = Scalar{1.0, 0.0};  // start excited

        for (int traj = 0; traj < N_traj; ++traj) {
            liquid::ode::RK4Stepper stepper(cfg.dt_initial);
            liquid::trajectory::MCWFPropagator<
                DenseOpenSystem, liquid::ode::RK4Stepper> prop(&sys, stepper, cfg);

            auto ts = liquid::trajectory::make_trajectory_state(
                psi0, 0.0, T_ss,
                static_cast<TrajId>(traj), 54321ULL,
                DiagnosticLevel::None);

            prop.run_to_completion(ts);

            const Real ns      = ts.core.psi.norm_sq();
            const double rho_e = ns > 1e-30
                ? std::norm(ts.core.psi[0]) / ns
                : 0.0;
            sum_rho  += rho_e;
            sum_rho2 += rho_e * rho_e;
        }

        const double mean = sum_rho / N_traj;
        const double var  = sum_rho2 / N_traj - mean * mean;
        const double sem  = std::sqrt(std::max(0.0, var) / N_traj);

        std::printf("Weak drive (Omega/gamma = %.2f):\n", Omega / gamma);
        std::printf("  rho_ee(MCWF)  = %.4f +/- %.4f\n", mean, sem);
        std::printf("  rho_ee(exact) = %.4f\n", exact_ss);
        std::printf("  |error|       = %.4f  (%.1f sigma)\n\n",
            std::abs(mean - exact_ss),
            std::abs(mean - exact_ss) / sem);
    }

    // -- Regime 2: strong drive (Omega = gamma) ----------------------------
    {
        const double Omega     = 1.0 * gamma;
        const double exact_ss  = analytic_rho_ee_ss(Omega, gamma);

        auto sys = make_driven_system(Omega, gamma);

        PropagatorConfig cfg;
        cfg.dt_initial = 5e-4;

        double sum_rho = 0.0, sum_rho2 = 0.0;
        StateVector psi0(2);
        psi0[0] = Scalar{1.0, 0.0};

        for (int traj = 0; traj < N_traj; ++traj) {
            liquid::ode::RK4Stepper stepper(cfg.dt_initial);
            liquid::trajectory::MCWFPropagator<
                DenseOpenSystem, liquid::ode::RK4Stepper> prop(&sys, stepper, cfg);

            auto ts = liquid::trajectory::make_trajectory_state(
                psi0, 0.0, T_ss,
                static_cast<TrajId>(traj), 99999ULL,
                DiagnosticLevel::None);

            prop.run_to_completion(ts);

            const Real ns      = ts.core.psi.norm_sq();
            const double rho_e = ns > 1e-30
                ? std::norm(ts.core.psi[0]) / ns
                : 0.0;
            sum_rho  += rho_e;
            sum_rho2 += rho_e * rho_e;
        }

        const double mean = sum_rho / N_traj;
        const double var  = sum_rho2 / N_traj - mean * mean;
        const double sem  = std::sqrt(std::max(0.0, var) / N_traj);

        std::printf("Strong drive (Omega/gamma = %.2f):\n", Omega / gamma);
        std::printf("  rho_ee(MCWF)  = %.4f +/- %.4f\n", mean, sem);
        std::printf("  rho_ee(exact) = %.4f  [saturation: 1/3 limit]\n", exact_ss);
        std::printf("  |error|       = %.4f  (%.1f sigma)\n",
            std::abs(mean - exact_ss),
            std::abs(mean - exact_ss) / sem);
    }

    return 0;
}
