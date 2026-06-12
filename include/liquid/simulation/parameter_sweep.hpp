#pragma once

// liquid/simulation/parameter_sweep.hpp
// ─────────────────────────────────────────────────────────────────────────────
// ParameterSweep: run the same simulation at multiple parameter values
// and collect results with full statistical metadata.
//
// Design:
//   The sweep is defined by a builder pattern:
//     - A range of parameter values (any type)
//     - A factory function: parameter_value → Simulation
//     - A shared initial state and time interval
//
//   Execution is sequential by default; parallel with OpenMP if available.
//
//   Results are returned as a vector of SweepPoint, each containing:
//     - The parameter value (as double, for output)
//     - The full EnsembleResult
//     - Wall time for this point
//
// Usage:
//   auto sweep = ParameterSweepBuilder{}
//       .parameter("gamma", {0.1, 0.5, 1.0, 2.0, 5.0})
//       .simulation_factory([](double gamma) {
//           return SimulationBuilder{}
//               .hamiltonian(H)
//               .collapse_operator(make_L(gamma))
//               .observe("sz", sz)
//               .max_trajectories(1000)
//               .build();
//       })
//       .initial_state(psi0)
//       .time_interval(0.0, 10.0)
//       .build();
//
//   SweepResult result = sweep.run();
//   result.save_csv("sweep_gamma.csv");
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/simulation/simulation.hpp"
#include <cassert>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace liquid {

// ── SweepPoint: result for one parameter value ────────────────────────────────

struct SweepPoint {
    double                    param_value;
    std::string               param_name;
    ensemble::EnsembleResult  result;
    double                    wall_time_seconds;
};

// ── SweepResult: all points + output utilities ────────────────────────────────

struct SweepResult {
    std::string              param_name;
    std::vector<SweepPoint>  points;
    double                   total_wall_time_seconds;

    // ── CSV output ────────────────────────────────────────────────────────────
    // Format:
    //   param,obs_name,mean,sem,rel_sem,N_trajectories,wall_time
    //   0.1,sigma_z,-0.2635,0.0031,0.0118,1000,0.234
    //   ...
    void save_csv(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) throw std::runtime_error("Cannot open file: " + path);

        // Header
        std::fprintf(f, "%s", param_name.c_str());
        if (!points.empty() && !points[0].result.observables.empty()) {
            for (const auto& obs : points[0].result.observables) {
                std::fprintf(f, ",%s_mean,%s_sem,%s_rel_sem",
                    obs.name.c_str(), obs.name.c_str(), obs.name.c_str());
            }
        }
        std::fprintf(f, ",N_trajectories,N_failed,mean_jumps_per_traj,"
                        "total_wall_s,traj_wall_s\n");

        // Data rows
        for (const auto& pt : points) {
            std::fprintf(f, "%.10g", pt.param_value);
            for (const auto& obs : pt.result.observables) {
                std::fprintf(f, ",%.10g,%.10g,%.10g",
                    obs.mean, obs.sem, obs.rel_sem);
            }
            std::fprintf(f, ",%zu,%zu,%.6g,%.6g,%.6g\n",
                pt.result.total_trajectories,
                pt.result.failed_trajectories,
                pt.result.mean_jumps_per_trajectory,
                pt.wall_time_seconds,
                pt.result.mean_trajectory_time_seconds);
        }
        std::fclose(f);
    }

    // ── JSON output ───────────────────────────────────────────────────────────
    void save_json(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) throw std::runtime_error("Cannot open file: " + path);

        std::fprintf(f, "{\n");
        std::fprintf(f, "  \"parameter\": \"%s\",\n", param_name.c_str());
        std::fprintf(f, "  \"total_wall_time_s\": %.6g,\n",
                     total_wall_time_seconds);
        std::fprintf(f, "  \"points\": [\n");

        for (std::size_t k = 0; k < points.size(); ++k) {
            const auto& pt = points[k];
            std::fprintf(f, "    {\n");
            std::fprintf(f, "      \"%s\": %.10g,\n",
                         param_name.c_str(), pt.param_value);
            std::fprintf(f, "      \"N_trajectories\": %zu,\n",
                         pt.result.total_trajectories);
            std::fprintf(f, "      \"N_failed\": %zu,\n",
                         pt.result.failed_trajectories);
            std::fprintf(f, "      \"mean_jumps_per_traj\": %.6g,\n",
                         pt.result.mean_jumps_per_trajectory);
            std::fprintf(f, "      \"wall_time_s\": %.6g,\n",
                         pt.wall_time_seconds);
            std::fprintf(f, "      \"observables\": {\n");

            for (std::size_t i = 0; i < pt.result.observables.size(); ++i) {
                const auto& obs = pt.result.observables[i];
                std::fprintf(f, "        \"%s\": {\"mean\": %.10g, "
                                "\"sem\": %.10g, \"rel_sem\": %.10g, "
                                "\"variance\": %.10g}",
                             obs.name.c_str(),
                             obs.mean, obs.sem, obs.rel_sem, obs.variance);
                if (i + 1 < pt.result.observables.size())
                    std::fprintf(f, ",");
                std::fprintf(f, "\n");
            }
            std::fprintf(f, "      },\n");
            std::fprintf(f, "      \"convergence\": \"%s\"\n",
                         pt.result.convergence.reason.c_str());
            std::fprintf(f, "    }");
            if (k + 1 < points.size()) std::fprintf(f, ",");
            std::fprintf(f, "\n");
        }

        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
    }

    // ── Console print ─────────────────────────────────────────────────────────
    void print_table() const {
        if (points.empty()) return;

        // Build column headers
        std::printf("\n%-14s", param_name.c_str());
        for (const auto& obs : points[0].result.observables)
            std::printf("  %-12s %-10s", obs.name.c_str(), "sem");
        std::printf("  %8s  %8s\n", "N_traj", "wall(s)");

        // Separator
        const int width = 14
            + static_cast<int>(points[0].result.observables.size()) * 24
            + 20;
        for (int i = 0; i < width; ++i) std::printf("-");
        std::printf("\n");

        // Rows
        for (const auto& pt : points) {
            std::printf("%-14.6g", pt.param_value);
            for (const auto& obs : pt.result.observables)
                std::printf("  %+12.6f %10.6f", obs.mean, obs.sem);
            std::printf("  %8zu  %8.3f\n",
                pt.result.total_trajectories,
                pt.wall_time_seconds);
        }
        std::printf("\n  Total wall time: %.3f s\n\n",
                    total_wall_time_seconds);
    }
};

// ── ParameterSweep: runnable sweep object ─────────────────────────────────────

class ParameterSweep {
public:
    using SimFactory = std::function<Simulation(double)>;

    ParameterSweep(std::string                param_name,
                   std::vector<double>        param_values,
                   SimFactory                 factory,
                   StateVector                psi0,
                   double                     t0,
                   double                     t_final,
                   std::size_t                num_threads)
        : param_name_(std::move(param_name))
        , param_values_(std::move(param_values))
        , factory_(std::move(factory))
        , psi0_(std::move(psi0))
        , t0_(t0)
        , t_final_(t_final)
        , num_threads_(num_threads)
    {
        assert(!param_values_.empty());
        assert(t_final > t0);
    }

    SweepResult run() const {
        const std::size_t M = param_values_.size();
        std::vector<SweepPoint> points(M);

        auto wall_start = std::chrono::steady_clock::now();

        // Serial execution (OpenMP parallel over parameter points is tricky
        // because each Simulation creates its own OpenMP threads internally —
        // nested parallelism requires careful management).
        // Phase 7: serial over parameter points, each simulation uses its
        // own internal parallelism if configured.
        for (std::size_t k = 0; k < M; ++k) {
            const double pv = param_values_[k];

            auto pt_start = std::chrono::steady_clock::now();

            // Build and run simulation for this parameter value
            Simulation sim = factory_(pv);
            ensemble::EnsembleResult res = sim.run(psi0_, t0_, t_final_);

            const double pt_wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pt_start).count();

            points[k] = SweepPoint{pv, param_name_, std::move(res), pt_wall};
        }

        const double total_wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();

        return SweepResult{param_name_, std::move(points), total_wall};
    }

private:
    std::string          param_name_;
    std::vector<double>  param_values_;
    SimFactory           factory_;
    StateVector          psi0_;
    double               t0_, t_final_;
    std::size_t          num_threads_;
};

// ── ParameterSweepBuilder ─────────────────────────────────────────────────────

class ParameterSweepBuilder {
public:
    ParameterSweepBuilder() = default;

    ParameterSweepBuilder& parameter(std::string name,
                                      std::vector<double> values) {
        param_name_   = std::move(name);
        param_values_ = std::move(values);
        return *this;
    }

    // Linear range: N points from lo to hi (inclusive)
    ParameterSweepBuilder& parameter_range(std::string name,
                                            double lo, double hi,
                                            std::size_t N) {
        param_name_ = std::move(name);
        param_values_.resize(N);
        for (std::size_t k = 0; k < N; ++k)
            param_values_[k] = lo + (hi - lo) * k / (N > 1 ? N - 1 : 1);
        return *this;
    }

    // Log-spaced range: N points from lo to hi (inclusive, base-10)
    ParameterSweepBuilder& parameter_logrange(std::string name,
                                               double lo, double hi,
                                               std::size_t N) {
        param_name_ = std::move(name);
        param_values_.resize(N);
        const double log_lo = std::log10(lo);
        const double log_hi = std::log10(hi);
        for (std::size_t k = 0; k < N; ++k) {
            const double exponent = log_lo + (log_hi - log_lo)
                                    * k / (N > 1 ? N - 1 : 1);
            param_values_[k] = std::pow(10.0, exponent);
        }
        return *this;
    }

    ParameterSweepBuilder& simulation_factory(ParameterSweep::SimFactory f) {
        factory_ = std::move(f);
        return *this;
    }

    ParameterSweepBuilder& initial_state(StateVector psi0) {
        psi0_ = std::move(psi0);
        return *this;
    }

    ParameterSweepBuilder& time_interval(double t0, double t_final) {
        t0_      = t0;
        t_final_ = t_final;
        return *this;
    }

    ParameterSweepBuilder& num_threads(std::size_t n) {
        num_threads_ = n;
        return *this;
    }

    ParameterSweep build() const {
        if (param_name_.empty())
            throw std::runtime_error("ParameterSweepBuilder: no parameter set.");
        if (param_values_.empty())
            throw std::runtime_error("ParameterSweepBuilder: empty parameter range.");
        if (!factory_)
            throw std::runtime_error(
                "ParameterSweepBuilder: no simulation_factory set.");
        if (psi0_.empty())
            throw std::runtime_error(
                "ParameterSweepBuilder: no initial_state set.");
        if (t_final_ <= t0_)
            throw std::runtime_error(
                "ParameterSweepBuilder: t_final must be > t0.");

        return ParameterSweep(param_name_, param_values_, factory_,
                              psi0_, t0_, t_final_, num_threads_);
    }

private:
    std::string                    param_name_;
    std::vector<double>            param_values_;
    ParameterSweep::SimFactory     factory_;
    StateVector                    psi0_;
    double                         t0_{0.0};
    double                         t_final_{1.0};
    std::size_t                    num_threads_{1};
};

} // namespace liquid
