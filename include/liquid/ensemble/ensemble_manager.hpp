#pragma once

// liquid/ensemble/ensemble_manager.hpp — Phase 3 (OpenMP parallel)

#include "liquid/core/types.hpp"
#include "liquid/core/config.hpp"
#include "liquid/linalg/dense.hpp"
#include "liquid/ode/rk4.hpp"
#include "liquid/trajectory/trajectory_state.hpp"
#include "liquid/trajectory/mcwf_propagator.hpp"
#include "liquid/ensemble/running_statistics.hpp"
#include "liquid/ensemble/observable_accumulator.hpp"
#include "liquid/ensemble/convergence_monitor.hpp"
#include "liquid/ensemble/adaptive_allocator.hpp"
#include <cassert>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace liquid::ensemble {

struct ObservableResult {
    std::string name;
    Real        mean;
    Real        variance;
    Real        sem;
    Real        rel_sem;
};

struct EnsembleResult {
    std::vector<ObservableResult> observables;
    ConvergenceReport             convergence;
    std::size_t total_trajectories;
    std::size_t failed_trajectories;
    Real        total_wall_time_seconds;
    Real        mean_trajectory_time_seconds;
    Real        total_jumps;
    Real        mean_jumps_per_trajectory;
    std::size_t num_threads_used;
};

template<typename SystemType, typename StepperType>
class EnsembleManager {
public:
    using PropType = liquid::trajectory::MCWFPropagator<SystemType, StepperType>;

    EnsembleManager(
        const SystemType*          system,
        std::vector<ObservableDef> observables,
        StoppingCriteria           stopping,
        EnsembleConfig             config,
        AllocatorPolicy            policy)
        : system_(system)
        , obs_defs_(std::move(observables))
        , global_accum_(obs_defs_)
        , monitor_(stopping)
        , allocator_(std::move(policy))
        , config_(config)
    {
        assert(system_ != nullptr);
    }

    EnsembleResult run(const StateVector& psi0, Real t0, Real t_final) {
        global_accum_.reset();
        total_jumps_  = 0.0;
        failed_count_ = 0;

        auto wall_start = std::chrono::steady_clock::now();

        int n_threads = static_cast<int>(config_.num_threads);
#ifdef _OPENMP
        if (n_threads < 1) n_threads = 1;
#else
        n_threads = 1;
#endif

        const std::size_t batch = std::max(
            static_cast<std::size_t>(n_threads) * 4UL, std::size_t(20));

        ConvergenceReport report;
        report.decision = StoppingDecision::Continue;
        TrajId next_id = 0;

        while (report.decision == StoppingDecision::Continue) {
            const std::size_t done = global_accum_.statistics().count();
            const std::size_t max_rem =
                monitor_.criteria().max_trajectories - done;
            const std::size_t this_batch = std::min(batch, max_rem);
            if (this_batch == 0) break;

            // Pre-generate seeds via the allocator policy.
            // This routes through AllocatorPolicy so ML and custom policies
            // are actually called (not bypassed by inline seed generation).
            std::vector<TrajId> ids(this_batch);
            std::vector<Seed>   seeds(this_batch);
            {
                const Real elapsed_now = elapsed_seconds(wall_start);
                for (std::size_t b = 0; b < this_batch; ++b) {
                    EnsembleSummary alloc_summary = build_summary(elapsed_now);
                    auto dec  = allocator_.decide(alloc_summary);
                    ids[b]    = dec.new_traj_id;
                    seeds[b]  = dec.new_seed;
                    // Keep next_id in sync so build_summary stays accurate
                    next_id   = ids[b] + 1;
                }
            }

            const int actual_threads = n_threads;
            std::vector<ObservableAccumulator> local_accums(
                actual_threads, ObservableAccumulator(obs_defs_));
            std::vector<Real>        local_jumps(actual_threads, 0.0);
            std::vector<std::size_t> local_failed(actual_threads, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(actual_threads) \
        if(actual_threads > 1)
#endif
            for (int b = 0; b < static_cast<int>(this_batch); ++b) {
                int tid = 0;
#ifdef _OPENMP
                if (actual_threads > 1) tid = omp_get_thread_num();
#endif
                StepperType stepper;
                stepper.reset(config_.propagator.dt_initial);
                PropType prop(system_, stepper, config_.propagator);

                auto ts = liquid::trajectory::make_trajectory_state(
                    psi0, t0, t_final,
                    ids[b], seeds[b],
                    config_.diag_level);

                prop.run_to_completion(ts);

                if (ts.is_failed()) {
                    ++local_failed[tid];
                } else {
                    local_accums[tid].accumulate(ts.core.psi);
                    if (ts.has_diagnostics()) {
                        local_jumps[tid] +=
                            static_cast<Real>(ts.diag().total_jumps);
                    }
                }
            }

            for (int t = 0; t < actual_threads; ++t) {
                global_accum_.merge(local_accums[t]);
                total_jumps_  += local_jumps[t];
                failed_count_ += local_failed[t];
            }

            const Real elapsed = elapsed_seconds(wall_start);
            report = monitor_.evaluate(global_accum_, elapsed);
        }

        const Real total_wall = elapsed_seconds(wall_start);
        return build_result(report, total_wall,
                            static_cast<std::size_t>(n_threads));
    }

    const ObservableAccumulator& accumulator() const noexcept {
        return global_accum_;
    }
    const ConvergenceMonitor& monitor() const noexcept { return monitor_; }
    void request_stop() { monitor_.request_stop(); }

private:
    static Real elapsed_seconds(
        const std::chrono::steady_clock::time_point& s) {
        return std::chrono::duration<Real>(
            std::chrono::steady_clock::now() - s).count();
    }

    EnsembleSummary build_summary(Real elapsed) const {
        EnsembleSummary s;
        s.trajectories_completed = global_accum_.statistics().count();
        s.trajectories_running   = 0;
        s.trajectories_failed    = failed_count_;
        s.current_rel_sem =
            (global_accum_.statistics().count() >= 2)
            ? global_accum_.statistics().max_relative_sem()
            : std::numeric_limits<Real>::infinity();
        s.wall_time_elapsed   = elapsed;
        s.completed_summaries = nullptr;
        return s;
    }

    EnsembleResult build_result(const ConvergenceReport& report,
                                 Real total_wall,
                                 std::size_t n_threads) const {
        EnsembleResult result;
        result.convergence             = report;
        result.total_trajectories      = global_accum_.statistics().count();
        result.failed_trajectories     = failed_count_;
        result.total_wall_time_seconds = total_wall;
        result.total_jumps             = total_jumps_;
        result.num_threads_used        = n_threads;

        const std::size_t N = result.total_trajectories;
        result.mean_trajectory_time_seconds =
            (N > 0) ? (total_wall / static_cast<Real>(N)) : 0.0;
        result.mean_jumps_per_trajectory =
            (N > 0) ? (total_jumps_ / static_cast<Real>(N)) : 0.0;

        const auto& stats = global_accum_.statistics();
        result.observables.resize(global_accum_.num_observables());
        for (std::size_t i = 0; i < global_accum_.num_observables(); ++i) {
            auto& obs    = result.observables[i];
            obs.name     = global_accum_.name(i);
            obs.mean     = (N > 0) ? stats.mean(i)     : 0.0;
            obs.variance = (N > 1) ? stats.variance(i) : 0.0;
            obs.sem      = (N > 1) ? stats.sem(i)
                                   : std::numeric_limits<Real>::infinity();
            obs.rel_sem  = (N > 1) ? stats.relative_sem(i)
                                   : std::numeric_limits<Real>::infinity();
        }
        return result;
    }

    const SystemType*           system_;
    std::vector<ObservableDef>  obs_defs_;
    ObservableAccumulator       global_accum_;
    ConvergenceMonitor          monitor_;
    AdaptiveAllocator           allocator_;
    EnsembleConfig              config_;
    Real                        total_jumps_{0.0};
    std::size_t                 failed_count_{0};
    std::vector<TrajectorySummary> traj_summaries_;
};

} // namespace liquid::ensemble
