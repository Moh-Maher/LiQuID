#pragma once

// liquid/simulation/simulation.hpp
// ─────────────────────────────────────────────────────────────────────────────
// The user-facing entry point for LiQuID.
//
// Usage:
//
//   auto sim = liquid::SimulationBuilder{}
//       .hamiltonian(H)
//       .collapse_operator(L)
//       .observe("sigma_z", sz)
//       .num_threads(1)
//       .seed(42)
//       .stop_at_sem(0.01)
//       .min_trajectories(100)
//       .max_trajectories(10000)
//       .dt(1e-3)
//       .build();
//
//   auto result = sim.run(psi0, 0.0, 10.0);
//
//   std::printf("<sz> = %.4f +/- %.4f\n",
//       result.observables[0].mean,
//       result.observables[0].sem);
//
// Design:
//   SimulationBuilder accumulates configuration as plain data.
//   build() validates, constructs the concrete EnsembleManager<...>,
//   and wraps it in a type-erased Simulation handle.
//
//   The virtual dispatch in Simulation::run() is paid ONCE per simulation
//   run — completely negligible relative to simulation cost.
//
//   Phase 2: dense operators only, time-independent, single-thread, RK4.
//   Phase 3+: sparse, time-dependent, OpenMP, RK45 will extend this.
// ─────────────────────────────────────────────────────────────────────────────

#include "liquid/core/types.hpp"
#include "liquid/core/config.hpp"
#include "liquid/linalg/dense.hpp"
#include "liquid/ode/rk4.hpp"
#include "liquid/ode/dopri45.hpp"
#include "liquid/system/open_system.hpp"
#include "liquid/system/open_system_ext.hpp"
#include "liquid/ensemble/ensemble_manager.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace liquid {

// ── Forward declarations ──────────────────────────────────────────────────────

class SimulationBuilder;
class Simulation;

// ── Simulation: type-erased run handle ───────────────────────────────────────

class Simulation {
public:
    // Run from initial state psi0 over [t0, t_final].
    // Blocks until stopping criterion is satisfied.
    ensemble::EnsembleResult run(const StateVector& psi0,
                                  Real t0,
                                  Real t_final) {
        assert(impl_);
        return impl_->run(psi0, t0, t_final);
    }

    // Request early stop (thread-safe).
    void request_stop() { impl_->request_stop(); }

    // ── Public static factories for non-dense systems ─────────────────────
    // SimulationBuilder only supports dense operators.
    // These factories handle arbitrary SystemType / StepperType combinations.

    template<typename SystemType, typename StepperType>
    static Simulation make(
        SystemType                           system,
        std::vector<ensemble::ObservableDef> observables,
        StoppingCriteria                     stopping,
        EnsembleConfig                       config,
        ensemble::AllocatorPolicy            policy =
            ensemble::make_uniform_policy(42))
    {
        struct GenericImpl : Impl {
            SystemType system;
            ensemble::EnsembleManager<SystemType, StepperType> manager;

            GenericImpl(SystemType                           sys,
                        std::vector<ensemble::ObservableDef> obs,
                        StoppingCriteria                     sc,
                        EnsembleConfig                       ec,
                        ensemble::AllocatorPolicy            pol)
                : system(std::move(sys))
                , manager(&this->system, std::move(obs), sc, ec,
                          std::move(pol))
            {}

            ensemble::EnsembleResult run(
                const StateVector& psi0,
                Real t0, Real t_final) override {
                return manager.run(psi0, t0, t_final);
            }
            void request_stop() override { manager.request_stop(); }
        };

        return Simulation(std::make_unique<GenericImpl>(
            std::move(system),
            std::move(observables),
            stopping, config, std::move(policy)));
    }

    // Convenience: sparse time-independent system with DOPRI45
    static Simulation make_sparse_dopri45(
        SparseOpenSystem                     system,
        std::vector<ensemble::ObservableDef> observables,
        StoppingCriteria                     stopping,
        EnsembleConfig                       config,
        ensemble::AllocatorPolicy            policy =
            ensemble::make_uniform_policy(42))
    {
        return make<SparseOpenSystem, ode::DormandPrince45>(
            std::move(system), std::move(observables),
            stopping, config, std::move(policy));
    }

    // Convenience: dense time-independent system with DOPRI45
    static Simulation make_dense_dopri45(
        DenseOpenSystem                      system,
        std::vector<ensemble::ObservableDef> observables,
        StoppingCriteria                     stopping,
        EnsembleConfig                       config,
        ensemble::AllocatorPolicy            policy =
            ensemble::make_uniform_policy(42))
    {
        return make<DenseOpenSystem, ode::DormandPrince45>(
            std::move(system), std::move(observables),
            stopping, config, std::move(policy));
    }

    // Non-copyable (owns unique_ptr<Impl>), moveable.
    Simulation(const Simulation&)            = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&)                 = default;
    Simulation& operator=(Simulation&&)      = default;

private:
    friend class SimulationBuilder;

    struct Impl {
        virtual ensemble::EnsembleResult run(const StateVector&,
                                              Real, Real) = 0;
        virtual void request_stop() = 0;
        virtual ~Impl() = default;
    };

    // Concrete impl for DenseOpenSystem + RK4Stepper (Phase 2)
    struct DenseRK4Impl : Impl {
        DenseOpenSystem                                          system;
        ensemble::EnsembleManager<DenseOpenSystem, ode::RK4Stepper> manager;

        DenseRK4Impl(DenseOpenSystem sys,
                     std::vector<ensemble::ObservableDef> obs,
                     StoppingCriteria sc,
                     EnsembleConfig ec,
                     ensemble::AllocatorPolicy policy)
            : system(std::move(sys))
            , manager(&system, std::move(obs), sc, ec, std::move(policy))
        {}

        ensemble::EnsembleResult run(const StateVector& psi0,
                                      Real t0, Real t_final) override {
            return manager.run(psi0, t0, t_final);
        }

        void request_stop() override { manager.request_stop(); }
    };

    explicit Simulation(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    std::unique_ptr<Impl> impl_;
};

// ── SimulationBuilder ─────────────────────────────────────────────────────────

class SimulationBuilder {
public:
    SimulationBuilder() = default;

    // ── System definition ─────────────────────────────────────────────────────

    SimulationBuilder& hamiltonian(DenseOperator H) {
        H_ = std::move(H);
        return *this;
    }

    SimulationBuilder& collapse_operator(DenseOperator L) {
        L_ops_.push_back(std::move(L));
        return *this;
    }

    // ── Observables ───────────────────────────────────────────────────────────

    // Register a named operator observable: <ψ|O|ψ> / <ψ|ψ>
    SimulationBuilder& observe(std::string name, DenseOperator O) {
        observables_.push_back(
            ensemble::make_operator_observable(std::move(name), std::move(O)));
        return *this;
    }

    // Register a custom observable function
    SimulationBuilder& observe(std::string name, ensemble::ObservableFn fn) {
        observables_.push_back({std::move(name), std::move(fn)});
        return *this;
    }

    // ── Solver ────────────────────────────────────────────────────────────────

    SimulationBuilder& dt(Real dt_initial) {
        config_.propagator.dt_initial = dt_initial;
        return *this;
    }

    SimulationBuilder& tolerances(Real atol, Real rtol) {
        config_.propagator.atol = atol;
        config_.propagator.rtol = rtol;
        return *this;
    }

    // ── Ensemble ──────────────────────────────────────────────────────────────

    SimulationBuilder& num_threads(std::size_t n) {
        config_.num_threads = n;
        return *this;
    }

    SimulationBuilder& seed(Seed s) {
        config_.global_seed = s;
        return *this;
    }

    SimulationBuilder& diagnostics(DiagnosticLevel dl) {
        config_.diag_level = dl;
        return *this;
    }

    // ── Stopping criteria ─────────────────────────────────────────────────────

    SimulationBuilder& stop_at_sem(Real target) {
        stopping_.target_rel_sem = target;
        return *this;
    }

    SimulationBuilder& min_trajectories(std::size_t n) {
        stopping_.min_trajectories = n;
        return *this;
    }

    SimulationBuilder& max_trajectories(std::size_t n) {
        stopping_.max_trajectories = n;
        return *this;
    }

    SimulationBuilder& max_wall_time(Real seconds) {
        stopping_.max_wall_seconds = seconds;
        return *this;
    }

    // ── Custom allocator policy (ML hook) ─────────────────────────────────────

    SimulationBuilder& allocator_policy(ensemble::AllocatorPolicy policy) {
        policy_ = std::move(policy);
        has_custom_policy_ = true;
        return *this;
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    Simulation build() {
        validate();

        // Construct system
        LindbladSet<DenseTag> lindblad(std::move(L_ops_));
        DenseOpenSystem system(std::move(H_), std::move(lindblad));

        // Default observable: if none registered, add <σ_z> as a reminder
        // Actually: zero observables is a user error — enforce it.
        // (validate() already checks this)

        // Construct policy
        ensemble::AllocatorPolicy policy =
            has_custom_policy_
            ? std::move(policy_)
            : ensemble::make_uniform_policy(config_.global_seed);

        auto impl = std::make_unique<Simulation::DenseRK4Impl>(
            std::move(system),
            std::move(observables_),
            stopping_,
            config_,
            std::move(policy));

        return Simulation(std::move(impl));
    }

private:
    void validate() const {
        if (!H_.size()) {
            throw std::runtime_error(
                "SimulationBuilder: no Hamiltonian set. Call .hamiltonian(H).");
        }
        if (L_ops_.empty()) {
            throw std::runtime_error(
                "SimulationBuilder: no collapse operators. "
                "Call .collapse_operator(L) at least once. "
                "For closed systems, use a zero decay rate.");
        }
        if (observables_.empty()) {
            throw std::runtime_error(
                "SimulationBuilder: no observables registered. "
                "Call .observe(name, operator) at least once.");
        }
        // Dimension consistency
        const Dim dim = H_.size();
        for (std::size_t k = 0; k < L_ops_.size(); ++k) {
            if (L_ops_[k].size() != dim) {
                throw std::runtime_error(
                    "SimulationBuilder: collapse operator " +
                    std::to_string(k) + " has dimension " +
                    std::to_string(L_ops_[k].size()) +
                    " but Hamiltonian has dimension " +
                    std::to_string(dim) + ".");
            }
        }
        if (stopping_.min_trajectories > stopping_.max_trajectories) {
            throw std::runtime_error(
                "SimulationBuilder: min_trajectories > max_trajectories.");
        }
    }

    // System
    DenseOperator              H_;
    std::vector<DenseOperator> L_ops_;

    // Observables
    std::vector<ensemble::ObservableDef> observables_;

    // Config
    EnsembleConfig    config_;
    StoppingCriteria  stopping_;

    // Allocator policy
    ensemble::AllocatorPolicy policy_;
    bool                      has_custom_policy_ = false;
};

} // namespace liquid
