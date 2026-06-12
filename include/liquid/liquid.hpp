#pragma once

// liquid/liquid.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Master include for LiQuID Phase 1.
// Includes all public headers in dependency order.
// ─────────────────────────────────────────────────────────────────────────────

// Layer 0: Foundation
#include "liquid/core/types.hpp"
#include "liquid/core/config.hpp"
#include "liquid/core/rng.hpp"

// Layer 1: Linear algebra
#include "liquid/linalg/dense.hpp"

// Layer 1: ODE solvers
#include "liquid/ode/rk4.hpp"
#include "liquid/ode/dopri45.hpp"

// Layer 2: Quantum system
#include "liquid/linalg/sparse.hpp"
#include "liquid/system/lindblad.hpp"
#include "liquid/system/lindblad_sparse.hpp"
#include "liquid/system/open_system.hpp"
#include "liquid/system/open_system_ext.hpp"

// Layer 3: Trajectory
#include "liquid/trajectory/trajectory_state.hpp"
#include "liquid/trajectory/mcwf_propagator.hpp"

// Layer 4: Ensemble
#include "liquid/ensemble/running_statistics.hpp"
#include "liquid/ensemble/observable_accumulator.hpp"
#include "liquid/ensemble/convergence_monitor.hpp"
#include "liquid/ensemble/adaptive_allocator.hpp"
#include "liquid/ensemble/ensemble_manager.hpp"

// Layer 5: Simulation
#include "liquid/simulation/simulation.hpp"
#include "liquid/ensemble/smart_policies.hpp"
#include "liquid/ensemble/ml_policy.hpp"
#include "liquid/simulation/parameter_sweep.hpp"
