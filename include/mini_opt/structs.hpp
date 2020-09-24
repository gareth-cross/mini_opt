// Copyright 2020 Gareth Cross
#pragma once
#include <numeric>

namespace mini_opt {

// Fwd declare.
struct QPInteriorPointSolver;

// Possible methods of picking the barrier parameter, `mu`.
enum class BarrierStrategy {
  // Set mu = sigma * (s^T * z) / M, where sigma is a scalar decreased at each iteration.
  COMPLEMENTARITY = 0,
  // Staring from the initial mu, decrease by fixed sigma.
  FIXED_DECREASE,
  // Use Predictor corrector algorithm to select `mu`.
  PREDICTOR_CORRECTOR,
};

struct AlphaValues {
  // Alpha in the primal variables (x an s), set to 1 if we have no s
  double primal{1.};
  // Alpha in the dual variables (y and z), set to 1 if we have no z
  double dual{1.};
};

// Some intermediate values we compute during the iteration of the interior point solver.
struct IPIterationOutputs {
  // The barrier parameter on this iteration.
  double mu{0.};
  // Alpha as defined by equation (19.9).
  AlphaValues alpha{};
  // Optional alpha values computing during the MPC probing step.
  AlphaValues alpha_probe{std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()};
  // Mu as determined by taking a step with mu=0, then evaluating equation (19.21).
  // Only relevant in predictor-corrector mode.
  double mu_affine{std::numeric_limits<double>::quiet_NaN()};
};

// Squared errors in the first order KKT conditions.
// At a point that satisfies the conditions, these should all be zero.
struct KKTError {
  double r_dual{0};         // The dual objective: Gx + c - A_e^T * y - A_i * z
  double r_comp{0};         // Complementarity condition: s^T * z
  double r_primal_eq{0};    // Primal equality constraint: A_e*x + b_e
  double r_primal_ineq{0};  // Primal inequality constraint: A_i*x + b_i - s

  // Total squared error.
  double Total() const { return r_dual + r_comp + r_primal_eq + r_primal_ineq; }
};

// List of possible termination criteria.
enum class QPTerminationState {
  SATISFIED_KKT_TOL = 0,
  MAX_ITERATIONS,
};

// Results of QPInteriorPointSolver::Solve.
struct QPSolverOutputs {
  QPTerminationState termination_state;
  int num_iterations;

  QPSolverOutputs(const QPTerminationState state, const int iters)
      : termination_state(state), num_iterations(iters) {}
};

// Total squared errors from a nonlinear optimization.
struct Errors {
  // Sum of squared costs.
  double total_l2{0};
  // Squared error in the non-linear equality constraints.
  double equality_l2{0};
};

}  // namespace mini_opt