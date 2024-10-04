// Copyright 2021 Gareth Cross
#pragma once
#include <Eigen/Core>
#include <limits>
#include <ostream>
#include <vector>

namespace mini_opt {

// Fwd declare.
struct ConstrainedNonlinearLeastSquares;
struct QPInteriorPointSolver;

// Blocks into dynamic vectors.
using ConstVectorBlock = Eigen::VectorBlock<const Eigen::VectorXd>;
using VectorBlock = Eigen::VectorBlock<Eigen::VectorXd>;

// Possible methods of picking the barrier parameter, `mu`.
enum class BarrierStrategy {
  // Set mu = sigma * (s^T * z) / M, where sigma is a scalar decreased at each iteration.
  COMPLEMENTARITY = 0,
  // Staring from the initial mu, decrease by fixed sigma.
  FIXED_DECREASE,
  // Use Predictor corrector algorithm to select `mu`.
  PREDICTOR_CORRECTOR,
};

// Possible mechanisms of generating the QP initial guess.
enum class InitialGuessMethod {
  // Guesses zero for x, and initialize inequalities to match initial mu.
  NAIVE = 0,
  // Solve the equality constrained problem, then clamp to the feasible region.
  SOLVE_EQUALITY_CONSTRAINED,
  // Do nothing, we expect the variables to be initialized externally.
  USER_PROVIDED,
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

// L2 norm of errors in the first order KKT conditions.
// At a point that satisfies the conditions, these should all be zero.
struct KKTError {
  double r_dual{0};         // The dual objective: Gx + c - A_e^T * y - A_i * z
  double r_comp{0};         // Complementarity condition: s^T * z
  double r_primal_eq{0};    // Primal equality constraint: A_e*x + b_e
  double r_primal_ineq{0};  // Primal inequality constraint: A_i*x + b_i - s

  // Max of all elements.
  double Max() const;
};

// List of possible termination criteria.
enum class QPTerminationState {
  SATISFIED_KKT_TOL = 0,
  MAX_ITERATIONS,
};

// ostream for termination states
std::ostream& operator<<(std::ostream& stream, const QPTerminationState& state);

// Results of QPInteriorPointSolver::Solve.
struct QPSolverOutputs {
  QPTerminationState termination_state;
  int num_iterations;

  constexpr QPSolverOutputs(QPTerminationState state, int iters) noexcept
      : termination_state(state), num_iterations(iters) {}
};

// Different methods we can execute the linear search in the nonlinear LS solver.
enum class LineSearchStrategy {
  // Just decrease alpha according to: alpha[k+1] = alpha * scale
  ARMIJO_BACKTRACK = 0,
  // Approximate cost function as a polynomial and compute the minimum.
  POLYNOMIAL_APPROXIMATION = 1,
};

// ostream for LineSearchStrategy
std::ostream& operator<<(std::ostream& stream, const LineSearchStrategy& strategy);

// State of the nonlinear optimizer.
enum class OptimizerState {
  // Optimizer is making progress.
  NOMINAL = 0,
  // Attempting to restore progress via Levenberg Marquardt.
  ATTEMPTING_RESTORE_LM = 1,
};

std::ostream& operator<<(std::ostream& stream, const OptimizerState& v);

// Total squared errors from a nonlinear optimization.
struct Errors {
  // Sum of squared costs.
  double f{0};

  // Error in the non-linear equality constraints.
  double equality{0};

  // Total weighted cost function.
  constexpr double Total(double penalty) const noexcept { return f + penalty * equality; }

  // L-infinity norm
  constexpr double LInfinity() const noexcept { return std::max(f, equality); }
};

// Derivatives of `Errors`.
struct DirectionalDerivatives {
  // Derivative of sum of squared costs wrt alpha.
  double d_f;

  // Derivative of the equality constraints wrt alpha.
  double d_equality;

  // Total error after applying penalty to the equality constraints.
  constexpr double Total(double penalty) const noexcept { return d_f + penalty * d_equality; }

  // L-infinity norm
  constexpr double LInfinity() const noexcept {
    return std::max(std::abs(d_f), std::abs(d_equality));
  }
};

// Pair together a step size and the error achieved.
struct LineSearchStep {
  // Value of alpha this was computed at [0, 1]
  double alpha;

  // Cost function value at that step.
  Errors errors;

  constexpr LineSearchStep(double a, Errors e) noexcept : alpha(a), errors(e) {}
};

// Result of SelectStepSize
enum class StepSizeSelectionResult {
  // Success, found a valid alpha that decreases cost.
  SUCCESS = 0,
  // Failure, hit max # of iterations.
  FAILURE_MAX_ITERATIONS = 1,
  // Failure, directional derivative of cost is ~= 0 at the current point.
  FAILURE_FIRST_ORDER_SATISFIED = 2,
  // Failure, the directional derivative is positive in the direction selected by QP.
  FAILURE_POSITIVE_DERIVATIVE = 3,
};

std::ostream& operator<<(std::ostream& stream, const StepSizeSelectionResult& result);

// Exit condition of the non-linear optimization.
enum class NLSTerminationState {
  NONE = -1,
  // Hit max number of iterations.
  MAX_ITERATIONS,
  // Satisfied absolute tolerance on the cost function.
  SATISFIED_ABSOLUTE_TOL,
  // Satisfied relative tolerance on decrease of the cost function.
  SATISFIED_RELATIVE_TOL,
  // Satisfied tolerance on magnitude of the first derivative.
  SATISFIED_FIRST_ORDER_TOL,
  // Hit max lambda value while failing to decrease cost.
  MAX_LAMBDA,
  // User specified callback indicated exit.
  USER_CALLBACK,
};

// Outputs from NLS Solve() method.
struct NLSSolverOutputs {
  NLSTerminationState termination_state;
  int num_iterations;
  int num_qp_iterations;

  // Construct.
  constexpr NLSSolverOutputs(NLSTerminationState term_state, int num_iters,
                             int num_qp_iters) noexcept
      : termination_state(term_state), num_iterations(num_iters), num_qp_iterations(num_qp_iters) {}
};

// ostream for termination states
std::ostream& operator<<(std::ostream& stream, const NLSTerminationState& state);

// Summary of the eigenvalues of the hessian from the QP.
struct QPEigenvalues {
  // Minimum signed eigenvalue.
  double min;
  // Maximum signed eigenvalue.
  double max;
  // Minimum absolute eigenvalue.
  double abs_min;
};

// Details for the log, the current state of the non-linear optimizer.
struct NLSLogInfo {
  NLSLogInfo(int iteration, OptimizerState optimizer_state, double lambda, Errors errors_initial,
             QPSolverOutputs qp, QPEigenvalues qp_eigenvalues,
             DirectionalDerivatives directional_derivatives, double penalty,
             StepSizeSelectionResult step_result, const std::vector<LineSearchStep>& steps,
             NLSTerminationState termination_state)
      : iteration(iteration),
        optimizer_state(optimizer_state),
        lambda(lambda),
        errors_initial(errors_initial),
        qp_term_state(qp),
        qp_eigenvalues(qp_eigenvalues),
        directional_derivatives(directional_derivatives),
        penalty(penalty),
        step_result(step_result),
        steps(steps),
        termination_state(termination_state) {}

  int iteration;
  OptimizerState optimizer_state;
  double lambda;
  Errors errors_initial;
  QPSolverOutputs qp_term_state;
  QPEigenvalues qp_eigenvalues;
  DirectionalDerivatives directional_derivatives;
  double penalty;
  StepSizeSelectionResult step_result;
  const std::vector<LineSearchStep>& steps;
  NLSTerminationState termination_state;
};

}  // namespace mini_opt
