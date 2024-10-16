// Copyright 2021 Gareth Cross
#pragma once
#include <algorithm>
#include <limits>
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

#include <fmt/ostream.h>
#include <Eigen/Core>

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

std::ostream& operator<<(std::ostream& stream, InitialGuessMethod method);

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
  constexpr double Max() const noexcept {
    return std::max<double>({r_dual, r_comp, r_primal_eq, r_primal_ineq});
  }
};

// Represents the state of the QP at a given iteration.
struct QPInteriorPointIteration {
  KKTError kkt_initial{};
  KKTError kkt_final{};
  IPIterationOutputs ip_outputs{};

  QPInteriorPointIteration() noexcept = default;

  constexpr QPInteriorPointIteration(KKTError kkt_initial, KKTError kkt_final,
                                     IPIterationOutputs ip_outputs) noexcept
      : kkt_initial(kkt_initial), kkt_final(kkt_final), ip_outputs(ip_outputs) {}

  // Create a string summary of the iteration.
  std::string ToString() const;
};

// List of possible termination criteria.
enum class QPInteriorPointTerminationState {
  // Achieved numerical threshold `termination_kkt_tol`.
  SATISFIED_KKT_TOL = 0,
  // Hit max number of iterations.
  MAX_ITERATIONS,
};

// ostream for termination states
std::ostream& operator<<(std::ostream& stream, QPInteriorPointTerminationState state);

// Summary of the lagrange multipliers.
struct QPLagrangeMultipliers {
  // Minimum lagrange multiplier.
  double min;
  // The L-infinity (largest absolute value).
  double l_infinity;
};

// Results of QPInteriorPointSolver::Solve.
struct QPInteriorPointSolverOutputs {
  // Termination state of the solver.
  QPInteriorPointTerminationState termination_state;

  // Iterations executed before hitting the termination state.
  std::vector<QPInteriorPointIteration> iterations;

  // Lagrange multipliers on equality constraints, if there are any.
  std::optional<QPLagrangeMultipliers> lagrange_multipliers;

  QPInteriorPointSolverOutputs() noexcept = default;

  QPInteriorPointSolverOutputs(QPInteriorPointTerminationState state,
                               std::vector<QPInteriorPointIteration> iterations,
                               std::optional<QPLagrangeMultipliers> lagrange_multipliers) noexcept
      : termination_state(state),
        iterations(std::move(iterations)),
        lagrange_multipliers(lagrange_multipliers) {}
};

// Result of QPNullSpaceSolver::Solve.
enum class QPNullSpaceTerminationState {
  // Successfully solved the system.
  SUCCESS = 0,
  // The reduced hessian was not positive definite.
  NOT_POSITIVE_DEFINITE,
};

// ostream for QPNullSpaceTerminationState
std::ostream& operator<<(std::ostream& stream, QPNullSpaceTerminationState termination);

// Different methods we can execute the linear search in the nonlinear LS solver.
enum class LineSearchStrategy {
  // Just decrease alpha according to: alpha[k+1] = alpha * scale
  ARMIJO_BACKTRACK = 0,
  // Approximate cost function as a polynomial and compute the minimum.
  POLYNOMIAL_APPROXIMATION = 1,
};

// ostream for LineSearchStrategy
std::ostream& operator<<(std::ostream& stream, LineSearchStrategy strategy);

// State of the nonlinear optimizer.
enum class OptimizerState {
  // Optimizer is making progress.
  NOMINAL = 0,
  // Attempting to restore progress via Levenberg Marquardt.
  ATTEMPTING_RESTORE_LM = 1,
};

std::ostream& operator<<(std::ostream& stream, OptimizerState v);

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

  // Check if either of the values are NaN or Inf.
  bool ContainsInvalidValues() const noexcept {
    return !std::isfinite(f) || !std::isfinite(equality);
  }
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
};

// Result of SelectStepSize
enum class StepSizeSelectionResult {
  // Success, found a valid alpha that decreases cost.
  SUCCESS = 0,
  // Hit max # of iterations.
  MAX_ITERATIONS,
  // Directional derivative of cost is ~= 0 at the current point.
  FIRST_ORDER_SATISFIED,
  // The directional derivative is positive in the direction selected by QP.
  POSITIVE_DERIVATIVE,
  // Failure, the costs became infinite or NaN.
  FAILURE_NON_FINITE_COST,
  // Failure, alpha exited the range (0, 1].
  FAILURE_INVALID_ALPHA,
};

std::ostream& operator<<(std::ostream& stream, StepSizeSelectionResult result);

// Exit condition of the non-linear optimization.
enum class NLSTerminationState {
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
  // The QP approximation was indefinite.
  QP_INDEFINITE,
  // User specified callback indicated exit.
  USER_CALLBACK,
};

inline bool TerminationStateIndicatesSatisfiedTol(const NLSTerminationState state) {
  switch (state) {
    case NLSTerminationState::SATISFIED_ABSOLUTE_TOL:
    case NLSTerminationState::SATISFIED_RELATIVE_TOL:
    case NLSTerminationState::SATISFIED_FIRST_ORDER_TOL: {
      return true;
    }
    default:
      break;
  }
  return false;
}

// ostream for termination states
std::ostream& operator<<(std::ostream& stream, NLSTerminationState state);

// Summary of the eigenvalues of the hessian from the QP.
struct QPEigenvalues {
  // Minimum signed eigenvalue.
  double min;
  // Maximum signed eigenvalue.
  double max;
  // Minimum absolute eigenvalue.
  double abs_min;
};

// The current state of the non-linear optimizer. We generate one of these at every iteration.s
struct NLSIteration {
  NLSIteration(int iteration, OptimizerState optimizer_state, double lambda, Errors errors_initial,
               std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs> qp_outputs,
               std::optional<QPEigenvalues> qp_eigenvalues,
               DirectionalDerivatives directional_derivatives, double penalty,
               StepSizeSelectionResult step_result, std::vector<LineSearchStep> line_search_steps)
      : iteration(iteration),
        optimizer_state(optimizer_state),
        lambda(lambda),
        errors_initial(errors_initial),
        qp_outputs(std::move(qp_outputs)),
        qp_eigenvalues(qp_eigenvalues),
        directional_derivatives(directional_derivatives),
        penalty(penalty),
        step_result(step_result),
        line_search_steps(std::move(line_search_steps)) {}

  // Counter of optimization iteration.
  int iteration;

  // Whether or not we are trying to recover progress using LM.
  OptimizerState optimizer_state;

  // Current lambda value used by LM trust-region method.
  double lambda;

  // Errors at the start of this iteration (prior to computing an update).
  Errors errors_initial;

  // Internal stats from the QP solver.
  std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs> qp_outputs;

  // Optional: Eigenvalues of the approximated hessian, when `log_qp_eigenvalues` is true.
  std::optional<QPEigenvalues> qp_eigenvalues;

  // Directional derivatives of the nonlinear costs.
  DirectionalDerivatives directional_derivatives;

  // Current penalty `μ` on the equality constraints: f(x) + μ * c(x)
  double penalty;

  // Did we succeed in computing a step size that reduces our cost function?
  StepSizeSelectionResult step_result;

  // Log of all steps we tried while searching along the descent direction.
  std::vector<LineSearchStep> line_search_steps;

  // Convert to a string representation.
  std::string ToString(bool use_color = false, bool include_qp = false) const;
};

// Outputs from NLS Solve() method.
struct NLSSolverOutputs {
  NLSTerminationState termination_state;
  std::vector<NLSIteration> iterations;

  NLSSolverOutputs(NLSTerminationState term_state, std::vector<NLSIteration> iterations) noexcept
      : termination_state(term_state), iterations(std::move(iterations)) {}

  // Number of QP iterations over all iterations of the nonlinear optimization.
  std::size_t NumQPIterations() const noexcept;

  // Number of line-search steps over all iterations of the optimization.
  std::size_t NumLineSearchSteps() const noexcept;

  // Number of line-search steps that failed to reduce the cost.
  std::size_t NumFailedLineSearches() const noexcept;

  // Convert to string.
  std::string ToString(bool use_color = false, bool include_qp = false) const;
};

}  // namespace mini_opt

template <>
struct fmt::formatter<mini_opt::QPInteriorPointTerminationState> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mini_opt::QPNullSpaceTerminationState> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mini_opt::InitialGuessMethod> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mini_opt::LineSearchStrategy> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mini_opt::OptimizerState> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mini_opt::StepSizeSelectionResult> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mini_opt::NLSTerminationState> : fmt::ostream_formatter {};
