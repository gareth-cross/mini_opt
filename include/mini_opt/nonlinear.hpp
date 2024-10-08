// Copyright 2021 Gareth Cross
#pragma once
#include <optional>
#include <variant>

#include "mini_opt/qp.hpp"
#include "mini_opt/residual.hpp"

namespace mini_opt {

/*
 * Describes a simple [non-]linear least squares problem. The primary cost is a sum-of
 * squared errors.
 *
 * Supports simple linear inequality constraints on the variables.
 *
 * More formally:
 *
 *  min: f_0(x)  [where f_0(x) = (1/2) * h(x)^T * h(x)]
 *
 *  Subject to: diag(a) * x + b >= 0, and
 *  Subject to: g(x) == 0
 *
 * Note that we actually iteratively minimize the first order approximation of f(x):
 *
 *  h(x + dx) = h(x) + J * dx
 *
 * Such that: f_0(x) = (1/2) * h(x)^T * h(x) + (J * dx)^T * h(x) + (J * dx)^T * (J * dx)
 *
 * So in effect, we are solving a quadratic approximation of the nonlinear cost
 * with diagonal inequality constraints on the decision variables.
 */
struct Problem {
  using unique_ptr = std::unique_ptr<Problem>;

  // Problem dimension. (ie. max variable index + 1)
  int dimension;

  // The errors that form the sum of squares part of the cost function.
  std::vector<ResidualBase::unique_ptr> costs;

  // Linear inequality constraints.
  std::vector<LinearInequalityConstraint> inequality_constraints;

  // Nonlinear inequality constraints.
  std::vector<ResidualBase::unique_ptr> equality_constraints;
};

/*
 * Solve constrained non-linear least squares problems using SQP.
 *
 * At each step we approximate the problem as a quadratic with linear equality constraints
 * and inequality constraints about the current linearization point. We do this iteratively
 * until satisfied with the result on the original nonlinear cost.
 */
struct ConstrainedNonlinearLeastSquares {
 public:
  // Parameters, these defaults are largely made up.
  struct Params {
    // Max number of iterations to execute.
    int max_iterations{10};

    // Max number of inner iterations in the QP solver.
    int max_qp_iterations{10};

    // KKT tolerance for the QP inner solver.
    double termination_kkt_tolerance{1.0e-6};

    // Absolute tolerance on squared error to exit.
    double absolute_exit_tol{1.0e-12};

    // Relative exit tolerance. If error[k+1] > error[k] * (1 - tol), stop.
    double relative_exit_tol{1.0e-5};

    // Absolute tolerance on directional derivative of the cost function.
    // If |df(x)| < tol, we have satisfied first order optimality and stop.
    double absolute_first_derivative_tol{1.0e-6};

    // Max # of line-search iterations.
    int max_line_search_iterations{2};

    // Line search method.
    LineSearchStrategy line_search_strategy{LineSearchStrategy::POLYNOMIAL_APPROXIMATION};

    // Value by which to decrease alpha for the backtracking line search.
    double armijo_search_tau{0.8};

    // Initial value of the equality constraint penalty.
    double equality_penalty_initial{0.01};

    // Scale factor when increasing the equality penalty.
    // If µ_new > µ_old, we increase the penalty to µ_new * equality_penalty_scale_factor.
    double equality_penalty_scale_factor{1.01};

    // Parameter ρ used to scale the equality penalty according to equation (18.36).
    // The penalty is scaled up by 1 / (1 - ρ).
    double equality_penalty_rho{0.1};

    // Initial lambda value.
    double lambda_initial{0.0};

    // Initial lambda value on entering the ATTEMPTING_RESTORE_LM state.
    double lambda_failure_init{1.0e-2};

    // Maximum lambda value.
    double max_lambda{1.};

    // Minimum lambda value.
    double min_lambda{0.};
  };

  // Signature of custom retraction operator.
  using Retraction =
      std::function<void(Eigen::VectorXd* const x, const ConstVectorBlock& dx, const double alpha)>;

  // Signature of custom logger.
  using LoggingCallback =
      std::function<bool(const ConstrainedNonlinearLeastSquares& self, const NLSLogInfo& info)>;

  // Construct w/ const pointer to a problem definition.
  explicit ConstrainedNonlinearLeastSquares(const Problem* problem,
                                            Retraction retraction = nullptr);

  /*
   * Iteratively minimize the provided costs. Implements an SQP-like algorithm where
   * we approximate the nonlinear function as a QP, with linearized equality constraints
   * and inequality constraints shifted to the linearization point.
   *
   * We solve that sub-problem with the interior point QP solver, and then run a line search
   * on the descent direction. A merit function combines the quadratic objective with
   * a penalty on violations of the non-linear equality constraints.
   *
   * The solver can also damp the approximated hessian (LM/trust-region) if the optimizer fails to
   * make progress.
   */
  NLSSolverOutputs Solve(const Params& params, const Eigen::VectorXd& variables);

  // Set the callback which will be used for the QP solver.
  template <typename T>
  void SetQPLoggingCallback(T&& cb) {
    // TODO: Get rid of the logging callback altogether. For now I maintain this path for existing
    // unit tests.
    if (QPInteriorPointSolver* ip_solver = std::get_if<QPInteriorPointSolver>(&solver_);
        ip_solver != nullptr)
      ip_solver->SetLoggerCallback(std::forward<T>(cb));
  }

  // Set the callback that will be used for the nonlinear solver.
  template <typename T>
  void SetLoggingCallback(T&& cb) {
    logging_callback_ = std::forward<T>(cb);
  }

  // Get the current linearization point.
  constexpr const Eigen::VectorXd& variables() const noexcept { return variables_; }

  // Evaluate the non-linear error.
  Errors EvaluateNonlinearErrors(const Eigen::VectorXd& vars);

 private:
  // Update candidate_vars w/ a step size of alpha.
  void RetractCandidateVars(double alpha);

  // Linearize and fill the QP w/ the problem definition.
  static Errors LinearizeAndFillQP(const Eigen::VectorXd& variables, double lambda,
                                   const Problem& problem, QP* qp);

  // Solve the QP, and update the step direction `dx_`.
  std::tuple<QPSolverOutputs, std::optional<QPLagrangeMultipliers>> ComputeStepDirection(
      const Params& params);

  // Based on the outcome of the step selection, update lambda and check if
  // we should exit. Returns NONE if no exit is required.
  NLSTerminationState UpdateLambdaAndCheckExitConditions(const Params& params,
                                                         StepSizeSelectionResult step_result,
                                                         const Errors& initial_errors,
                                                         double penalty, double& lambda);

  // Execute a back-tracking search until the cost decreases, or we hit
  // a max number of iterations
  StepSizeSelectionResult SelectStepSize(int max_iterations, double abs_first_derivative_tol,
                                         const Errors& errors_pre,
                                         const DirectionalDerivatives& derivatives, double penalty,
                                         double armijo_c1, LineSearchStrategy strategy,
                                         double backtrack_search_tau);

  // Repeatedly approximate the cost function as a polynomial, and find the minimum.
  double ComputeAlphaPolynomialApproximation(int iteration, double alpha, const Errors& errors_pre,
                                             const DirectionalDerivatives& derivatives,
                                             double penalty) const;

  // Compute first derivative of the QP cost function.
  static DirectionalDerivatives ComputeQPCostDerivative(const QP& qp, const Eigen::VectorXd& dx);

  // Select a new penalty parameter, depending on norm type.
  static double SelectPenalty(const QP& qp, const Eigen::VectorXd& dx,
                              const std::optional<QPLagrangeMultipliers>& lagrange_multipliers,
                              double rho);

  // Solve the quadratic approximation of the cost function for best alpha.
  // Implements equation (3.57/3.58)
  static double QuadraticApproxMinimum(double phi_0, double phi_prime_0, double alpha_0,
                                       double phi_alpha_0);

  // Get the polynomial coefficients c0, c1 from the cubic approximation of the cost.
  // Implements equation after 3.58, returns [a, b]
  static Eigen::Vector2d CubicApproxCoeffs(double phi_0, double phi_prime_0, double alpha_0,
                                           double phi_alpha_0, double alpha_1, double phi_alpha_1);

  // Get the solution of the cubic approximation.
  static double CubicApproxMinimum(double phi_prime_0, const Eigen::Vector2d& ab);

  // Compute the second order correction to the equality constraints.
  static void ComputeSecondOrderCorrection(
      const Eigen::VectorXd& updated_x,
      const std::vector<ResidualBase::unique_ptr>& equality_constraints, QP* qp,
      Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>* solver, Eigen::VectorXd* dx_out);

  const Problem* const p_;

  // Custom retraction operator, optional.
  Retraction custom_retraction_;

  // Storage for the QP representation of the problem.
  QP qp_{};

  // The QP solver itself, which we re-use at each iteration.
  std::variant<QPInteriorPointSolver, QPNullSpaceSolver> solver_{};

  // Parameters (the current linearization point)
  Eigen::VectorXd variables_;
  Eigen::VectorXd candidate_vars_;
  Eigen::VectorXd dx_;

  // Storage for computing errors.
  Eigen::VectorXd error_buffer_;

  // Buffer for line search steps
  std::vector<LineSearchStep> steps_;

  // Current state of the optimization
  OptimizerState state_{OptimizerState::NOMINAL};

  // Logging callback.
  LoggingCallback logging_callback_{};

  friend class ConstrainedNLSTest;
};

}  // namespace mini_opt
