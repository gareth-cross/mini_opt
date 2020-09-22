// Copyright 2020 Gareth Cross
#pragma once
#include "qp.hpp"
#include "residual.hpp"

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
  std::size_t dimension;

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
 * and inequality constraints about the current lineariation point. We do this iteratively
 * until satisfied with the result on the original nonlinear cost.
 */
struct ConstrainedNonlinearLeastSquares {
 public:
  // Construct w/ const pointer to a problem definition.
  explicit ConstrainedNonlinearLeastSquares(const Problem* const problem);

  // Linearize and take a step.
  void LinearizeAndSolve(const double lambda = 0);

  // Set the variables.
  void SetVariables(const Eigen::VectorXd& variables) { variables_ = variables; }

  // Set the callback which will be used for the QP solver.
  template <typename T>
  void SetQPLoggingCallback(T cb) {
    qp_logger_callback_ = cb;
  }

  // Get the current linearization point.
  const Eigen::VectorXd& variables() const { return variables_; }

 private:
  const Problem* const p_;

  // Storage for the QP representation of the problem.
  QP qp_{};

  // Parameters (the current linearization point)
  Eigen::VectorXd variables_;

  // Callback we pass to the QP solver
  QPInteriorPointSolver::LoggingCallback qp_logger_callback_{};
};

}  // namespace mini_opt