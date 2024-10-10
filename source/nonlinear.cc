// Copyright 2021 Gareth Cross
#include "mini_opt/nonlinear.hpp"

#include <utility>

#include <fmt/ostream.h>
#include <Eigen/Dense>  //  for inverse()

namespace mini_opt {

ConstrainedNonlinearLeastSquares::ConstrainedNonlinearLeastSquares(const Problem* const problem,
                                                                   Retraction retraction)
    : p_(problem), custom_retraction_(std::move(retraction)) {
  F_ASSERT(p_ != nullptr);
  F_ASSERT_GT(p_->dimension, 0, "Need at least one variable");

  // allocate space
  qp_.G.resize(p_->dimension, p_->dimension);
  qp_.c.resize(p_->dimension);

  // determine the size of the equality constraint matrix
  int total_eq_size = 0;
  int max_error_size = 0;
  for (const ResidualBase::unique_ptr& constraint : p_->equality_constraints) {
    total_eq_size += constraint->Dimension();
    max_error_size = std::max(max_error_size, constraint->Dimension());
  }
  qp_.A_eq.resize(total_eq_size, p_->dimension);
  qp_.b_eq.resize(total_eq_size);

  // we'll fill these out later
  qp_.constraints.reserve(p_->inequality_constraints.size());

  // leave uninitialized, we'll fill this in later
  variables_.resize(p_->dimension);
  candidate_vars_.resizeLike(variables_);
  dx_.resizeLike(variables_);
  dx_.setZero();

  // also compute max error size for the soft costs too
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    max_error_size = std::max(max_error_size, cost->Dimension());
  }
  error_buffer_.resize(max_error_size);
}

static void CheckParams(const ConstrainedNonlinearLeastSquares::Params& params) {
  F_ASSERT_GE(params.max_iterations, 0);
  F_ASSERT_GE(params.max_qp_iterations, 1);
  F_ASSERT_GT(params.termination_kkt_tolerance, 0);
  F_ASSERT_GT(params.absolute_exit_tol, 0);
  F_ASSERT_GE(params.max_line_search_iterations, 0);
  F_ASSERT_GE(params.relative_exit_tol, 0);
  F_ASSERT_LE(params.relative_exit_tol, 1);
  F_ASSERT_GE(params.absolute_first_derivative_tol, 0);
  F_ASSERT_GT(params.armijo_search_tau, 0);
  F_ASSERT_LT(params.armijo_search_tau, 1);
  F_ASSERT_GE(params.equality_penalty_initial, 0);
  F_ASSERT_GE(params.equality_penalty_scale_factor, 1.0);
  F_ASSERT_GE(params.equality_penalty_rho, 0);
  F_ASSERT_LT(params.equality_penalty_rho, 1);
  F_ASSERT_GE(params.max_lambda, 0);
  F_ASSERT_LE(params.min_lambda, params.max_lambda);
  F_ASSERT_GE(params.lambda_initial, params.min_lambda);
  F_ASSERT_LE(params.lambda_initial, params.max_lambda);
  F_ASSERT_GE(params.lambda_failure_init, 0);
}

NLSSolverOutputs ConstrainedNonlinearLeastSquares::Solve(const Params& params,
                                                         const Eigen::VectorXd& variables) {
  F_ASSERT(p_ != nullptr, "Must have a valid problem");
  CheckParams(params);
  variables_ = variables;
  state_ = OptimizerState::NOMINAL;

  if (p_->inequality_constraints.empty() && !p_->equality_constraints.empty()) {
    solver_ = QPNullSpaceSolver();
  } else {
    solver_ = QPInteriorPointSolver();
  }

  // Iterate until max.
  double lambda{params.lambda_initial};
  double penalty{params.equality_penalty_initial};
  int num_qp_iters = 0;
  for (int iter = 0; iter < params.max_iterations; ++iter) {
    // Fill out the QP and compute current errors.
    const Errors errors_pre = LinearizeAndFillQP(variables_, lambda, *p_, &qp_);

    // Compute the descent direction, `dx`.
    const auto [qp_outputs, lagrange_multipliers] = ComputeStepDirection(params);
    num_qp_iters += qp_outputs.num_iterations;

    F_ASSERT(!dx_.hasNaN(), "QP produced NaN values. G:\n{}\nc: {}\nA_eq:\n{}\nb_eq: {}",
             fmt::streamed(qp_.G), fmt::streamed(qp_.c.transpose()), fmt::streamed(qp_.A_eq),
             fmt::streamed(qp_.b_eq.transpose()));

    // Compute the directional derivative of the cost function about the current linearization
    // point, in the direction of the QP step.
    const DirectionalDerivatives directional_derivative = ComputeQPCostDerivative(qp_, dx_);

    // Raise the penalty parameter if necessary.
    if (!p_->equality_constraints.empty()) {
      const double new_penalty =
          SelectPenalty(qp_, dx_, lagrange_multipliers, params.equality_penalty_rho);
      if (new_penalty > penalty) {
        penalty = new_penalty * params.equality_penalty_scale_factor;
      }
    }

    // Do line search.
    constexpr double armijo_c1 = 1.0e-4; /* todo: add param */
    const StepSizeSelectionResult step_result =
        SelectStepSize(params.max_line_search_iterations, params.absolute_first_derivative_tol,
                       errors_pre, directional_derivative, penalty, armijo_c1,
                       params.line_search_strategy, params.armijo_search_tau);

    // Check if we should terminate (this call also updates variables_ on success).
    const double old_lambda = lambda;
    NLSTerminationState maybe_exit =
        UpdateLambdaAndCheckExitConditions(params, step_result, errors_pre, penalty, lambda);
    if (logging_callback_) {
      // Log the eigenvalues of the QP as well.
      // TODO: Make this an optional step, since it comes with material cost and is only for
      //  logging.
      const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(qp_.G);
      const QPEigenvalues eigenvalues{solver.eigenvalues().minCoeff(),
                                      solver.eigenvalues().maxCoeff(),
                                      solver.eigenvalues().cwiseAbs().minCoeff()};
      const NLSLogInfo info{iter,
                            state_,
                            old_lambda,
                            errors_pre,
                            qp_outputs,
                            eigenvalues,
                            directional_derivative,
                            penalty,
                            step_result,
                            steps_,
                            maybe_exit};
      const bool should_proceed = logging_callback_(*this, info);
      if (maybe_exit == NLSTerminationState::NONE && !should_proceed) {
        maybe_exit = NLSTerminationState::USER_CALLBACK;
      }
    }

    if (maybe_exit != NLSTerminationState::NONE) {
      return {maybe_exit, iter + 1, num_qp_iters};
    }
  }
  return {NLSTerminationState::MAX_ITERATIONS, params.max_iterations, num_qp_iters};
}

void ConstrainedNonlinearLeastSquares::RetractCandidateVars(const double alpha) {
  candidate_vars_ = variables_;
  if (custom_retraction_) {
    custom_retraction_(candidate_vars_, const_cast<const Eigen::VectorXd&>(dx_).head(p_->dimension),
                       alpha);
  } else {
    candidate_vars_ += dx_ * alpha;
  }
}

Errors ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(const Eigen::VectorXd& variables,
                                                            const double lambda,
                                                            const Problem& problem, QP* const qp) {
  F_ASSERT(qp != nullptr);
  F_ASSERT_EQ(qp->G.rows(), problem.dimension);
  F_ASSERT_EQ(qp->G.cols(), problem.dimension);
  F_ASSERT_EQ(qp->c.rows(), problem.dimension);
  F_ASSERT_EQ(qp->A_eq.rows(), qp->b_eq.rows());
  Errors output_errors{};

  // zero out the linear system before adding all the costs to it
  qp->G.setZero();
  qp->c.setZero();
  for (const ResidualBase::unique_ptr& cost : problem.costs) {
    output_errors.f += cost->UpdateHessian(variables, &qp->G, &qp->c);
  }
  if (lambda > 0) {
    qp->G.diagonal().array() += lambda;
  }

  // linearize equality constraints
  qp->A_eq.setZero();
  qp->b_eq.setZero();
  int row = 0;
  for (const ResidualBase::unique_ptr& eq : problem.equality_constraints) {
    const int dim = eq->Dimension();
    F_ASSERT_LE(row + dim, qp->A_eq.rows());

    // block we write the error into
    auto b_seg = qp->b_eq.segment(row, dim);
    eq->UpdateJacobian(variables, qp->A_eq.middleRows(row, dim), b_seg);

    // total L1 norm in the equality constraints
    output_errors.equality += b_seg.lpNorm<1>();
    row += dim;
  }

  // shift constraints to the new linearization point:
  qp->constraints.clear();
  for (const LinearInequalityConstraint& c : problem.inequality_constraints) {
    qp->constraints.push_back(c.ShiftTo(variables));
  }
  return output_errors;
}

std::tuple<QPSolverOutputs, std::optional<QPLagrangeMultipliers>>
ConstrainedNonlinearLeastSquares::ComputeStepDirection(const Params& params) {
  dx_.setZero();

  if (QPInteriorPointSolver* const ip_solver = std::get_if<QPInteriorPointSolver>(&solver_);
      ip_solver) {
    ip_solver->Setup(&qp_);

    // Set up params, TODO(gareth): Tune this better.
    QPInteriorPointSolver::Params qp_params{};
    qp_params.max_iterations = params.max_qp_iterations;
    qp_params.termination_kkt_tol = params.termination_kkt_tolerance;
    qp_params.initial_mu = 1.0;
    qp_params.sigma = 0.1;
    qp_params.initialize_mu_with_complementarity = false;

    // If there are equality constraints, try initializing by just solving the
    // equality constrained quadratic problem.
    if (!p_->equality_constraints.empty()) {
      qp_params.initial_guess_method = InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED;
    } else {
      qp_params.initial_guess_method = InitialGuessMethod::NAIVE;
    }

    const QPSolverOutputs qp_outputs = ip_solver->Solve(qp_params);

    // Update our search direction:
    dx_ = ip_solver->x_block();

    // Determine stats on the lagrange multipliers - these are used in the penalty update rules.
    std::optional<QPLagrangeMultipliers> multipliers{};
    if (const auto y = ip_solver->y_block(); y.rows() > 0) {
      multipliers = QPLagrangeMultipliers{y.minCoeff(), y.lpNorm<Eigen::Infinity>(), y.norm()};
    }

    return std::make_tuple(qp_outputs, multipliers);
  }

  QPNullSpaceSolver* const null_solver = std::get_if<QPNullSpaceSolver>(&solver_);
  F_ASSERT(null_solver);

  null_solver->Setup(&qp_);
  const QPSolverOutputs qp_outputs = null_solver->Solve();
  dx_ = null_solver->variables();

  return std::make_tuple(qp_outputs, std::optional<QPLagrangeMultipliers>{std::nullopt});
}

Errors ConstrainedNonlinearLeastSquares::EvaluateNonlinearErrors(const Eigen::VectorXd& vars) {
  Errors output_errors{};
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    const auto err_out = error_buffer_.head(cost->Dimension());
    cost->ErrorVector(vars, err_out);
    output_errors.f += 0.5 * err_out.squaredNorm();
  }
  for (const ResidualBase::unique_ptr& eq : p_->equality_constraints) {
    const auto err_out = error_buffer_.head(eq->Dimension());
    eq->ErrorVector(vars, err_out);
    output_errors.equality += err_out.lpNorm<1>();
  }
  F_ASSERT(!std::isnan(output_errors.f), "vars = {}", fmt::streamed(vars.transpose()));
  F_ASSERT(!std::isnan(output_errors.equality), "vars = {}", fmt::streamed(vars.transpose()));
  return output_errors;
}

// TODO(gareth): Investigate an approach like algorithm 11.5?
NLSTerminationState ConstrainedNonlinearLeastSquares::UpdateLambdaAndCheckExitConditions(
    const Params& params, const StepSizeSelectionResult step_result, const Errors& initial_errors,
    const double penalty, double& lambda) {
  if (step_result == StepSizeSelectionResult::SUCCESS) {
    F_ASSERT(!steps_.empty(), "Must have logged a step");
    // Update the state, and decrease lambda.
    variables_.swap(candidate_vars_);  //  replace w/ the candidate variables
    state_ = OptimizerState::NOMINAL;
    lambda = std::max(lambda * 0.1, params.min_lambda);

    // Check termination criteria.
    const LineSearchStep& final_step = steps_.back();
    if (final_step.errors.LInfinity() < params.absolute_exit_tol) {
      // Satisfied absolute tolerance, exit.
      return NLSTerminationState::SATISFIED_ABSOLUTE_TOL;
    } else if (final_step.errors.Total(penalty) >
               initial_errors.Total(penalty) * (1 - params.relative_exit_tol)) {
      return NLSTerminationState::SATISFIED_RELATIVE_TOL;
    }
  } else if (step_result == StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED) {
    // The QP computed a direction where derivative of cost is zero.
    return NLSTerminationState::SATISFIED_FIRST_ORDER_TOL;
  } else if (step_result == StepSizeSelectionResult::FAILURE_MAX_ITERATIONS ||
             step_result == StepSizeSelectionResult::FAILURE_POSITIVE_DERIVATIVE) {
    if (state_ == OptimizerState::NOMINAL) {
      // Things were going well, but we failed - initialize lambda to attempt restore.
      lambda = std::max(params.lambda_failure_init, lambda);
      state_ = OptimizerState::ATTEMPTING_RESTORE_LM;
    } else {
      F_ASSERT_EQ(state_, OptimizerState::ATTEMPTING_RESTORE_LM);
      // We are already attempting to recover and failing, ramp up lambda.
      lambda *= 10;
    }
    if (lambda > params.max_lambda) {
      // failed
      return NLSTerminationState::MAX_LAMBDA;
    }
  }
  // continue
  return NLSTerminationState::NONE;
}

/*
 * See equations 3.6a/3.6b
 *
 * For now this just checks the first step it can find that produces a decrease, up
 * to some maximum number of iterations.
 */
StepSizeSelectionResult ConstrainedNonlinearLeastSquares::SelectStepSize(
    const int max_iterations, const double abs_first_derivative_tol, const Errors& errors_pre,
    const DirectionalDerivatives& derivatives, const double penalty, const double armijo_c1,
    const LineSearchStrategy strategy, const double backtrack_search_tau) {
  F_ASSERT_GT(penalty, 0.0);
  steps_.clear();

  // compute the directional derivative, w/ the current penalty
  const double directional_derivative = derivatives.Total(penalty);

  // iterate to find a viable alpha, or give up
  double alpha{1};
  for (int iter = 0; iter <= max_iterations; ++iter) {
    // compute new alpha value
    if (strategy == LineSearchStrategy::POLYNOMIAL_APPROXIMATION) {
      alpha = ComputeAlphaPolynomialApproximation(iter, alpha, errors_pre, derivatives, penalty);
    } else {
      F_ASSERT_EQ(strategy, LineSearchStrategy::ARMIJO_BACKTRACK);
      if (iter > 0) {
        alpha = alpha * backtrack_search_tau;
      }
    }

    // Update our candidate state.
    F_ASSERT(std::isfinite(alpha), "alpha = {}, iter = {}", alpha, iter);
    RetractCandidateVars(alpha);

    // Compute errors.
    const Errors errors_step = EvaluateNonlinearErrors(candidate_vars_);
    steps_.emplace_back(alpha, errors_step);

    if (derivatives.LInfinity() < abs_first_derivative_tol) {
      // at a stationary point
      return StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED;
    } else if (directional_derivative > 0) {
      // the derivative is positive
      return StepSizeSelectionResult::FAILURE_POSITIVE_DERIVATIVE;
    }

    // check the armijo condition, TODO: add curvature check as well
    if (errors_step.Total(penalty) <=
        errors_pre.Total(penalty) + directional_derivative * alpha * armijo_c1) {
      // success, accept it
      return StepSizeSelectionResult::SUCCESS;
    }
  }
  // hit max iterations
  return StepSizeSelectionResult::FAILURE_MAX_ITERATIONS;
}

double ConstrainedNonlinearLeastSquares::ComputeAlphaPolynomialApproximation(
    const int iteration, const double alpha, const Errors& errors_pre,
    const DirectionalDerivatives& derivatives, const double penalty) const {
  F_ASSERT_GE(iteration, 0);
  if (iteration == 0) {
    return alpha;
  } else if (iteration == 1) {
    F_ASSERT_EQ(steps_.size(), 1);
    // Pick a new alpha by approximating cost as a quadratic.
    // steps_.back() here is the "full step" error.
    const LineSearchStep& prev_step = steps_.back();
    const double new_alpha =
        QuadraticApproxMinimum(errors_pre.Total(penalty), derivatives.Total(penalty),
                               prev_step.alpha, prev_step.errors.Total(penalty));
    F_ASSERT_LT(new_alpha, prev_step.alpha, "Alpha must decrease, alpha = {}, prev_alpha = {}",
                alpha, prev_step.alpha);
    return new_alpha;
  }
  F_ASSERT_GE(steps_.size(), 2);

  // Try the cubic approximation.
  const LineSearchStep& second_last_step = steps_[steps_.size() - 2];
  const LineSearchStep& last_step = steps_.back();

  // Compute coefficients
  const Eigen::Vector2d ab = CubicApproxCoeffs(
      errors_pre.Total(penalty), derivatives.Total(penalty), second_last_step.alpha,
      second_last_step.errors.Total(penalty), last_step.alpha, last_step.errors.Total(penalty));

  // Solve.
  const double new_alpha = CubicApproxMinimum(derivatives.Total(penalty), ab);
  F_ASSERT_LT(new_alpha, last_step.alpha,
              "Alpha must decrease in the line search, alpha = {}, prev_alpha = {}", alpha,
              last_step.alpha);
  return new_alpha;
}

template <typename T>
static T Sign(T x) {
  static_assert(std::is_floating_point<T>::value, "");
  if (x > 0) {
    return 1;
  } else if (x < 0) {
    return -1;
  } else {
    return 0;
  }
}

DirectionalDerivatives ConstrainedNonlinearLeastSquares::ComputeQPCostDerivative(
    const QP& qp, const Eigen::VectorXd& dx) {
  // We want the first derivative of the cost function, evaluated at the current linearization
  // point.
  //  d( 0.5 * h(x + dx * alpha)^T * h(x + dx * alpha) ) =
  //    h(x + dx * alpha)^T * dh(y)/dy * dx, evaluated at y = x, alpha = 0
  //
  //  For the quadratic part, this is just: c^T * dx, since c = J(x)^T * f(x)
  //  For the equality constraint, we compute J(x)^T * f(x) here explicitly.
  F_ASSERT_EQ(qp.c.rows(), dx.rows(), "Mismatch between dx and c");
  F_ASSERT_EQ(qp.A_eq.cols(), dx.rows(), "Mismatch between dx and A_eq");

  DirectionalDerivatives out{};
  out.d_f = qp.c.dot(dx);

  // Now we do the equality constraints, which for now have an L1 norm:
  //    d|c(x + alpha * dx)|/dalpha = d|v|/dv * dc(x)/dx * dx
  // Where d|v|/dv evaluates to the sign of each element.
  // TODO: This should just be equal to b_eq, so we can simplify this.
  for (int i = 0; i < qp.A_eq.rows(); ++i) {
    const double sign_ci = Sign(qp.b_eq[i]);
    out.d_equality += sign_ci * qp.A_eq.row(i).dot(dx);
  }
  return out;
}

double ConstrainedNonlinearLeastSquares::SelectPenalty(
    const QP& qp, const Eigen::VectorXd& dx,
    const std::optional<QPLagrangeMultipliers>& lagrange_multipliers, const double rho) {
  if (lagrange_multipliers) {
    // Equation 18.32. Note that this is not the recommended algorithm in the book, instead
    // they suggest 18.33. But this seems to work better for me.
    return lagrange_multipliers->l_infinity;
  } else {
    // This code-path occurs when the null-space solver is in use.
    // We don't have lagrange multipliers in this context, but we still need to update the penalty.
    // Instead we apply the inequality (18.36).
    const double l1_eq = std::max(qp.b_eq.lpNorm<1>(), std::numeric_limits<double>::epsilon());

    // Compute: ∇ f^T * dx + (1/2) dx^T * (∇^2 f) * dx
    const double quadratic_cost_approx =
        qp.c.dot(dx) + 0.5 * std::max(0.0, dx.dot(qp.G.selfadjointView<Eigen::Lower>() * dx));
    return quadratic_cost_approx / ((1 - rho) * l1_eq);
  }
}

// For this approximation to provide a decrease, we must have:
//  a > 0 (ie. there is a minimum between a=0 and a=alpha_0)
//
// We have:
//  a = (phi_alpha_0 - phi_0 - alpha_0 * phi_prime_0) / (alpha_0 ** 2)
//
// So:
//
//  phi_alpha_0 - phi_0 - alpha_0 * phi_prime_0 > 0
//  phi_alpha_0 - phi_0 > alpha_0 * phi_prime_0
//
// We already enforce phi_alpha_0 - phi_0 > 0, and since phi_prime_0 has to be
// a descent direction (as a result of the QP solver), it will be < 0
double ConstrainedNonlinearLeastSquares::QuadraticApproxMinimum(const double phi_0,
                                                                const double phi_prime_0,
                                                                const double alpha_0,
                                                                const double phi_alpha_0) {
  F_ASSERT_GT(alpha_0, 0);
  F_ASSERT_LT(phi_prime_0, 0);
  const double numerator = 2 * (phi_alpha_0 - phi_0 - phi_prime_0 * alpha_0);
  F_ASSERT_GT(numerator, 0, "phi_alpha_0={}, phi_0={}, alpha_0={}", phi_alpha_0, phi_0, alpha_0);
  return -phi_prime_0 * alpha_0 * alpha_0 / numerator;
}

// This function fits a cubic polynomial of the form:
//
//  a*x^3 + b*x^2 + x*phi_prime_0 + phi_0
//
Eigen::Vector2d ConstrainedNonlinearLeastSquares::CubicApproxCoeffs(
    const double phi_0, const double phi_prime_0, const double alpha_0, const double phi_alpha_0,
    const double alpha_1, const double phi_alpha_1) {
  F_ASSERT_GT(alpha_1, 0);
  F_ASSERT_GT(alpha_0, alpha_1, "This must be satisfied for the system to be solvable");
  // clang-format off
  const Eigen::Matrix2d A = (Eigen::Matrix2d() <<
      alpha_0 * alpha_0 * alpha_0, alpha_0 * alpha_0,
      alpha_1 * alpha_1 * alpha_1, alpha_1 * alpha_1).finished();
  const Eigen::Vector2d b{
      phi_alpha_0 - phi_0 - phi_prime_0 * alpha_0,
      phi_alpha_1 - phi_0 - phi_prime_0 * alpha_1
  };
  // clang-format on
  return A.inverse() * b;
}

double ConstrainedNonlinearLeastSquares::CubicApproxMinimum(const double phi_prime_0,
                                                            const Eigen::Vector2d& ab) {
  F_ASSERT_GT(std::abs(ab[0]), 0, "Coefficient a cannot be zero");
  const double arg_sqrt = ab[1] * ab[1] - 3 * ab[0] * phi_prime_0;
  constexpr double kNegativeTol = -1.0e-12;
  F_ASSERT_GE(arg_sqrt, kNegativeTol, "This term must be positive: a={}, b={}, phi_prime_0={}",
              ab[0], ab[1], phi_prime_0);
  return (std::sqrt(std::max(arg_sqrt, 0.)) - ab[1]) / (3 * ab[0]);
}

// These objects are passed as arguments to a static function for ease of testing this method.
void ConstrainedNonlinearLeastSquares::ComputeSecondOrderCorrection(
    const Eigen::VectorXd& updated_x,
    const std::vector<ResidualBase::unique_ptr>& equality_constraints, QP* qp,
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>* solver, Eigen::VectorXd* dx_out) {
  F_ASSERT(qp);
  F_ASSERT(solver);
  F_ASSERT(dx_out);

  // we use the QP `b` vector as storage for this operation
  int row = 0;
  for (const ResidualBase::unique_ptr& eq : equality_constraints) {
    const int dim = eq->Dimension();
    F_ASSERT_LE(row + dim, qp->b_eq.rows(), "Insufficient rows in vector b");
    eq->ErrorVector(updated_x, qp->b_eq.segment(row, dim));
    row += dim;
  }

  // compute the pseudo-inverse
  solver->compute(qp->A_eq);
  dx_out->noalias() -= solver->solve(qp->b_eq);
}

}  // namespace mini_opt
