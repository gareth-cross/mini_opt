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
  F_ASSERT_GE(params.lambda_decrease_on_success, 0);
  F_ASSERT_LT(params.lambda_decrease_on_success, 1.0);
  F_ASSERT_GE(params.lambda_decrease_on_restore, 0);
  F_ASSERT_LT(params.lambda_decrease_on_restore, 1.0);
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
  std::vector<NLSIteration> iterations;
  iterations.reserve(10);
  for (int iter = 0; iter < params.max_iterations; ++iter) {
    // Fill out the QP and compute current errors.
    const Errors errors_pre = LinearizeAndFillQP(variables_, lambda, *p_, &qp_);

    // Compute the descent direction, `dx`.
    auto qp_outputs = ComputeStepDirection(params);

    if (QPWasIndefinite(qp_outputs)) {
      return {NLSTerminationState::QP_INDEFINITE, std::move(iterations)};
    }

    // Compute the directional derivative of the cost function about the current linearization
    // point, in the direction of the QP step.
    const DirectionalDerivatives directional_derivative = ComputeQPCostDerivative(qp_, dx_);

    // Raise the penalty parameter if necessary.
    if (!p_->equality_constraints.empty()) {
      const double new_penalty = SelectPenalty(qp_, dx_, MaybeGetLagrangeMultipliers(qp_outputs),
                                               params.equality_penalty_rho);
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
    const NLSTerminationState maybe_exit =
        UpdateLambdaAndCheckExitConditions(params, step_result, errors_pre, penalty, lambda);

    NLSIteration info{
        iter,
        state_,
        old_lambda,
        errors_pre,
        std::move(qp_outputs),
        params.log_qp_eigenvalues ? qp_.ComputeEigenvalueStats() : std::optional<QPEigenvalues>{},
        directional_derivative,
        penalty,
        step_result,
        std::move(steps_),
        maybe_exit};
    iterations.push_back(std::move(info));

    // If the user callback returns false, we will terminate early.
    if (user_exit_callback_) {
      const bool should_proceed = user_exit_callback_(iterations.back());
      if (maybe_exit == NLSTerminationState::NONE && !should_proceed) {
        return {NLSTerminationState::USER_CALLBACK, std::move(iterations)};
      }
    }

    if (maybe_exit != NLSTerminationState::NONE) {
      return {maybe_exit, std::move(iterations)};
    }
  }
  return {NLSTerminationState::MAX_ITERATIONS, std::move(iterations)};
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

std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs>
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

    QPInteriorPointSolverOutputs qp_outputs = ip_solver->Solve(qp_params);

    // Update our search direction:
    dx_ = ip_solver->x_block();

    return {std::move(qp_outputs)};
  }

  QPNullSpaceSolver* const null_solver = std::get_if<QPNullSpaceSolver>(&solver_);
  F_ASSERT(null_solver);

  null_solver->Setup(&qp_);
  const QPNullSpaceTerminationState term = null_solver->Solve();
  if (term == QPNullSpaceTerminationState::SUCCESS) {
    dx_ = null_solver->variables();
  } else {
    dx_.setZero();
  }
  return term;
}

std::optional<QPLagrangeMultipliers> ConstrainedNonlinearLeastSquares::MaybeGetLagrangeMultipliers(
    const std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs>& output) {
  if (const QPInteriorPointSolverOutputs* ip = std::get_if<QPInteriorPointSolverOutputs>(&output);
      ip != nullptr) {
    return ip->lagrange_multipliers;
  }
  return std::nullopt;
}

bool ConstrainedNonlinearLeastSquares::QPWasIndefinite(
    const std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs>& output) {
  if (const QPNullSpaceTerminationState* ns = std::get_if<QPNullSpaceTerminationState>(&output);
      ns != nullptr) {
    return *ns == QPNullSpaceTerminationState::NOT_POSITIVE_DEFINITE;
  }
  return false;  // TODO: Return a "non-SPD" condition from the interior point solver.
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
    if (state_ == OptimizerState::ATTEMPTING_RESTORE_LM) {
      lambda = std::max(lambda * params.lambda_decrease_on_restore, params.min_lambda);

    } else {
      lambda = std::max(lambda * params.lambda_decrease_on_success, params.min_lambda);
    }
    state_ = OptimizerState::NOMINAL;

    // Check termination criteria.
    const LineSearchStep& final_step = steps_.back();
    if (final_step.errors.LInfinity() < params.absolute_exit_tol) {
      // Satisfied absolute tolerance, exit.
      return NLSTerminationState::SATISFIED_ABSOLUTE_TOL;
    } else if (final_step.errors.Total(penalty) >
               initial_errors.Total(penalty) * (1 - params.relative_exit_tol)) {
      return NLSTerminationState::SATISFIED_RELATIVE_TOL;
    }
  } else if (step_result == StepSizeSelectionResult::FIRST_ORDER_SATISFIED) {
    // The QP computed a direction where derivative of cost is zero.
    return NLSTerminationState::SATISFIED_FIRST_ORDER_TOL;
  } else if (step_result == StepSizeSelectionResult::MAX_ITERATIONS ||
             step_result == StepSizeSelectionResult::POSITIVE_DERIVATIVE) {
    if (state_ == OptimizerState::NOMINAL) {
      // Things were going well, but we failed - initialize lambda to attempt restore.
      lambda = std::max(params.lambda_failure_init, lambda * 10.0);
      state_ = OptimizerState::ATTEMPTING_RESTORE_LM;
    } else {
      F_ASSERT_EQ(state_, OptimizerState::ATTEMPTING_RESTORE_LM);
      // We are already attempting to recover and failing, ramp up lambda.
      lambda *= 10.0;
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
  F_ASSERT(!errors_pre.ContainsInvalidValues(), "{}, {}", errors_pre.f, errors_pre.equality);
  steps_.clear();
  steps_.reserve(5);

  // compute the directional derivative, w/ the current penalty
  const double directional_derivative = derivatives.Total(penalty);

  // iterate to find a viable alpha, or give up
  double alpha{1};
  for (int iter = 0; iter <= max_iterations; ++iter) {
    // compute new alpha value
    if (strategy == LineSearchStrategy::POLYNOMIAL_APPROXIMATION) {
      // On iteration 0 we'll just use `alpha = 1`.
      if (iter > 0) {
        const std::optional<double> new_alpha =
            ComputeAlphaPolynomialApproximation(iter, errors_pre, derivatives, penalty);
        // Double check that we produced a valid solution:
        if (!new_alpha.has_value() || !std::isfinite(*new_alpha) || *new_alpha <= 0.0 ||
            *new_alpha >= alpha) {
          return StepSizeSelectionResult::FAILURE_INVALID_ALPHA;
        } else {
          alpha = new_alpha.value();
        }
      }
    } else {
      F_ASSERT_EQ(strategy, LineSearchStrategy::ARMIJO_BACKTRACK);
      if (iter > 0) {
        alpha = alpha * backtrack_search_tau;
      }
    }

    // Update our candidate state.
    RetractCandidateVars(alpha);

    // Compute errors, and double check that they were numerically valid.
    const Errors errors_step = EvaluateNonlinearErrors(candidate_vars_);
    steps_.emplace_back(alpha, errors_step);

    if (errors_step.ContainsInvalidValues()) {
      return StepSizeSelectionResult::FAILURE_NON_FINITE_COST;
    }

    if (derivatives.LInfinity() < abs_first_derivative_tol) {
      // at a stationary point
      return StepSizeSelectionResult::FIRST_ORDER_SATISFIED;
    } else if (directional_derivative > 0) {
      // the derivative is positive
      return StepSizeSelectionResult::POSITIVE_DERIVATIVE;
    }

    // check the armijo condition, TODO: add curvature check as well
    if (errors_step.Total(penalty) <=
        errors_pre.Total(penalty) + directional_derivative * alpha * armijo_c1) {
      // success, accept it
      return StepSizeSelectionResult::SUCCESS;
    }
  }
  // hit max iterations
  return StepSizeSelectionResult::MAX_ITERATIONS;
}

std::optional<double> ConstrainedNonlinearLeastSquares::ComputeAlphaPolynomialApproximation(
    const int iteration, const Errors& errors_pre, const DirectionalDerivatives& derivatives,
    const double penalty) const {
  F_ASSERT_GE(iteration, 1);
  if (iteration == 1) {
    F_ASSERT_EQ(steps_.size(), 1);
    // Pick a new alpha by approximating cost as a quadratic.
    // steps_.back() here is the "full step" error, because this is iteration 1.
    const LineSearchStep& prev_step = steps_.back();
    return QuadraticApproxMinimum(errors_pre.Total(penalty), derivatives.Total(penalty),
                                  prev_step.alpha, prev_step.errors.Total(penalty));
  }
  F_ASSERT_GE(steps_.size(), 2);

  // Try the cubic approximation.
  const LineSearchStep& second_last_step = steps_[steps_.size() - 2];
  const LineSearchStep& last_step = steps_.back();

  // Compute coefficients.
  const Eigen::Vector2d ab = CubicApproxCoeffs(
      errors_pre.Total(penalty), derivatives.Total(penalty), second_last_step.alpha,
      second_last_step.errors.Total(penalty), last_step.alpha, last_step.errors.Total(penalty));

  // Solve the cubic. The caller of this method is responsible for sanity-checking the result.
  return CubicApproxMinimum(derivatives.Total(penalty), ab);
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
    // We don't have lagrange multipliers in this context, but we still need to update the
    // penalty. Instead we apply the inequality (18.36).
    const double l1_eq = std::max(qp.b_eq.lpNorm<1>(), std::numeric_limits<double>::epsilon());

    // Compute: ∇ f^T * dx + (1/2) dx^T * (∇^2 f) * dx
    const double quadratic_cost_approx =
        qp.c.dot(dx) + 0.5 * std::max(0.0, dx.dot(qp.G.selfadjointView<Eigen::Lower>() * dx));
    return quadratic_cost_approx / ((1 - rho) * l1_eq);
  }
}

// For this approximation to provide a decrease there must be a minimum between α = 0 and α = α_0.
//
// We are fitting a quadratic with 3 unknowns. We have three constraints:
//
// - φ(0) is known.
// - φ'(0) is known.
// - φ(α_0) is known.
//
// φ(α) = a*α^2 + b*α + c, where:
//
// c = φ(0)
// b = φ'(0)
// a = (φ(α_0) - b * α_0 - c) / α_0**2 = (φ(α_0) - φ'(0) * α_0 - φ(0)) / α_0**2
//
// The minimum occurs when:
//
// φ'(α_min) = 2*a*α_min + b = 0
//
// Or: α_min = -b / (2 * a) = (1/2) * -(φ'(0) * α_0**2) / (φ(α_0) - φ'(0) * α_0 - φ(0))
//
// Thus we need: φ(α_0) - φ'(0) * α_0 - φ(0) > 0   (1)
//
// - a_0 is greater than zero.
// - φ'(0) is negative (because the gradient at a = 0 was a descent direction).
// - φ(α_0) >= φ(0), because otherwise we would have accepted α_0 as the solution.
//
// So (1) should be satisfied. That being said, we check the result for numerical safety.
std::optional<double> ConstrainedNonlinearLeastSquares::QuadraticApproxMinimum(
    const double phi_0, const double phi_prime_0, const double alpha_0, const double phi_alpha_0) {
  const double numerator = phi_alpha_0 - phi_prime_0 * alpha_0 - phi_0;
  if (phi_prime_0 > 0 || numerator <= 0) {
    return std::nullopt;
  }
  return -phi_prime_0 * alpha_0 * alpha_0 / (2.0 * numerator);
}

// This function fits a cubic polynomial of the form:
//
//  φ(α) = a*α^3 + b*α^2 + α*φ'(0) + φ(0)
//
// The constraints we have are:
//
// - φ(0) is known.
// - φ'(0) is known.
// - φ(α_0) is known.
// - φ(α_1) is known.
//
// Only the coefficients [a, b] need to be obtained, so we build the linear system:
//
//  A * [a; b] = rhs
//
// From the equations: a*α^3 + b*α^2 = φ(α) - α*φ'(0) - φ(0)
//
// The determinant of `A` is:
//
//  |A| = α_0^3 * α_1^2 - α_0^2 * α_1^3
//      = (α_0 * α_1)^2 * (α_0 - α_1)
//
// In order to be invertible we must have: α_0 > α_1 and α_1 > 0.
Eigen::Vector2d ConstrainedNonlinearLeastSquares::CubicApproxCoeffs(
    const double phi_0, const double phi_prime_0, const double alpha_0, const double phi_alpha_0,
    const double alpha_1, const double phi_alpha_1) {
  F_ASSERT_GT(alpha_1, 0);
  F_ASSERT_GT(alpha_0, alpha_1, "This must be satisfied for the system to be solvable");
  // clang-format off
  const Eigen::Matrix2d A = (Eigen::Matrix2d() <<
      alpha_0 * alpha_0 * alpha_0, alpha_0 * alpha_0,
      alpha_1 * alpha_1 * alpha_1, alpha_1 * alpha_1).finished();
  const Eigen::Vector2d rhs{
      phi_alpha_0 - phi_0 - phi_prime_0 * alpha_0,
      phi_alpha_1 - phi_0 - phi_prime_0 * alpha_1
  };
  // clang-format on
  return A.inverse() * rhs;
}

// We are computing the minimum of:
//
//  φ(α) = a*α^3 + b*α^2 + α*φ'(0) + φ(0)
//
// Which occurs at:
//
//  φ'(α) = 3*a*α^2 + 2*b*α + φ'(0) = 0
//
// The solution of which is:
//
//  α = (-2*b ± sqrt(4*b^2 - 12*a*φ'(0))) / 2 * a
//    = (-b ± sqrt(b^2 - 3*a*φ'(0)) / a
//
//
std::optional<double> ConstrainedNonlinearLeastSquares::CubicApproxMinimum(
    const double phi_prime_0, const Eigen::Vector2d& ab) {
  const double a = ab[0];
  const double b = ab[1];
  const double arg_sqrt = b * b - 3 * a * phi_prime_0;
  constexpr double kNegativeTol = -1.0e-12;
  if (a == 0.0 || arg_sqrt < kNegativeTol) {
    return std::nullopt;
  }
  const double denominator = -b + std::sqrt(std::max(arg_sqrt, 0.));
  return denominator / (3 * a);
}

}  // namespace mini_opt
