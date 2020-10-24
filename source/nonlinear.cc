// Copyright 2020 Gareth Cross
#include "mini_opt/nonlinear.hpp"

#include <Eigen/Dense>  //  for inverse()
#include <utility>

namespace mini_opt {

ConstrainedNonlinearLeastSquares::ConstrainedNonlinearLeastSquares(const Problem* const problem,
                                                                   Retraction retraction)
    : p_(problem), custom_retraction_(std::move(retraction)) {
  ASSERT(p_ != nullptr);
  ASSERT(p_->dimension > 0, "Need at least one variable");

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
  prev_variables_.resizeLike(variables_);
  dx_.resizeLike(variables_);
  dx_.setZero();

  // also compute max error size for the soft costs too
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    max_error_size = std::max(max_error_size, cost->Dimension());
  }
  error_buffer_.resize(max_error_size);
}

static void CheckParams(const ConstrainedNonlinearLeastSquares::Params& params) {
  ASSERT(params.max_iterations >= 0);
  ASSERT(params.max_qp_iterations >= 1);
  ASSERT(params.termination_kkt_tolerance > 0);
  ASSERT(params.absolute_exit_tol > 0);
  ASSERT(params.max_line_search_iterations >= 0);
  ASSERT(params.relative_exit_tol >= 0);
  ASSERT(params.relative_exit_tol <= 1);
  ASSERT(params.absolute_first_derivative_tol >= 0);
  ASSERT(params.max_lambda >= 0);
  ASSERT(params.min_lambda <= params.max_lambda);
  ASSERT(params.lambda_initial >= params.min_lambda);
  ASSERT(params.lambda_initial <= params.max_lambda);
  ASSERT(params.lambda_failure_init >= 0);
  ASSERT(params.equality_constraint_norm == Norm::L1 ||
         params.equality_constraint_norm == Norm::QUADRATIC);
}

NLSSolverOutputs ConstrainedNonlinearLeastSquares::Solve(const Params& params,
                                                         const Eigen::VectorXd& variables) {
  ASSERT(p_ != nullptr, "Must have a valid problem");
  CheckParams(params);
  variables_ = variables;
  prev_variables_ = variables;
  state_ = OptimizerState::NOMINAL;
  bool has_cached_qp_state{false};

  // Set up params, TODO(gareth): Tune this better.
  QPInteriorPointSolver::Params qp_params{};
  qp_params.max_iterations = params.max_qp_iterations;
  qp_params.termination_kkt_tol = params.termination_kkt_tolerance;
  qp_params.initial_mu = 1.0;
  qp_params.sigma = 0.1;
  qp_params.initialize_mu_with_complementarity = false;

  // Iterate until max.
  double lambda{params.lambda_initial};
  double penalty{params.equality_penalty_initial};
  int num_qp_iters = 0;
  for (int iter = 0; iter < params.max_iterations; ++iter) {
    // Fill out the QP and compute current errors.
    const Errors errors_pre =
        LinearizeAndFillQP(variables_, lambda, params.equality_constraint_norm, *p_, &qp_);

    if (iter == 0 || !has_cached_qp_state) {
      // If there are inequality constraints, try initializing by just solving the
      // equality constrained quadratic problem.
      if (!qp_.constraints.empty()) {
        qp_params.initial_guess_method = InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED;
      } else {
        qp_params.initial_guess_method = InitialGuessMethod::NAIVE;
      }
    } else {
      // We'll use our initial guess from the previous iteration.
      // This seems to produce a 2-3x reduction in # of QP iterations.
      qp_params.initial_guess_method = InitialGuessMethod::USER_PROVIDED;
      qp_params.initialize_mu_with_complementarity = true;
      // Reduce the barrier more aggressively. This seems to make a small reduction in
      // number of iterations, but needs more testing.
      qp_params.sigma = 0.05;
    }

    // Solve the QP.
    solver_.Setup(&qp_);
    if (has_cached_qp_state) {
      solver_.SetVariables(cached_qp_states_);
      solver_.x_block().setZero();
    }
    const QPSolverOutputs qp_outputs = solver_.Solve(qp_params);
    num_qp_iters += qp_outputs.num_iterations;

    // Compute the directional derivative of the cost function about the current linearization
    // point, in the direction of the QP step.
    dx_ = solver_.x_block();
    const DirectionalDerivatives directional_derivative =
        ComputeQPCostDerivative(qp_, dx_, params.equality_constraint_norm);

    // Raise the penalty parameter if necessary.
    if (!p_->equality_constraints.empty()) {
      const double new_penalty = SelectPenalty(
          params.equality_constraint_norm,
          const_cast<const QPInteriorPointSolver&>(solver_).y_block(), errors_pre.equality);
      if (new_penalty > penalty) {
        penalty = new_penalty * 1.01;
      }
    }

    // Do line search.
    const StepSizeSelectionResult step_result = SelectStepSize(
        params.max_line_search_iterations, params.absolute_first_derivative_tol, errors_pre,
        directional_derivative, penalty, 1.0e-4 /* todo: add param */, params.line_search_strategy,
        params.armijo_search_tau, params.equality_constraint_norm);

    if (step_result == StepSizeSelectionResult::SUCCESS) {
      // save the states of slacks, multipliers, etc...
      cached_qp_states_ = solver_.variables();
      has_cached_qp_state = true;
    }

    // Check if we should terminate (this call also updates variables_ on success).
    const double old_lambda = lambda;
    NLSTerminationState maybe_exit =
        UpdateLambdaAndCheckExitConditions(params, step_result, errors_pre, penalty, &lambda);
    if (logging_callback_) {
      // log the eigenvalues of the QP as well
      const auto dx_block = const_cast<const Eigen::VectorXd&>(dx_).head(p_->dimension);
      const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(qp_.G);
      const NLSLogInfo info{iter,       state_,
                            old_lambda, errors_pre,
                            qp_outputs, solver.eigenvalues(),
                            dx_block,   directional_derivative,
                            penalty,    step_result,
                            steps_,     maybe_exit};
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
    custom_retraction_(&candidate_vars_,
                       const_cast<const Eigen::VectorXd&>(dx_).head(p_->dimension), alpha);
  } else {
    candidate_vars_ += dx_ * alpha;
  }
}

Errors ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(const Eigen::VectorXd& variables,
                                                            const double lambda,
                                                            const Norm& equality_norm,
                                                            const Problem& problem, QP* const qp) {
  ASSERT(qp != nullptr);
  ASSERT(qp->G.rows() == problem.dimension);
  ASSERT(qp->G.cols() == problem.dimension);
  ASSERT(qp->c.rows() == problem.dimension);
  ASSERT(qp->A_eq.rows() == qp->b_eq.rows());
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
    ASSERT(row + dim <= qp->A_eq.rows());
    // block we write the error into
    auto b_seg = qp->b_eq.segment(row, dim);
    eq->UpdateJacobian(variables, qp->A_eq.middleRows(row, dim), b_seg);

    // total L1 norm in the equality constraints
    if (equality_norm == Norm::L1) {
      output_errors.equality += b_seg.lpNorm<1>();
    } else if (equality_norm == Norm::QUADRATIC) {
      output_errors.equality += 0.5 * b_seg.squaredNorm();
    }
    row += dim;
  }

  // shift constraints to the new linearization point:
  qp->constraints.clear();
  for (const LinearInequalityConstraint& c : problem.inequality_constraints) {
    qp->constraints.push_back(c.ShiftTo(variables));
  }
  return output_errors;
}

Errors ConstrainedNonlinearLeastSquares::EvaluateNonlinearErrors(const Eigen::VectorXd& vars,
                                                                 const Norm& equality_norm) {
  Errors output_errors{};
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    const auto err_out = error_buffer_.head(cost->Dimension());
    cost->ErrorVector(vars, err_out);
    output_errors.f += 0.5 * err_out.squaredNorm();
  }
  for (const ResidualBase::unique_ptr& eq : p_->equality_constraints) {
    const auto err_out = error_buffer_.head(eq->Dimension());
    eq->ErrorVector(vars, err_out);
    if (equality_norm == Norm::L1) {
      output_errors.equality += err_out.lpNorm<1>();
    } else if (equality_norm == Norm::QUADRATIC) {
      output_errors.equality += 0.5 * err_out.squaredNorm();
    }
  }
  return output_errors;
}

// TODO(gareth): Investigate an approach like algorithm 11.5?
NLSTerminationState ConstrainedNonlinearLeastSquares::UpdateLambdaAndCheckExitConditions(
    const Params& params, const StepSizeSelectionResult& step_result, const Errors& initial_errors,
    const double penalty, double* const lambda) {
  ASSERT(lambda != nullptr);

  if (step_result == StepSizeSelectionResult::SUCCESS) {
    ASSERT(!steps_.empty(), "Must have logged a step");
    // Update the state, and decrease lambda.
    prev_variables_.swap(variables_);  //  save the current variables
    variables_.swap(candidate_vars_);  //  replace w/ the candidate variables
    state_ = OptimizerState::NOMINAL;
    *lambda = std::max(*lambda * 0.1, params.min_lambda);

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
      *lambda = std::max(params.lambda_failure_init, *lambda);
      state_ = OptimizerState::ATTEMPTING_RESTORE_LM;
    } else {
      ASSERT(state_ == OptimizerState::ATTEMPTING_RESTORE_LM);
      // We are already attempting to recover and failing, ramp up lambda.
      *lambda *= 10;
    }
    if (*lambda > params.max_lambda) {
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
    const LineSearchStrategy& strategy, const double backtrack_search_tau,
    const Norm& equality_norm) {
  steps_.clear();

  // compute the directional derivative, w/ the current penalty
  const double directional_derivative = derivatives.Total(penalty);

  // iterate to find a viable alpha, or give up
  double alpha{1};
  for (int iter = 0; iter <= max_iterations; ++iter) {
    // compute nwe alpha value
    if (strategy == LineSearchStrategy::POLYNOMIAL_APPROXIMATION) {
      alpha = ComputeAlphaPolynomialApproximation(iter, alpha, errors_pre, derivatives, penalty);
    } else {
      ASSERT(strategy == LineSearchStrategy::ARMIJO_BACKTRACK);
      if (iter > 0) {
        alpha = alpha * backtrack_search_tau;
      }
    }

    // Update our candidate state.
    RetractCandidateVars(alpha);

    // Compute errors.
    const Errors errors_step = EvaluateNonlinearErrors(candidate_vars_, equality_norm);
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
  ASSERT(iteration >= 0);
  if (iteration == 0) {
    return alpha;
  } else if (iteration == 1) {
    ASSERT(steps_.size() == 1);
    // Pick a new alpha by approximating cost as a quadratic.
    // steps_.back() here is the "full step" error.
    const LineSearchStep& prev_step = steps_.back();
    const double new_alpha =
        QuadraticApproxMinimum(errors_pre.Total(penalty), derivatives.Total(penalty),
                               prev_step.alpha, prev_step.errors.Total(penalty));
    ASSERT(new_alpha < prev_step.alpha, "Alpha must decrease, alpha = %f, prev_alpha = %f", alpha,
           prev_step.alpha);
    return new_alpha;
  }
  ASSERT(steps_.size() >= 2);
  // Try the cubic approximation.
  const LineSearchStep& second_last_step = steps_[steps_.size() - 2];
  const LineSearchStep& last_step = steps_.back();

  // Compute coefficients
  const Eigen::Vector2d ab = CubicApproxCoeffs(
      errors_pre.Total(penalty), derivatives.Total(penalty), second_last_step.alpha,
      second_last_step.errors.Total(penalty), last_step.alpha, last_step.errors.Total(penalty));

  // Solve.
  const double new_alpha = CubicApproxMinimum(derivatives.Total(penalty), ab);
  ASSERT(new_alpha < last_step.alpha,
         "Alpha must decrease in the line search, alpha = %f, prev_alpha = %f", alpha,
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
    const QP& qp, const Eigen::VectorXd& dx, const Norm& equality_norm) {
  // We want the first derivative of the cost function, evaluated at the current linearization
  // point.
  //  d( 0.5 * h(x + dx * alpha)^T * h(x + dx * alpha) ) =
  //    h(x + dx * alpha)^T * dh(y)/dy * dx, evaluated at y = x, alpha = 0
  //
  //  For the quadratic part, this is just: c^T * dx, since c = J(x)^T * f(x)
  //  For the equality constraint, we compute J(x)^T * f(x) here explicitly.
  ASSERT(qp.c.rows() == dx.rows(), "Mismatch between dx and c");
  ASSERT(qp.A_eq.cols() == dx.rows(), "Mismatch between dx and A_eq");
  DirectionalDerivatives out{};
  out.d_f = qp.c.dot(dx);

  // Now we do the equality constraints, which for now have an L1 norm:
  //    d|c(x + alpha * dx)|/dalpha = d|v|/dv * dc(x)/dx * dx
  // Where d|v|/dv evaluates to the sign of each element.
  if (equality_norm == Norm::L1) {
    for (int i = 0; i < qp.A_eq.rows(); ++i) {
      const double sign_ci = Sign(qp.b_eq[i]);
      out.d_equality += sign_ci * qp.A_eq.row(i).dot(dx);
    }
  } else if (equality_norm == Norm::QUADRATIC) {
    // Simple L2 squared cost.
    for (int i = 0; i < qp.A_eq.rows(); ++i) {
      out.d_equality += qp.b_eq[i] * qp.A_eq.row(i).dot(dx);
    }
  }
  return out;
}

double ConstrainedNonlinearLeastSquares::SelectPenalty(const Norm& norm_type,
                                                       const ConstVectorBlock& lagrange_multipliers,
                                                       double equality_cost) {
  if (norm_type == Norm::L1) {
    // Equation 18.32. Note that this is not the recommended algorithm in the book, instead
    // they suggest 18.33. But this seems to work better for me.
    return lagrange_multipliers.lpNorm<Eigen::Infinity>();
  } else {
    // norm_type == Norm::QUADRATIC
    // See: http://www.numerical.rl.ac.uk/people/nimg/oumsc/lectures/part5.2.pdf
    const double l2_eq = std::sqrt(equality_cost);
    if (l2_eq == 0) {
      // If the cost is zero, the penalty will be multiplied by zero.
      return 0;
    }
    // Ratio of the two L2 norms (non quadratic).
    return lagrange_multipliers.norm() / l2_eq;
  }
}

// Equation (18.33)
// The idea here is to compute a value of the penalty (mu in the textbook, but not the same mu
// as used in the IP solver, of course) that will result in `dx` being a descent direction of
// the aggregated cost:
//    phi(x) = f(x) + mu * |c(x)|, where we are taking the L1 norm of the equality
// constraint c(x). This works because for the L1 norm you can show that the derivative of the
// aggregated cost is:
//    dphi(x + alpha * dx)/dalpha = df(x)^T * dx - mu * |c(x)|
// Then (and this seems arbitrary, AFAIK), you require that the derivative satisfy:
//   df(x)^T * dx - mu * |c(x)| <=-rho * |c(x)|
// Where rho is a parameter between [0, 1). Small rho will lead to lower penalty, and rho near
// 1 will lead to a massive penalty. In practice on some problems I have found that this
// approximation does not play nice with the quadratic approximation line search, in that it
// produces large penalties as |c(x)| -> 0, resulting in tiny step sizes and very slow convergence.
double ConstrainedNonlinearLeastSquares::ComputeEqualityPenalty(double d_f, double c, double rho) {
  ASSERT(rho < 1);
  ASSERT(rho >= 0);
  if (c == 0) {
    return 0;
  }
  return -d_f / ((1 - rho) * c);
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
  ASSERT(alpha_0 > 0);
  ASSERT(phi_prime_0 < 0);
  const double numerator = 2 * (phi_alpha_0 - phi_0 - phi_prime_0 * alpha_0);
  ASSERT(numerator > 0, "phi_alpha_0=%f, phi_0=%f, alpha_0=%f", phi_alpha_0, phi_0, alpha_0);
  return -phi_prime_0 * alpha_0 * alpha_0 / numerator;
}

// This function fits a cubic polynomial of the form:
//
//  a*x^3 + b*x^2 + x*phi_prime_0 + phi_0
//
Eigen::Vector2d ConstrainedNonlinearLeastSquares::CubicApproxCoeffs(
    const double phi_0, const double phi_prime_0, const double alpha_0, const double phi_alpha_0,
    const double alpha_1, const double phi_alpha_1) {
  ASSERT(alpha_1 > 0);
  ASSERT(alpha_0 > alpha_1, "This must be satisfied for the system to be solvable");
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
  ASSERT(std::abs(ab[0]) > 0, "Coefficient a cannot be zero");
  const double arg_sqrt = ab[1] * ab[1] - 3 * ab[0] * phi_prime_0;
  constexpr double kNegativeTol = -1.0e-12;
  ASSERT(arg_sqrt >= kNegativeTol, "This term must be positive: a=%f, b=%f, phi_prime_0=%f", ab[0],
         ab[1], phi_prime_0);
  return (std::sqrt(std::max(arg_sqrt, 0.)) - ab[1]) / (3 * ab[0]);
}

// These objects are passed as arguments to a static function for ease of testing this method.
void ConstrainedNonlinearLeastSquares::ComputeSecondOrderCorrection(
    const Eigen::VectorXd& updated_x,
    const std::vector<ResidualBase::unique_ptr>& equality_constraints, QP* qp,
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>* solver, Eigen::VectorXd* dx_out) {
  ASSERT(qp);
  ASSERT(solver);
  ASSERT(dx_out);

  // we use the QP `b` vector as storage for this operation
  int row = 0;
  for (const ResidualBase::unique_ptr& eq : equality_constraints) {
    const int dim = eq->Dimension();
    ASSERT(row + dim <= qp->b_eq.rows(), "Insufficient rows in vector b");
    eq->ErrorVector(updated_x, qp->b_eq.segment(row, dim));
    row += dim;
  }

  // compute the pseudo-inverse
  solver->compute(qp->A_eq);
  dx_out->noalias() -= solver->solve(qp->b_eq);
}

}  // namespace mini_opt
