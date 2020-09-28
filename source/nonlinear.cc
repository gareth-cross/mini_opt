// Copyright 2020 Gareth Cross
#include "mini_opt/nonlinear.hpp"

#include <Eigen/Dense>  //  for inverse()
#include <iostream>

namespace mini_opt {

ConstrainedNonlinearLeastSquares::ConstrainedNonlinearLeastSquares(const Problem* const problem,
                                                                   const Retraction& retraction)
    : p_(problem), custom_retraction_(retraction) {
  ASSERT(p_ != nullptr);
  ASSERT(p_->dimension > 0, "Need at least one variable");

  // allocate space
  qp_.G.resize(p_->dimension, p_->dimension);
  qp_.c.resize(p_->dimension);

  // determine the size of the equality constraint matrix
  std::size_t total_eq_size = 0;
  for (const ResidualBase::unique_ptr& residual : p_->equality_constraints) {
    total_eq_size += residual->Dimension();
  }
  qp_.A_eq.resize(total_eq_size, p_->dimension);
  qp_.b_eq.resize(total_eq_size);

  // we'll fill these out later
  qp_.constraints.reserve(p_->inequality_constraints.size());

  // leave uninitialized, we'll fill this in later
  variables_.resize(p_->dimension);
  candidate_vars_.resizeLike(variables_);
}

template <typename T>
static T Mod2Pi(T value) {
  if (value < 0) {
    return -Mod2Pi(-value);
  } else {
    constexpr T two_pi = static_cast<T>(2 * M_PI);
    return std::fmod(value, two_pi);
  }
}

static void CheckParams(const ConstrainedNonlinearLeastSquares::Params& params) {
  ASSERT(params.max_iterations >= 0);
  ASSERT(params.max_qp_iterations >= 1);
  ASSERT(params.termination_kkt2_tolerance > 0);
  ASSERT(params.absolute_exit_tol > 0);
  ASSERT(params.max_line_search_iterations >= 0);
  ASSERT(params.relative_exit_tol >= 0);
  ASSERT(params.relative_exit_tol <= 1);
  ASSERT(params.absolute_first_derivative_tol >= 0);
}

NLSTerminationState ConstrainedNonlinearLeastSquares::Solve(const Params& params,
                                                            const Eigen::VectorXd& variables) {
  ASSERT(p_ != nullptr, "Must have a valid problem");
  CheckParams(params);
  variables_ = variables;

  // Iterate until max.
  double lambda{params.lambda_initial};
  for (int iter = 0; iter < params.max_iterations; ++iter) {
    // Fill out the QP and compute current errors.
    const Errors errors_pre = LinearizeAndFillQP(variables_, lambda, *p_, &qp_);

    // Set up params.
    QPInteriorPointSolver::Params qp_params{};
    qp_params.max_iterations = params.max_qp_iterations;
    qp_params.termination_kkt_tol = params.termination_kkt2_tolerance;
    qp_params.initial_mu = 1.0;
    qp_params.sigma = 0.1;
    qp_params.initialize_mu_with_complementarity = true;
    if (iter == 0) {
      qp_params.initial_guess_method = InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED;
    }

    // Solve the QP.
    solver_.Setup(&qp_);
    const QPSolverOutputs qp_outputs = solver_.Solve(qp_params);

    // Compute the directional derivative of the cost function about the current linearization
    // point, in the direction of the QP step.
    const double directional_deriv = ComputeQPCostDerivative(qp_, solver_.x_block());

    // Select the step size.
    const StepSizeSelectionResult step_result =
        SelectStepSize(params.max_line_search_iterations, params.absolute_first_derivative_tol,
                       errors_pre, directional_deriv);

    // Check if we should terminate.
    const NLSTerminationState maybe_exit =
        UpdateLambdaAndCheckExitConditions(params, step_result, errors_pre, &lambda);
    if (logging_callback_) {
      const NLSLogInfo info{iter,   lambda,    errors_pre, qp_outputs, directional_deriv,
                            steps_, maybe_exit};
      logging_callback_(*this, info);
    }

    if (maybe_exit != NLSTerminationState::NONE) {
      return maybe_exit;
    }
  }
  return NLSTerminationState::MAX_ITERATIONS;
}

void ConstrainedNonlinearLeastSquares::RetractCandidateVars(const double alpha) {
  candidate_vars_ = variables_;
  if (custom_retraction_) {
    const QPInteriorPointSolver& const_solver = solver_;
    custom_retraction_(&candidate_vars_, const_solver.x_block(), alpha);
  } else {
    candidate_vars_ += solver_.x_block() * alpha;
  }
}

Errors ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(const Eigen::VectorXd& variables,
                                                            const double lambda,
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
    output_errors.total_l2 += cost->UpdateHessian(variables, &qp->G, &qp->c);
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
    output_errors.equality_l2 +=
        eq->UpdateJacobian(variables, qp->A_eq.middleRows(row, dim), qp->b_eq.segment(row, dim));
    row += dim;
  }

  // shift constraints to the new linearization point:
  qp->constraints.clear();
  for (const LinearInequalityConstraint& c : problem.inequality_constraints) {
    qp->constraints.push_back(c.ShiftTo(variables));
  }
  return output_errors;
}

Errors ConstrainedNonlinearLeastSquares::EvaluateNonlinearErrors(const Eigen::VectorXd& vars) {
  Errors output_errors{};
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    output_errors.total_l2 += cost->Error(vars);
  }
  for (const ResidualBase::unique_ptr& eq : p_->equality_constraints) {
    output_errors.equality_l2 += eq->Error(vars);
  }
  return output_errors;
}

NLSTerminationState ConstrainedNonlinearLeastSquares::UpdateLambdaAndCheckExitConditions(
    const Params& params, const StepSizeSelectionResult& step_result, const Errors& initial_errors,
    double* const lambda) {
  ASSERT(!steps_.empty(), "Must have logged an attempted step");
  ASSERT(lambda != nullptr);

  if (step_result == StepSizeSelectionResult::SUCCESS) {
    // Update the state, and decrease lambda.
    variables_.swap(candidate_vars_);
    *lambda = std::max(*lambda * 0.1, params.min_lambda);

    // Check termination criteria.
    const LineSearchStep& final_step = steps_.back();
    if (final_step.errors.Total() < params.absolute_exit_tol) {
      // Satisfied absolute tolerance, exit.
      return NLSTerminationState::SATISFIED_ABSOLUTE_TOL;
    } else if (final_step.errors.Total() >
               initial_errors.Total() * (1 - params.relative_exit_tol)) {
      return NLSTerminationState::SATISFIED_RELATIVE_TOL;
    }
  } else if (step_result == StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED) {
    // The QP computed a direction where derivative of cost is zero.
    return NLSTerminationState::SATISFIED_FIRST_ORDER_TOL;
  } else if (step_result == StepSizeSelectionResult::FAILURE_MAX_ITERATIONS) {
    // We did not find an acceptable step, increase lambda.
    if (*lambda == 0) {
      *lambda = params.lambda_failure_init;
    } else {
      // TODO(gareth): Investigate an approach like algorithm 11.5
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
 * TODO(gareth): Implement Algorithm 3.5/3.6 and check the wolfe conditions properly?
 *
 * See equations 3.6a/3.6b
 *
 * For now this just checks the first step it can find that produces a decrease, up
 * to some maximum number of iterations.
 */
StepSizeSelectionResult ConstrainedNonlinearLeastSquares::SelectStepSize(
    const int max_iterations, const double abs_first_derivative_tol, const Errors& errors_pre,
    const double phi_prime_0) {
  steps_.clear();
  double alpha{1};
  for (int iter = 0; iter <= max_iterations; ++iter) {
    if (iter == 1) {
      // Pick a new alpha by approximating cost as a quadratic.
      // steps_.back() here is the "full step" error.
      const LineSearchStep& prev_step = steps_.back();
      alpha = QuadraticApproxMinimum(errors_pre.Total(), phi_prime_0, prev_step.alpha,
                                     prev_step.errors.Total());
      ASSERT(alpha < prev_step.alpha, "Alpha must decrease, alpha = %f, prev_alpha = %f", alpha,
             prev_step.alpha);
    } else if (iter > 1) {
      // Try the cubic approximation.
      const LineSearchStep& second_last_step = steps_[steps_.size() - 2];
      const LineSearchStep& last_step = steps_.back();

      // Compute coeffs.
      const Eigen::Vector2d ab = CubicApproxCoeffs(
          errors_pre.Total(), phi_prime_0, second_last_step.alpha, second_last_step.errors.Total(),
          last_step.alpha, last_step.errors.Total());

      // Solve.
      alpha = CubicApproxMinimum(phi_prime_0, ab);
      ASSERT(alpha < last_step.alpha,
             "Alpha must decrease in the line search, alpha = %f, prev_alpha = %f", alpha,
             last_step.alpha);
    }

    // Update our candidate state.
    RetractCandidateVars(alpha);

    // Compute errors.
    const Errors errors_step = EvaluateNonlinearErrors(candidate_vars_);
    steps_.emplace_back(alpha, errors_step);

    if (phi_prime_0 > -abs_first_derivative_tol) {
      // Either derivative is negative - but below the tolerance, or the derivative
      // is actually positive and we cannot decrease anymore.
      // TODO(gareth): Log a large positive value separately?
      return StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED;
    }

    if (errors_step.Total() < errors_pre.Total()) {
      return StepSizeSelectionResult::SUCCESS;
    }
  }
  // hit max iterations
  return StepSizeSelectionResult::FAILURE_MAX_ITERATIONS;
}

double ConstrainedNonlinearLeastSquares::ComputeQPCostDerivative(const QP& qp,
                                                                 const Eigen::VectorXd& dx) {
  // We want the first derivative of the cost function, evaluated at the current linearization
  // point.
  //  d( 0.5 * h(x + dx * alpha)^T * h(x + dx * alpha) ) =
  //    h(x + dx * alpha)^T * dh(y)/dy * dx, evaluated at y = x, alpha = 0
  //
  //  For the quadratic part, this is just: c^T * dx, since c = J(x)^T * f(x)
  //  For the equality constraint, we compute J(x)^T * f(x) here explicitly.
  ASSERT(qp.c.rows() == dx.rows(), "Mismatch between dx and c");
  ASSERT(qp.A_eq.cols() == dx.rows(), "Mismatch between dx and A_eq");
  double total = qp.c.dot(dx);
  // TODO(gareth): Check if this allocates?
  total += (qp.A_eq.transpose() * qp.b_eq).dot(dx);
  return total;
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
// a descent direction (as a result of the QP solver), it will be <= 0
double ConstrainedNonlinearLeastSquares::QuadraticApproxMinimum(const double phi_0,
                                                                const double phi_prime_0,
                                                                const double alpha_0,
                                                                const double phi_alpha_0) {
  ASSERT(phi_alpha_0 > phi_0);
  ASSERT(alpha_0 > 0);
  ASSERT(phi_prime_0 < 0);
  const double numerator = 2 * (phi_alpha_0 - phi_0 - phi_prime_0 * alpha_0);
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

}  // namespace mini_opt
