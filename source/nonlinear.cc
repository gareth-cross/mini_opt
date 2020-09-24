// Copyright 2020 Gareth Cross
#include "mini_opt/nonlinear.hpp"

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

void ConstrainedNonlinearLeastSquares::Solve(const Params& params,
                                             const Eigen::VectorXd& variables) {
  ASSERT(p_ != nullptr, "Must have a valid problem");
  ASSERT(static_cast<Eigen::Index>(p_->dimension) == variables.rows(),
         "Variables must match problem dimension");
  variables_ = variables;

  //
  for (int iter = 0; iter < params.max_iterations; ++iter) {
    // Fill out the QP and compute current errors.
    const Errors errors_iter_start = LinearizeAndFillQP(/* lambda */ 0.0);

    // Set up params.
    QPInteriorPointSolver::Params qp_params{};
    qp_params.max_iterations = params.max_qp_iterations;
    qp_params.termination_kkt2_tol = params.termination_kkt2_tolerance;

    // Solve the QP.
    const QPInteriorPointSolver::TerminationState term_state = solver_.Solve(qp_params);

    // Add the delta to our updated solution.
    candidate_vars_ = variables_;
    if (custom_retraction_) {
      custom_retraction_(&candidate_vars_,
                         const_cast<const QPInteriorPointSolver&>(solver_).x_block());
    } else {
      candidate_vars_ += solver_.x_block();
    }

    // Evaluate the errors again after taking a step.
    const Errors errors_iter_end = EvaluateNonlinearErrors(candidate_vars_);

    // todo: smart logic here
    const bool keep_solution = true;
    if (keep_solution) {
      variables_.swap(candidate_vars_);
    }
  }

  // std::cout << "lambda = " << lambda << std::endl;
  // std::cout << "error before = " << total_l2 << std::endl;
  // std::cout << "termination state = " << term_state << std::endl;

  // get the update and retract it onto the state
  variables_ += solver_.x_block();
  // for (int i = 0; i < variables_.rows(); ++i) {
  //  variables_[i] = Mod2Pi(variables_[i]);
  //}

  // std::cout << "variables: " << variables_.transpose() << "\n";

  // compute the error after
  /*double total_l2_after = 0;
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    total_l2_after += cost->Error(variables_);
  }
  std::cout << "error after = " << total_l2_after << std::endl;*/
}

Errors ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(const double lambda) {
  Errors output_errors{};

  // zero out the linear system before adding all the costs to it
  qp_.G.setZero();
  qp_.c.setZero();
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    output_errors.total_l2 += cost->UpdateHessian(variables_, &qp_.G, &qp_.c);
  }
  if (lambda > 0) {
    qp_.G.diagonal().array() += lambda;
  }

  // linearize equality constraints
  qp_.A_eq.setZero();
  qp_.b_eq.setZero();
  int row = 0;
  for (const ResidualBase::unique_ptr& eq : p_->equality_constraints) {
    const int dim = eq->Dimension();
    output_errors.equality_l2 +=
        eq->UpdateJacobian(variables_, qp_.A_eq.middleRows(row, dim), qp_.b_eq.segment(row, dim));
    row += dim;
  }

  // shift constraints to the new linearization point:
  qp_.constraints.clear();
  for (const LinearInequalityConstraint& c : p_->inequality_constraints) {
    qp_.constraints.push_back(c.ShiftTo(variables_));
  }

  // set up the optimizer
  solver_.Setup(&qp_);
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
}  // namespace mini_opt
