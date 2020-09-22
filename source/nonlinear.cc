// Copyright 2020 Gareth Cross
#include "mini_opt/nonlinear.hpp"

#include <iostream>

namespace mini_opt {

ConstrainedNonlinearLeastSquares::ConstrainedNonlinearLeastSquares(const Problem* const problem)
    : p_(problem) {
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
}

// lazy man's mod-2pi
static double Mod2Pi(double value) {
  const std::complex<double> c{std::cos(value), std::sin(value)};
  return std::arg(c);
}

void ConstrainedNonlinearLeastSquares::LinearizeAndSolve(const double lambda) {
  ASSERT(p_ != nullptr);
  ASSERT(static_cast<Eigen::Index>(p_->dimension) == variables_.rows());

  // zero out the linear system before adding all the costs to it
  qp_.G.setZero();
  qp_.c.setZero();
  double total_l2 = 0;
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    total_l2 += cost->UpdateHessian(variables_, &qp_.G, &qp_.c);
  }
  qp_.G.diagonal().array() += lambda;

  // linearize equality constraints
  qp_.A_eq.setZero();
  qp_.b_eq.setZero();
  int row = 0;
  for (const ResidualBase::unique_ptr& eq : p_->equality_constraints) {
    const int dim = static_cast<int>(eq->Dimension());
    eq->UpdateJacobian(variables_, qp_.A_eq.middleRows(row, dim), qp_.b_eq.segment(row, dim));
    row += dim;
  }

  // shift constraints to the new linearization point:
  qp_.constraints.clear();
  for (const LinearInequalityConstraint& c : p_->inequality_constraints) {
    qp_.constraints.push_back(c.ShiftTo(variables_));
  }

  // TODO(gareth): Cache this object!
  // tune all these params.
  QPInteriorPointSolver solver(qp_, false);
  solver.SetLoggerCallback(qp_logger_callback_);

  // Initialize z near zero, so inequalities are initially inactive
  solver.z_block().setConstant(1.0e-3);

  QPInteriorPointSolver::Params params{};
  params.barrier_strategy = BarrierStrategy::PREDICTOR_CORRECTOR;
  params.max_iterations = 10;
  params.termination_kkt2_tol = 1.0e-8;

  // solve it
  const QPInteriorPointSolver::TerminationState term_state = solver.Solve(params);

  std::cout << "lambda = " << lambda << std::endl;
  std::cout << "error before = " << total_l2 << std::endl;
  std::cout << "termination state = " << term_state << std::endl;

  // get the update and retract it onto the state
  variables_ += solver.x_block();
  for (int i = 0; i < variables_.rows(); ++i) {
    variables_[i] = Mod2Pi(variables_[i]);
  }

  std::cout << "variables: " << variables_.transpose() << "\n";

  // compute the error after
  double total_l2_after = 0;
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    total_l2_after += cost->Error(variables_);
  }
  std::cout << "error after = " << total_l2_after << std::endl;
}

}  // namespace mini_opt
