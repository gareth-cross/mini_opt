// Copyright 2020 Gareth Cross
#pragma once
#include <sstream>

#include "mini_opt/structs.hpp"

namespace mini_opt {

/**
 * For use primarily in tests. Logger receives callbacks w/ iteration info
 * and records it in a string stream. We print it later if the test fails.
 */
struct Logger {
  Logger(bool print_qp_variables = false, bool print_nonlinear_variables = false)
      : print_qp_variables_(print_qp_variables),
        print_nonlinear_variables_(print_nonlinear_variables) {}

  // Callback for the QP solver.
  void QPSolverCallback(const QPInteriorPointSolver& solver, const KKTError& kkt2_prev,
                        const KKTError& kkt2_after, const IPIterationOutputs& outputs);

  // Callback for the nonlinear solver.
  void NonlinearSolverCallback(const ConstrainedNonlinearLeastSquares& solver,
                               const NLSLogInfo& info);

  // Get the resulting string from the stream.
  std::string GetString() const;

 private:
  const bool print_qp_variables_;
  const bool print_nonlinear_variables_;
  std::stringstream stream_;
};

}  // namespace mini_opt
