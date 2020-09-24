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
  // Callback for the QP solver.
  void QPSolverCallback(const QPInteriorPointSolver& solver, const KKTError& kkt2_prev,
                        const KKTError& kkt2_after, const IPIterationOutputs& outputs);

  // Verbose callback for the QP solver.
  // Includes dump of state variables as well.
  void QPSolverCallbackVerbose(const QPInteriorPointSolver& solver, const KKTError& kkt2_prev,
                               const KKTError& kkt2_after, const IPIterationOutputs& outputs);

  // Callback for the nonlinear solver.
  void NonlinearSolverCallback(const int iteration, const Errors& errors_before,
                               const Errors& errors_after, const QPTerminationState& term_state);

  // Get the resulting string from the stream.
  std::string GetString() const;

 private:
  std::stringstream stream_;
};

}  // namespace mini_opt
