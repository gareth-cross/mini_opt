// Copyright 2020 Gareth Cross
#pragma once
#include <map>
#include <sstream>

#include "mini_opt/structs.hpp"

namespace mini_opt {

// Accumulate counts for testing.
struct StatCounters {
  enum Stats : int32_t {
    NUM_NLS_ITERATIONS = 0,
    NUM_QP_ITERATIONS,
    NUM_FAILED_LINE_SEARCHES,
  };

  // All the counts.
  std::map<StatCounters::Stats, int> counts;

  StatCounters operator+(const StatCounters& c);
  StatCounters& operator+=(const StatCounters& c);
};

std::ostream& operator<<(std::ostream& s, const StatCounters::Stats& val);

/**
 * For use primarily in tests. Logger receives callbacks w/ iteration info
 * and records it in a string stream. We print it later if the test fails.
 */
struct Logger {
  explicit Logger(bool print_qp_variables = false, bool print_nonlinear_variables = false)
      : print_qp_variables_(print_qp_variables),
        print_nonlinear_variables_(print_nonlinear_variables) {}

  // Callback for the QP solver.
  void QPSolverCallback(const QPInteriorPointSolver& solver, const KKTError& kkt_prev,
                        const KKTError& kkt_after, const IPIterationOutputs& outputs);

  // Callback for the nonlinear solver.
  void NonlinearSolverCallback(const ConstrainedNonlinearLeastSquares& solver,
                               const NLSLogInfo& info);

  // Get the resulting string from the stream.
  std::string GetString() const;

  // Access the stream.
  std::stringstream& stream() { return stream_; }

  // Get the counters.
  StatCounters counters() const { return counters_; }

 private:
  const bool print_qp_variables_;
  const bool print_nonlinear_variables_;
  std::stringstream stream_;
  StatCounters counters_{};
};

}  // namespace mini_opt
