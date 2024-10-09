// Copyright 2021 Gareth Cross
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
    NUM_LINE_SEARCH_STEPS,
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
  // Callback for the QP solver.
  void QPSolverCallback(const QPInteriorPointSolver& solver, const KKTError& kkt_prev,
                        const KKTError& kkt_after, const IPIterationOutputs& outputs);

  // Callback for the nonlinear solver (returns true).
  bool NonlinearSolverCallback(const ConstrainedNonlinearLeastSquares& solver,
                               const NLSLogInfo& info);

  // Get the resulting string from the stream.
  std::string GetString() const;

  // Access the stream.
  std::stringstream& stream() { return stream_; }

  // Get the counters.
  const StatCounters& counters() const { return counters_; }

  // Get a count for a given enum value.
  int GetCount(const StatCounters::Stats& v) const {
    return (counters_.counts.count(v) > 0) ? counters_.counts.at(v) : 0;
  }

  // Enable or disable use of colors
  constexpr void SetUseColors(bool c) noexcept { use_colors_ = c; }

 private:
  bool use_colors_{true};
  std::stringstream stream_;
  StatCounters counters_{};
};

}  // namespace mini_opt
