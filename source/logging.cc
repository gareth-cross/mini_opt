// Copyright 2020 Gareth Cross
#include "mini_opt/logging.hpp"

#include <iomanip>

#include "mini_opt/nonlinear.hpp"
#include "mini_opt/qp.hpp"

// TODO(gareth): Would really like to use libfmt for this instead...
namespace mini_opt {

static const Eigen::IOFormat kMatrixFmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

#define GREEN (112)
#define RED (160)
#define NO_COLOR (-1)

struct Color {
  explicit Color(int code) : code(code) {}

  const int code;
};

std::ostream& operator<<(std::ostream& stream, const Color& c) {
  if (c.code >= 0) {
    stream << "\u001b[38;5;" << c.code << "m";
  } else {
    stream << "\u001b[0m";
  }
  return stream;
}

StatCounters StatCounters::operator+(const StatCounters& c) {
  StatCounters out = *this;
  out += c;
  return out;
}

StatCounters& StatCounters::operator+=(const StatCounters& c) {
  for (const auto& pair : c.counts) {
    counts[pair.first] += pair.second;
  }
  return *this;
}

std::ostream& operator<<(std::ostream& s, const StatCounters::Stats& val) {
  switch (val) {
    case StatCounters::NUM_FAILED_LINE_SEARCHES:
      s << "NUM_FAILED_LINE_SEARCHES";
      break;
    case StatCounters::NUM_NLS_ITERATIONS:
      s << "NUM_NLS_ITERATIONS";
      break;
    case StatCounters::NUM_QP_ITERATIONS:
      s << "NUM_QP_ITERATIONS";
      break;
    case StatCounters::NUM_LINE_SEARCH_STEPS:
      s << "NUM_LINE_SEARCH_STEPS";
      break;
  }
  return s;
}

void Logger::QPSolverCallback(const QPInteriorPointSolver& solver, const KKTError& kkt_prev,
                              const KKTError& kkt_after, const IPIterationOutputs& outputs) {
  counters_.counts[StatCounters::NUM_QP_ITERATIONS]++;
  stream_ << "Iteration summary: ";
  stream_ << "||kkt|| max: " << kkt_prev.Max() << " --> " << kkt_after.Max()
          << ", mu = " << outputs.mu << ", a_p = " << outputs.alpha.primal
          << ", a_d = " << outputs.alpha.dual << "\n";

  if (!std::isnan(outputs.mu_affine)) {
    // print only if filled...
    stream_ << " Probe alphas: a_p = " << outputs.alpha_probe.primal
            << ", a_d = " << outputs.alpha_probe.dual << ", mu_affine = " << outputs.mu_affine
            << "\n";
  }

  // dump progress of individual KKT conditions
  stream_ << " KKT errors, L2:\n";
  stream_ << "  r_dual = " << kkt_prev.r_dual << " --> " << kkt_after.r_dual << "\n";
  stream_ << "  r_comp = " << kkt_prev.r_comp << " --> " << kkt_after.r_comp << "\n";
  stream_ << "  r_p_eq = " << kkt_prev.r_primal_eq << " --> " << kkt_after.r_primal_eq << "\n";
  stream_ << "  r_p_ineq = " << kkt_prev.r_primal_ineq << " --> " << kkt_after.r_primal_ineq
          << "\n";

  if (print_qp_variables_) {
    // dump the state with labels
    stream_ << " Variables post-update:\n";
    stream_ << "  x = " << solver.x_block().transpose().format(kMatrixFmt) << "\n";
    stream_ << "  s = " << solver.s_block().transpose().format(kMatrixFmt) << "\n";
    stream_ << "  y = " << solver.y_block().transpose().format(kMatrixFmt) << "\n";
    stream_ << "  z = " << solver.z_block().transpose().format(kMatrixFmt) << "\n";
  }
  // summarize where the inequality constraints are
#if 0
  stream_ << " Constraints:\n";
  std::size_t i = 0;
  for (const LinearInequalityConstraint& c : solver.problem().constraints) {
    stream_ << "  Constraint " << i << ": ax[" << c.variable
            << "] + b - s == " << c.a * solver.x_block()[c.variable] + c.b - solver.s_block()[i]
            << "  (" << c.a << " * " << solver.x_block()[c.variable] << " + " << c.b << " - "
            << solver.s_block()[i] << ")\n";
    ++i;
  }
#endif
}

void Logger::NonlinearSolverCallback(const ConstrainedNonlinearLeastSquares& solver,
                                     const NLSLogInfo& info) {
  counters_.counts[StatCounters::NUM_NLS_ITERATIONS]++;
  counters_.counts[StatCounters::NUM_LINE_SEARCH_STEPS] += info.steps.size();
  if (info.termination_state != NLSTerminationState::MAX_LAMBDA &&
      info.termination_state != NLSTerminationState::MAX_ITERATIONS) {
    stream_ << Color(GREEN);
  } else {
    stream_ << Color(RED);
  }
  stream_ << "Iteration #" << info.iteration << ", state = " << info.optimizer_state
          << ", lambda = " << info.lambda;
  stream_ << ", f(0): " << std::setprecision(std::numeric_limits<double>::max_digits10)
          << info.errors_initial.f << ", c(0): " << info.errors_initial.equality
          << ", total: " << info.errors_initial.Total(info.penalty) << "\n";
  stream_ << "  termination = " << info.termination_state << "\n";
  stream_ << "  penalty = " << info.penalty << "\n";
  stream_ << "  QP: " << info.qp_term_state.termination_state << ", "
          << info.qp_term_state.num_iterations << "\n";
  stream_ << "  df/dalpha = " << info.directional_derivatives.d_f
          << ", dc/dalpha = " << info.directional_derivatives.d_equality << "\n";
  stream_ << Color(NO_COLOR);
  if (info.step_result == StepSizeSelectionResult::SUCCESS) {
    stream_ << Color(GREEN);
  } else {
    if (info.step_result != StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED) {
      counters_.counts[StatCounters::NUM_FAILED_LINE_SEARCHES]++;
    }
    stream_ << Color(RED);
  }
  stream_ << "  Search result: " << info.step_result << "\n" << Color(NO_COLOR);

  int i = 0;
  for (const LineSearchStep& step : info.steps) {
    stream_ << "  f(" << i << "): " << std::setprecision(std::numeric_limits<double>::max_digits10)
            << step.errors.f << ", c(" << i << "): " << step.errors.equality
            << ", total: " << step.errors.Total(info.penalty) << ", alpha = " << step.alpha << "\n";
    ++i;
  }
  // print extra details
  if (print_nonlinear_variables_) {
    stream_ << "  Variables:\n";
    stream_ << "    x_old = " << solver.previous_variables().transpose().format(kMatrixFmt) << "\n";
    stream_ << "    x_new = " << solver.variables().transpose().format(kMatrixFmt) << "\n";
  }
}

std::string Logger::GetString() const { return stream_.str(); }

}  // namespace mini_opt
