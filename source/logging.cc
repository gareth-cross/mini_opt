// Copyright 2021 Gareth Cross
#include "mini_opt/logging.hpp"

#include <fmt/ostream.h>

#include "mini_opt/nonlinear.hpp"

namespace mini_opt {

static const Eigen::IOFormat kMatrixFmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

#define GREEN (112)
#define RED (160)
#define NO_COLOR (-1)

struct Color {
  Color(int code, bool enabled) : code(code), enabled(enabled) {}

  const int code;
  const int enabled;
};

std::ostream& operator<<(std::ostream& stream, const Color& c) {
  if (c.enabled) {
    if (c.code >= 0) {
      stream << fmt::format("\u001b[38;5;{}m", c.code);
    } else {
      stream << "\u001b[0m";
    }
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

void Logger::QPSolverCallback(const QPInteriorPointSolver&, const KKTError& kkt_prev,
                              const KKTError& kkt_after, const IPIterationOutputs& outputs) {
  counters_.counts[StatCounters::NUM_QP_ITERATIONS]++;
  stream_ << fmt::format(
      "Iteration summary: "
      "||kkt|| max: {} --> {}, mu = {}, a_p = {}, a_d = {}\n",
      kkt_prev.Max(), kkt_after.Max(), outputs.mu, outputs.alpha.primal, outputs.alpha.dual);

  if (!std::isnan(outputs.mu_affine)) {
    // print only if filled...
    stream_ << fmt::format(" Probe alphas: a_p = {}, a_d = {}, mu_affine = {}\n",
                           outputs.alpha_probe.primal, outputs.alpha_probe.dual, outputs.mu_affine);
  }

  // dump progress of individual KKT conditions
  stream_ << fmt::format(
      " KKT errors, L2:\n"
      " r_dual = {} --> {}\n"
      " r_comp = {} --> {}\n"
      " r_p_eq = {} --> {}\n"
      " r_p_ineq = {} --> {}\n",
      kkt_prev.r_dual, kkt_after.r_dual, kkt_prev.r_comp, kkt_after.r_comp, kkt_prev.r_primal_eq,
      kkt_after.r_primal_eq, kkt_prev.r_primal_ineq, kkt_after.r_primal_ineq);

#if 0
  if (print_qp_variables_) {
    // dump the state with labels
    stream_ << fmt::format(
        " Variables post-update:\n"
        "  x = {}\n"
        "  s = {}\n"
        "  y = {}\n"
        "  z = {}\n",
        fmt::streamed(solver.x_block().transpose().format(kMatrixFmt)),
        fmt::streamed(solver.s_block().transpose().format(kMatrixFmt)),
        fmt::streamed(solver.y_block().transpose().format(kMatrixFmt)),
        fmt::streamed(solver.z_block().transpose().format(kMatrixFmt)));
  }
#endif
  // summarize where the inequality constraints are
#if 0
  stream_ << " Constraints:\n";
  int i = 0;
  for (const LinearInequalityConstraint& c : solver.problem().constraints) {
    stream_ << fmt::format(
        "  Constraint {}: a * x[{}] + b - s = {} = ({} * {:.6f} + {:.6f} - {:.6f})\n", i,
        c.variable, c.a * solver.x_block()[c.variable] + c.b - solver.s_block()[i], c.a,
        solver.x_block()[c.variable], c.b, solver.s_block()[i]);
    ++i;
  }
#endif
}

bool Logger::NonlinearSolverCallback(const ConstrainedNonlinearLeastSquares&,
                                     const NLSLogInfo& info) {
  counters_.counts[StatCounters::NUM_NLS_ITERATIONS]++;
  counters_.counts[StatCounters::NUM_LINE_SEARCH_STEPS] += static_cast<int>(info.steps.size());
  if (info.termination_state != NLSTerminationState::MAX_LAMBDA &&
      info.termination_state != NLSTerminationState::MAX_ITERATIONS) {
    stream_ << Color(GREEN, use_colors_);
  } else {
    stream_ << Color(RED, use_colors_);
  }
  stream_ << fmt::format("Iteration # {}, state = {}, lambda = {}\n", info.iteration,
                         fmt::streamed(info.optimizer_state), info.lambda);
  stream_ << fmt::format("  f(0): {:.16e}, c(0): {:.16e}, total: {:.16e}\n", info.errors_initial.f,
                         info.errors_initial.equality, info.errors_initial.Total(info.penalty));
  stream_ << fmt::format("  min, max, |min| eig = {:.16e}, {:.16e}, {:.16e}\n",
                         info.qp_eigenvalues.min, info.qp_eigenvalues.max,
                         info.qp_eigenvalues.abs_min);
  stream_ << fmt::format("  termination = {}\n", fmt::streamed(info.termination_state));
  stream_ << fmt::format("  penalty = {:.16f}\n", info.penalty);
  stream_ << fmt::format("  QP: {}, {}\n", fmt::streamed(info.qp_term_state.termination_state),
                         info.qp_term_state.num_iterations);
  stream_ << fmt::format("  df/dalpha = {}, dc/dalpha = {}\n", info.directional_derivatives.d_f,
                         info.directional_derivatives.d_equality);
  stream_ << Color(NO_COLOR, use_colors_);

  if (info.step_result == StepSizeSelectionResult::SUCCESS) {
    stream_ << Color(GREEN, use_colors_);
  } else {
    if (info.step_result != StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED) {
      counters_.counts[StatCounters::NUM_FAILED_LINE_SEARCHES]++;
    }
    stream_ << Color(RED, use_colors_);
  }
  stream_ << fmt::format("  Search result: {}\n", fmt::streamed(info.step_result))
          << Color(NO_COLOR, use_colors_);

  int i = 0;
  for (const LineSearchStep& step : info.steps) {
    stream_ << fmt::format("  f({}): {:.16e}, c({}): {:.16e}, total: {:.16e}, alpha = {:.10f}\n", i,
                           step.errors.f, i, step.errors.equality, step.errors.Total(info.penalty),
                           step.alpha);
    ++i;
  }

// TODO: Get rid of this and log such values in a smarter way.
#if 0
  const QPInteriorPointSolver& qp = solver.solver();
  const auto s_block = qp.s_block();
  if (s_block.rows() > 0) {
    stream_ << fmt::format("  Slack variables: {}\n",
                           fmt::streamed(s_block.transpose().format(kMatrixFmt)));
  }

  // print extra details
  if (print_nonlinear_variables_) {
    stream_ << fmt::format(
        "  Variables:\n"
        "    x_old = {}\n"
        "    x_new = {}\n",
        fmt::streamed(solver.previous_variables().transpose().format(kMatrixFmt)),
        fmt::streamed(solver.variables().transpose().format(kMatrixFmt)));
  }
#endif
  return true;
}

std::string Logger::GetString() const { return stream_.str(); }

}  // namespace mini_opt
