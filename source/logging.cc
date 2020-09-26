// Copyright 2020 Gareth Cross
#include "mini_opt/logging.hpp"

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

void Logger::QPSolverCallback(const QPInteriorPointSolver& solver, const KKTError& kkt2_prev,
                              const KKTError& kkt2_after, const IPIterationOutputs& outputs) {
  stream_ << "Iteration summary: ";
  stream_ << "||kkt||^2: " << kkt2_prev.Total() << " --> " << kkt2_after.Total()
          << ", mu = " << outputs.mu << ", a_p = " << outputs.alpha.primal
          << ", a_d = " << outputs.alpha.dual << "\n";
  stream_ << " Probe alphas: a_p = " << outputs.alpha_probe.primal
          << ", a_d = " << outputs.alpha_probe.dual << ", mu_affine = " << outputs.mu_affine
          << "\n";

  // dump progress of individual KKT conditions
  stream_ << " KKT errors (squared):\n";
  stream_ << "  r_dual = " << kkt2_prev.r_dual << " --> " << kkt2_after.r_dual << "\n";
  stream_ << "  r_comp = " << kkt2_prev.r_comp << " --> " << kkt2_after.r_comp << "\n";
  stream_ << "  r_p_eq = " << kkt2_prev.r_primal_eq << " --> " << kkt2_after.r_primal_eq << "\n";
  stream_ << "  r_p_ineq = " << kkt2_prev.r_primal_ineq << " --> " << kkt2_after.r_primal_ineq
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
  if (info.termination_state != NLSTerminationState::MAX_LAMBDA &&
      info.termination_state != NLSTerminationState::MAX_ITERATIONS) {
    stream_ << Color(GREEN);
  } else {
    stream_ << Color(RED);
  }
  stream_ << "Iteration #" << info.iteration << ", lambda = " << info.lambda;
  stream_ << ", L2(0): " << info.errors_initial.total_l2
          << ", L2-eq(0): " << info.errors_initial.equality_l2
          << ", termination = " << info.termination_state << "\n";
  stream_ << "  QP: " << info.qp_term_state.termination_state << ", "
          << info.qp_term_state.num_iterations << "\n";
  stream_ << "  dphi(alpha)/dalpha = " << info.cost_directional_derivative << "\n";
  stream_ << Color(NO_COLOR);

  int i = 0;
  for (const LineSearchStep& step : info.steps) {
    stream_ << "  L2(" << i << "): " << step.errors.total_l2 << ", L2-eq(" << i
            << "): " << step.errors.equality_l2 << ", alpha = " << step.alpha << "\n";
    ++i;
  }
  if (print_nonlinear_variables_) {
    stream_ << "  Variables post update:\n";
    stream_ << "  x = " << solver.variables().transpose().format(kMatrixFmt) << "\n";
  }
}

std::string Logger::GetString() const { return stream_.str(); }

}  // namespace mini_opt
