// Copyright 2020 Gareth Cross
#include "mini_opt/logging.hpp"

#include "mini_opt/qp.hpp"

// TODO(gareth): Would really like to use libfmt for this instead...
namespace mini_opt {

static const Eigen::IOFormat kMatrixFmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

void Logger::QPSolverCallback(const QPInteriorPointSolver& solver, const KKTError& kkt2_prev,
                              const KKTError& kkt2_after, const IPIterationOutputs& outputs) {
  (void)solver;  //  unused
  stream_ << "Iteration summary: ";
  stream_ << "||kkt||^2: " << kkt2_prev.Total() << " --> " << kkt2_after.Total()
          << ", mu = " << outputs.mu << ", sigma = " << outputs.sigma
          << ", a_p = " << outputs.alpha.primal << ", a_d = " << outputs.alpha.dual << "\n";
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
}

void Logger::QPSolverCallbackVerbose(const QPInteriorPointSolver& solver, const KKTError& kkt2_prev,
                                     const KKTError& kkt2_after,
                                     const IPIterationOutputs& outputs) {
  QPSolverCallback(solver, kkt2_prev, kkt2_after, outputs);

  // dump the state with labels
  stream_ << " Variables post-update:\n";
  stream_ << "  x = " << solver.x_block().transpose().format(kMatrixFmt) << "\n";
  stream_ << "  s = " << solver.s_block().transpose().format(kMatrixFmt) << "\n";
  stream_ << "  y = " << solver.y_block().transpose().format(kMatrixFmt) << "\n";
  stream_ << "  z = " << solver.z_block().transpose().format(kMatrixFmt) << "\n";

  // summarize where the inequality constraints are
  stream_ << " Constraints:\n";
  std::size_t i = 0;
  for (const LinearInequalityConstraint& c : solver.problem().constraints) {
    stream_ << "  Constraint " << i << ": ax[" << c.variable
            << "] + b - s == " << c.a * solver.x_block()[c.variable] + c.b - solver.s_block()[i]
            << "  (" << c.a << " * " << solver.x_block()[c.variable] << " + " << c.b << " - "
            << solver.s_block()[i] << ")\n";
    ++i;
  }
}

std::string Logger::GetString() const { return stream_.str(); }

}  // namespace mini_opt
