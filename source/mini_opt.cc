#include "mini_opt.hpp"

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
namespace mini_opt {

const IOFormat kMatrixFmt(FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

ResidualBase::~ResidualBase() {}

bool LinearInequalityConstraint::IsFeasible(double x) const {
  // There might be an argument to be made we should tolerate some epsilon > 0 here?
  return a * x - b < 0.0;
}

QPInteriorPointSolver::QPInteriorPointSolver(const QP& problem) : p_(problem) {
  ASSERT(p_.G.rows() == p_.G.cols(), "G must be square");
  ASSERT(p_.G.rows() == p_.c.rows(), "Dims of G and c must match");
  ASSERT(p_.A_eq.rows() == p_.b_eq.rows(), "Rows of A_e and b_e must match");

  // If equality constraints were specified, we must be able to multiply A*x
  if (p_.A_eq.size() > 0) {
    ASSERT(p_.A_eq.cols() == p_.G.rows());
  } else {
    ASSERT(p_.A_eq.rows() == 0);
    ASSERT(p_.b_eq.rows() == 0);
  }

  // Order is [slacks (s), equality multipliers(y), inequality multiplers (lambda)]
  dims_.N = p_.G.rows();
  dims_.M = p_.constraints.size();
  dims_.K = p_.A_eq.rows();

  variables_.resize(dims_.N + dims_.M * 2 + dims_.K);
  prev_variables_.resizeLike(variables_);

  // Since this is solving a problem in the tangent space of a larger nonlinear problem,
  // we can guess zero for `x`.
  XBlock(dims_, variables_).setZero();

  // TODO(gareth): A better initialization strategy for these?
  // Could assume constraints are active, in which case we compute lambda from the KKT conditions.
  SBlock(dims_, variables_).setConstant(1);
  ZBlock(dims_, variables_).setConstant(1);
  YBlock(dims_, variables_).setConstant(0);

  // Allocate space for solving
  const std::size_t reduced_system_size = dims_.N + dims_.K;
  H_.resize(reduced_system_size, reduced_system_size);
  H_.setZero();
  H_inv_.resizeLike(H_);

  // Use the total size here
  r_.resizeLike(variables_);
  r_dual_aug_.resize(dims_.N);

  // Space for the output of all variables
  delta_.resizeLike(variables_);
  delta_affine_.resizeLike(variables_);
  delta_affine_.setZero();

  const bool check_feasible = false;  // TODO(gareth): Param?
  for (const LinearInequalityConstraint& c : p_.constraints) {
    ASSERT(c.variable < static_cast<int>(dims_.N), "Constraint index is out of bounds");
    const bool is_feasible = c.IsFeasible(variables_[c.variable]);
    if (!is_feasible && check_feasible) {
      std::stringstream ss;
      ss << "Constraint is not feasible: " << c.a << " * x[" << c.variable << "] + " << c.b
         << " >= 0, x = " << variables_[c.variable];
      throw InfeasibleGuess(ss.str());
    }
  }
}

// Assert params are in valid range.
static void CheckParams(const QPInteriorPointSolver::Params& params) {
  ASSERT(params.initial_sigma > 0);
  ASSERT(params.initial_sigma <= 1.0);
  ASSERT(params.sigma_reduction > 0);
  ASSERT(params.sigma_reduction <= 1.0);
  ASSERT(params.termination_kkt2_tol > 0);
  ASSERT(params.max_iterations > 0);
}

QPInteriorPointSolver::TerminationState QPInteriorPointSolver::Solve(
    const QPInteriorPointSolver::Params& params) {
  CheckParams(params);

  // on the first iteration, the residual needs to be filled first
  EvaluateKKTConditions();

  double sigma{params.initial_sigma};
  for (int iter = 0; iter < params.max_iterations; ++iter) {
    // copy current state
    prev_variables_ = variables_;

    // compute squared norm of the residual, prior to any updates
    const double kkt2 = r_.squaredNorm();

    // solve for the update
    const IterationOutputs iteration_outputs = Iterate(sigma, params.barrier_strategy);

    // evaluate the residual again, which fills `r_` for the next iteration
    EvaluateKKTConditions();

    const double kkt2_after = r_.squaredNorm();
    if (logger_callback_) {
      // pass progress to the logger callback for printing in the test
      logger_callback_(kkt2, kkt2_after, iteration_outputs);
    }

    if (kkt2_after < params.termination_kkt2_tol) {
      // error is low enough, stop
      return TerminationState::SATISFIED_KKT_TOL;
    }

    // TODO(gareth): Try the strategy described by equation (19.20) here?
    // This probably decreases too quickly in some cases, and too slowly in others.
    sigma *= params.sigma_reduction;
  }

  return TerminationState::MAX_ITERATIONS;
}

QPInteriorPointSolver::IterationOutputs QPInteriorPointSolver::Iterate(
    const double sigma, const BarrierStrategy& strategy) {
  // fill out `r_`
  EvaluateKKTConditions();

  // evaluate the complementarity condition
  IterationOutputs outputs{};
  if (HasInequalityConstraints()) {
    outputs.mu = ConstSBlock(dims_, r_).sum() / static_cast<double>(dims_.M);
  }

  // solve the system w/ the LDLT factorization
  ComputeLDLT();

  if (!HasInequalityConstraints()) {
    // No inequality constraints, ignore mu & sigma.
    SolveForUpdate(0.0);
  } else if (strategy == BarrierStrategy::SCALED_COMPLEMENTARITY) {
    // Use the complementarity condition, and scale by sigma.
    SolveForUpdate(sigma * outputs.mu);
    outputs.sigma = sigma;
  } else if (strategy == BarrierStrategy::PREDICTOR_CORRECTOR) {
    // Use the MPC/predictor-corrector (algorithm 16.4).
    // Solve with mu=0 and compute the largest step size.
    SolveForUpdate(0.0);
    ComputeAlpha(&outputs.alpha_probe, /* tau = */ 1.0);

    // save the value of the affine step
    delta_affine_ = delta_;

    // Compute complementarity had we applied this update we just solved.
    outputs.mu_affine = ComputePredictorCorrectorMuAffine(outputs.mu, outputs.alpha_probe);
    outputs.sigma = std::pow(outputs.mu_affine / outputs.mu, 3);  // equation (19.22)

    // Solve again (alpha will be computed again below), input sigma is ignored.
    // This time, delta_affine_ will be incorporated to add the diag(dz)*ds term.
    SolveForUpdate(outputs.mu * outputs.sigma);
  }

  // compute alpha values
  if (HasInequalityConstraints()) {
    ComputeAlpha(&outputs.alpha, /* tau = */ 0.995);
  }

  // update
  XBlock(dims_, variables_).noalias() += ConstXBlock(dims_, delta_) * outputs.alpha.primal;
  SBlock(dims_, variables_).noalias() += ConstSBlock(dims_, delta_) * outputs.alpha.primal;
  YBlock(dims_, variables_).noalias() += ConstYBlock(dims_, delta_) * outputs.alpha.dual;
  ZBlock(dims_, variables_).noalias() += ConstZBlock(dims_, delta_) * outputs.alpha.dual;
  return outputs;
}

// TODO(gareth): All the accessing by index is a little gross, could maybe split
// up the variables into separate vectors? I do like having the full state as one object.
ConstVectorBlock QPInteriorPointSolver::x_block() const { return ConstXBlock(dims_, variables_); }

ConstVectorBlock QPInteriorPointSolver::s_block() const { return ConstSBlock(dims_, variables_); }

ConstVectorBlock QPInteriorPointSolver::y_block() const { return ConstYBlock(dims_, variables_); }

ConstVectorBlock QPInteriorPointSolver::z_block() const { return ConstZBlock(dims_, variables_); }

/*
 * We start with the full symmetric system (Equation 19.12):
 *
 * delta = [p_x; p_s; -p_y; -p_z]
 *
 * [[G       0        A_e^T     A_i^T]                 [r_d;
 *  [0       \Sigma   0        -I    ]     * delta = -  diag(s)^-1 * r_comp;
 *  [A_e     0        0         0    ]                  r_pe;
 *  [A_i    -I        0         0    ]]                 r_pi]
 *
 * Where r_d is the dual objective, \Sigma = diag(s)^-1 * Z, r_pe and r_pi are the primal
 * equality and inequality residuals. Note:
 *
 *  r_d = G*x + c - A_e^T * y - A_i^T * z
 *  r_comp = diag(s)*z - \mu * e
 *  r_pe = c_e(x) = (A_e * x + b_e)
 *  r_pi = c_i(x) - s = (A_i * x + b_i) - s
 *
 * Note that in the code below, we subtract `mu` from r_comp inline where it is
 * used, rather than putting it in `r_` directly.
 *
 * We first eliminate the second row using:
 *
 *  p_s = \Sigma^-1 * (-diag(s)^-1 * r_comp - I * p_z)
 *
 * Which reduces the system to:
 *
 * [[G    A_e^T  A_i^T    ]                [r_d;
 *  [A_e  0      0        ]    * delta = -  r_pe;
 *  [A_i  0     -\Sigma^-1]]                r_pi + \Sigma^-1 * diag(s)^-1 * r_comp]
 *
 * Then we reduce it again using:
 *
 *  p_z = \Sigma * (-A_i * -p_x - r_pi) - diag(s)^-1 * r_comp
 *
 * Such that:
 *
 * [[G + A_i^T * \Sigma * A_i   A_e^T]  * delta = -[r_x_aug;
 *  [A_e                        0    ]]             r_pe]
 *
 * Where: r_x_aug = r_d + A_i^T * \Sigma * r_pi + A_i^T * diag(s)^-1 * r_comp
 *
 * Then we solve this system, and substitute for the other variables. We leverage the fact
 * that \Sigma, diag(s), I, etc are diagonal matrices.
 *
 * It is assumed that `EvaluateKKTConditions` was called first.
 */
void QPInteriorPointSolver::ComputeLDLT() {
  const std::size_t N = dims_.N;
  const std::size_t M = dims_.M;
  const std::size_t K = dims_.K;

  // const-block expressions for these, for convenience
  const auto s = ConstSBlock(dims_, variables_);
  const auto z = ConstZBlock(dims_, variables_);

  // shouldn't happen due to selection of alpha, but double check
  const bool any_non_positive_s = (s.array() <= 0.0).any();
  if (any_non_positive_s) {
    std::stringstream ss;
    ss << "Some slack variables s <= 0: " << s.transpose().format(kMatrixFmt);
    throw std::runtime_error(ss.str());
  }

  // build the left-hand side (we only need lower triangular)
  H_.topLeftCorner(N, N).triangularView<Eigen::Lower>() = p_.G.triangularView<Eigen::Lower>();
  if (K > 0) {
    H_.bottomLeftCorner(K, N) = p_.A_eq;
  }
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    H_(c.variable, c.variable) += c.a * (z[i] / s[i]) * c.a;
  }

  // factorize, TODO(gareth): preallocate ldlt...
  const LDLT<MatrixXd, Eigen::Lower> ldlt(H_);
  if (ldlt.info() != Eigen::ComputationInfo::Success) {
    std::stringstream ss;
    ss << "Failed to solve system:\n" << H_.format(kMatrixFmt) << "\n";
    throw FailedFactorization(ss.str());
  }

  // compute H^-1
  H_inv_.setIdentity();
  ldlt.solveInPlace(H_inv_);

  // clear update steps
  delta_.setZero();
  delta_affine_.setZero();
}

void QPInteriorPointSolver::SolveForUpdate(const double mu) {
  const std::size_t N = dims_.N;
  const std::size_t M = dims_.M;
  const std::size_t K = dims_.K;

  const auto s = ConstSBlock(dims_, variables_);
  const auto z = ConstZBlock(dims_, variables_);

  // create the right-hand side of the 'augmented system'
  const auto r_d = ConstXBlock(dims_, r_);
  const auto r_comp = ConstSBlock(dims_, r_);
  const auto r_pe = ConstYBlock(dims_, r_);
  const auto r_pi = ConstZBlock(dims_, r_);

  // relevant for MPC (zero if not doing the corrector step)
  const auto s_aff = ConstSBlock(dims_, delta_affine_);
  const auto z_aff = ConstZBlock(dims_, delta_affine_);

  // apply the variable elimination, which updates r_d (make a copy to save the original)
  r_dual_aug_.noalias() = r_d;
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    r_dual_aug_[c.variable] += c.a * (z[i] / s[i]) * r_pi[i];
    r_dual_aug_[c.variable] += c.a * (r_comp[i] + (s_aff[i] * z_aff[i]) - mu) / s[i];
  }

  // compute [px, -py]
  auto dx = XBlock(dims_, delta_);
  auto ds = SBlock(dims_, delta_);
  auto dy = YBlock(dims_, delta_);
  auto dz = ZBlock(dims_, delta_);

  dx.noalias() = H_inv_.block(0, 0, N, N) * -r_dual_aug_;
  if (K > 0) {
    dx.noalias() += H_inv_.block(0, N, N, K) * -r_pe;
    // Negate here since py is negative in the solution vector.
    dy.noalias() = H_inv_.block(N, 0, K, N) * r_dual_aug_;
    dy.noalias() += H_inv_.block(N, N, K, K) * r_pe;
  }

  // Go back and solve for dz and ds
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    ds[i] = c.a * dx[c.variable] + r_pi[i];
    dz[i] = -(z[i] / s[i]) * ds[i] - (1 / s[i]) * (r_comp[i] + (s_aff[i] * z_aff[i]) - mu);
  }
}

/*
 * Build the right hand side of the system illustrated in the comment on SolveForUpdate.
 */
void QPInteriorPointSolver::EvaluateKKTConditions() {
  const auto x = ConstXBlock(dims_, variables_);
  const auto s = ConstSBlock(dims_, variables_);
  const auto y = ConstYBlock(dims_, variables_);
  const auto z = ConstZBlock(dims_, variables_);

  // create the right-hand side
  auto r_d = XBlock(dims_, r_);
  auto r_comp = SBlock(dims_, r_);
  auto r_pe = YBlock(dims_, r_);
  auto r_pi = ZBlock(dims_, r_);

  // build the dual objective first
  r_d.noalias() = p_.G.selfadjointView<Eigen::Lower>() * x + p_.c;
  if (dims_.K > 0) {
    r_d.noalias() -= p_.A_eq.transpose() * y;
    // equality constraints
    r_pe.noalias() = p_.A_eq * x + p_.b_eq;
  }

  // contributions from inequality constraints, there is some redundant work here
  for (std::size_t i = 0; i < dims_.M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    r_d[c.variable] -= c.a * z[i];
    r_pi[i] = c.a * x[c.variable] + c.b - s[i];
    r_comp[i] = s[i] * z[i];
  }
}

// Formula 19.9
void QPInteriorPointSolver::ComputeAlpha(AlphaValues* const output, const double tau) const {
  ASSERT(output != nullptr);
  output->primal = ComputeAlpha(ConstSBlock(dims_, variables_), ConstSBlock(dims_, delta_), tau);
  output->dual = ComputeAlpha(ConstZBlock(dims_, variables_), ConstZBlock(dims_, delta_), tau);
}

double QPInteriorPointSolver::ComputeAlpha(const ConstVectorBlock& val,
                                           const ConstVectorBlock& d_val, const double tau) const {
  ASSERT(val.rows() == d_val.rows());
  ASSERT(tau > 0 && tau <= 1);
  double alpha = 1.0;
  // TODO(gareth): Make this value adjustable for possibly faster convergence?
  for (int i = 0; i < val.rows(); ++i) {
    const double updated_val = val[i] + d_val[i];
    if (updated_val <= 0.0 && std::abs(d_val[i]) > 0) {
      const double candidate_alpha = -tau * val[i] / d_val[i];
      if (candidate_alpha < alpha) {
        alpha = candidate_alpha;
      }
    }
  }
  return alpha;
}

// We don't re-evaluate the s^T * z / M term, because it is already stored in mu.
double QPInteriorPointSolver::ComputePredictorCorrectorMuAffine(
    const double mu, const AlphaValues& alpha_probe) const {
  const auto s = ConstSBlock(dims_, variables_);
  const auto z = ConstZBlock(dims_, variables_);
  const auto ds = ConstSBlock(dims_, delta_);
  const auto dz = ConstZBlock(dims_, delta_);
  // here we just compute the missing terms from (s + ds * a_p)^T * (z + dz * a_d)
  double mu_affine = mu;
  mu_affine += alpha_probe.dual * s.dot(dz) / static_cast<double>(dims_.M);
  mu_affine += alpha_probe.primal * z.dot(ds) / static_cast<double>(dims_.N);
  mu_affine += (alpha_probe.dual * alpha_probe.primal) * ds.dot(dz) / static_cast<double>(dims_.M);
  return mu_affine;
}

ConstVectorBlock QPInteriorPointSolver::ConstXBlock(const ProblemDims& dims,
                                                    const Eigen::VectorXd& vec) {
  return vec.head(dims.N);
}

ConstVectorBlock QPInteriorPointSolver::ConstSBlock(const ProblemDims& dims,
                                                    const Eigen::VectorXd& vec) {
  return vec.segment(dims.N, dims.M);
}

ConstVectorBlock QPInteriorPointSolver::ConstYBlock(const ProblemDims& dims,
                                                    const Eigen::VectorXd& vec) {
  return vec.segment(dims.N + dims.M, dims.K);
}

ConstVectorBlock QPInteriorPointSolver::ConstZBlock(const ProblemDims& dims,
                                                    const Eigen::VectorXd& vec) {
  return vec.tail(dims.M);
}

VectorBlock QPInteriorPointSolver::XBlock(const ProblemDims& dims, Eigen::VectorXd& vec) {
  return vec.head(dims.N);
}

VectorBlock QPInteriorPointSolver::SBlock(const ProblemDims& dims, Eigen::VectorXd& vec) {
  return vec.segment(dims.N, dims.M);
}

VectorBlock QPInteriorPointSolver::YBlock(const ProblemDims& dims, Eigen::VectorXd& vec) {
  return vec.segment(dims.N + dims.M, dims.K);
}

VectorBlock QPInteriorPointSolver::ZBlock(const ProblemDims& dims, Eigen::VectorXd& vec) {
  return vec.tail(dims.M);
}

/*
 * Just build this matrix directly:
 *
 * [[G       0        A_e^T     A_i^T]                 [r_d;
 *  [0       \Sigma   0        -I    ]     * delta = -  diag(s)^-1 * r_comp;
 *  [A_e     0        0         0    ]                  r_pe;
 *  [A_i    -I        0         0    ]]                 r_pi]
 *
 * We don't solve this system since it has a lot of empty blocks, and many of the
 * sub blocks are diagonal (I, \Sigma, diag(s)^-1, etc).
 */
void QPInteriorPointSolver::BuildFullSystem(Eigen::MatrixXd* const H,
                                            Eigen::VectorXd* const r) const {
  ASSERT(H != nullptr);
  ASSERT(r != nullptr);
  const std::size_t N = dims_.N;
  const std::size_t M = dims_.M;
  const std::size_t K = dims_.K;

  const auto x = ConstXBlock(dims_, variables_);
  const auto s = ConstSBlock(dims_, variables_);
  const auto y = ConstYBlock(dims_, variables_);
  const auto z = ConstZBlock(dims_, variables_);

  H->resize(N + K + M * 2, N + K + M * 2);
  H->setZero();
  r->resize(H->rows());
  r->setZero();

  H->topLeftCorner(N, N) = p_.G.selfadjointView<Eigen::Lower>();
  if (K > 0) {
    H->block(0, N + M, N, K) = p_.A_eq.transpose();
    H->block(N + M, 0, K, N) = p_.A_eq;
  }

  Eigen::MatrixXd A_i(M, N);
  Eigen::VectorXd b_i(M);
  if (M > 0) {
    A_i.setZero();
    // create sparse A_i for simplicity
    for (std::size_t i = 0; i < M; ++i) {
      const LinearInequalityConstraint& c = p_.constraints[i];
      A_i(i, c.variable) = c.a;
      b_i[i] = c.b;
    }

    H->topRightCorner(N, M) = A_i.transpose();
    H->bottomLeftCorner(M, N) = A_i;
    H->bottomRows(M).middleCols(N, M).diagonal().setConstant(-1);

    // Sigma, row for p_s
    H->block(N, N, M, M).diagonal() = z.array() / s.array();
    H->topRows(N + M).bottomRightCorner(M, M).diagonal().setConstant(-1);
  }

  auto r_d = XBlock(dims_, *r);
  auto s_inv_r_comp = SBlock(dims_, *r);
  auto r_pe = YBlock(dims_, *r);
  auto r_pi = ZBlock(dims_, *r);

  r_d.noalias() = p_.G.selfadjointView<Eigen::Lower>() * x + p_.c;
  if (K > 0) {
    r_d.noalias() -= p_.A_eq.transpose() * y;
    // equality constraints
    r_pe.noalias() = p_.A_eq * x + p_.b_eq;
  }
  if (M > 0) {
    r_d.noalias() -= A_i.transpose() * z;
    s_inv_r_comp = z;  //  mu = 0
    r_pi = A_i * x + b_i - s;
  }
}

std::ostream& operator<<(std::ostream& stream,
                         const QPInteriorPointSolver::TerminationState& state) {
  using TerminationState = QPInteriorPointSolver::TerminationState;
  switch (state) {
    case TerminationState::SATISFIED_KKT_TOL:
      stream << "SATISFIED_KKT_TOL";
      break;
    case TerminationState::MAX_ITERATIONS:
      stream << "MAX_ITERATIONS";
      break;
  }
  return stream;
}

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

void ConstrainedNonlinearLeastSquares::LinearizeAndSolve() {
  // zero out the linear system before adding all the costs to it
  qp_.G.setZero();
  qp_.c.setZero();
  double total_l2 = 0;
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    total_l2 += cost->UpdateHessian(variables_, &qp_.G, &qp_.c);
  }

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
  QPInteriorPointSolver solver(qp_);
  solver.SetLoggerCallback(qp_logger_callback_);

  QPInteriorPointSolver::Params params{};
  params.barrier_strategy = BarrierStrategy::PREDICTOR_CORRECTOR;
  params.max_iterations = 10;
  params.termination_kkt2_tol = 1.0e-5;

  // solve it
  const QPInteriorPointSolver::TerminationState term_state = solver.Solve(params);

  std::cout << "error before = " << total_l2 << std::endl;
  std::cout << "termination state = " << term_state << std::endl;

  // get the update and retract it onto the state
  variables_ += solver.x_block();

  // compute the error after
  double total_l2_after = 0;
  for (const ResidualBase::unique_ptr& cost : p_->costs) {
    total_l2_after += cost->Error(variables_);
  }
  std::cout << "error after = " << total_l2_after << std::endl;
}

}  // namespace mini_opt
