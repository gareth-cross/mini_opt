#include "mini_opt.hpp"

#include <Eigen/Dense>

using namespace Eigen;
namespace mini_opt {

const IOFormat kMatrixFmt(FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

ResidualBase::~ResidualBase() {}

bool LinearInequalityConstraint::IsFeasible(double x) const {
  // There might be an argument to be made we should tolerate some epsilon > 0 here?
  return a * x - b >= 0.0;
}

QPInteriorPointSolver::QPInteriorPointSolver(const QP& problem, const Eigen::VectorXd& x_guess)
    : p_(problem) {
  ASSERT(p_.G.rows() == p_.G.cols());
  ASSERT(p_.G.rows() == p_.c.rows());
  ASSERT(p_.A_eq.rows() == p_.b_eq.rows());

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

  // if a guess was provided, copy it in
  if (x_guess.size() > 0) {
    ASSERT(x_guess.rows() == static_cast<Eigen::Index>(dims_.N));
    XBlock(dims_, variables_) = x_guess;
  } else {
    // otherwise guess zero
    XBlock(dims_, variables_).setZero();
  }

  // TODO(gareth): A better initialization strategy for these?
  // Could assume constraints are active, in which case we compute lambda from the KKT conditions.
  SBlock(dims_, variables_).setConstant(1);
  ZBlock(dims_, variables_).setConstant(1);
  YBlock(dims_, variables_).setConstant(1);

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

QPInteriorPointSolver::TerminationState QPInteriorPointSolver::Solve(const Params& params) {
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
    const IterationOutputs iteration_outputs = Iterate(sigma);

    // evaluate the residual again, which fills `r_` for the next iteration
    EvaluateKKTConditions();
    const double kkt2_after = r_.squaredNorm();

    if (logger_callback_) {
      // pass progress to the logger callback for printing in the test
      logger_callback_(
          kkt2_after,
          std::min(iteration_outputs.alpha_primal, iteration_outputs.alpha_primal) * sigma,
          iteration_outputs.mu);
    }

    // check for termination
    if (kkt2_after > kkt2) {
      // newton step took us in a bad direction, roll back the update
      // if we want to try to recover, we need to reset `r_` here as well
      variables_ = prev_variables_;
      return TerminationState::BAD_STEP;
    }
    if (kkt2_after < params.termination_kkt2_tol) {
      // error is low enough, stop
      return TerminationState::SATISFIED_KKT_TOL;
    }

    // TODO(gareth): Implement one of the smarter strategies here.
    sigma *= params.sigma_reduction;
  }

  return TerminationState::MAX_ITERATIONS;
}

QPInteriorPointSolver::IterationOutputs QPInteriorPointSolver::Iterate(const double sigma) {
  // fill out `r_`
  EvaluateKKTConditions();

  // evaluate the complementarity condition
  IterationOutputs outputs{};
  if (dims_.M > 0) {
    outputs.mu = ConstSBlock(dims_, r_).sum() / static_cast<double>(dims_.M);
  }

  // solve the system, multiply by sigma to get equation 19.19
  SolveForUpdate(sigma * outputs.mu);

  // compute step size
  if (dims_.M > 0) {
    ComputeAlpha(&outputs);
  }

  // update
  const double alpha = std::min(outputs.alpha_dual, outputs.alpha_primal);
  variables_.noalias() += delta_ * alpha;

  // return mu and alpha
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
void QPInteriorPointSolver::SolveForUpdate(const double mu) {
  const std::size_t N = dims_.N;
  const std::size_t M = dims_.M;
  const std::size_t K = dims_.K;

  // const-block expressions for these, for convenience
  const auto x = ConstXBlock(dims_, variables_);
  const auto s = ConstSBlock(dims_, variables_);
  const auto y = ConstYBlock(dims_, variables_);
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

  // create the right-hand side (const because these were filled out already)
  const auto r_d = ConstXBlock(dims_, r_);
  const auto r_comp = ConstSBlock(dims_, r_);
  const auto r_pe = ConstYBlock(dims_, r_);
  const auto r_pi = ConstZBlock(dims_, r_);

  // apply the variable elimination, which updates r_d (make a copy to save the original)
  r_dual_aug_.noalias() = r_d;
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    r_dual_aug_[c.variable] += c.a * (z[i] / s[i]) * r_pi[i];
    r_dual_aug_[c.variable] += c.a * (r_comp[i] - mu) / s[i];
  }

  // factorize, TODO(gareth): preallocate ldlt.
  const LDLT<MatrixXd, Eigen::Lower> ldlt(H_);
  if (ldlt.info() != Eigen::ComputationInfo::Success) {
    std::stringstream ss;
    ss << "Failed to solve:\n" << H_.format(kMatrixFmt) << "\n";
    throw std::runtime_error(ss.str());
  }

  // compute H^-1
  H_inv_.setIdentity();
  ldlt.solveInPlace(H_inv_);

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
    dz[i] = -(z[i] / s[i]) * ds[i] - (1 / s[i]) * (r_comp[i] - mu);
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
void QPInteriorPointSolver::ComputeAlpha(IterationOutputs* const output) const {
  output->alpha_primal = ComputeAlpha(ConstSBlock(dims_, variables_), ConstSBlock(dims_, delta_));
  output->alpha_dual = ComputeAlpha(ConstZBlock(dims_, variables_), ConstZBlock(dims_, delta_));
}

double QPInteriorPointSolver::ComputeAlpha(
    const Eigen::VectorBlock<const Eigen::VectorXd>& val,
    const Eigen::VectorBlock<const Eigen::VectorXd>& d_val) const {
  ASSERT(val.rows() == d_val.rows());
  double alpha = 1.0;
  // TODO(gareth): Make this value adjustable for possibly faster convergence?
  constexpr double tau = .995;
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
    case TerminationState::BAD_STEP:
      stream << "BAD_STEP";
      break;
    case TerminationState::MAX_ITERATIONS:
      stream << "MAX_ITERATIONS";
      break;
  }
  return stream;
}

}  // namespace mini_opt
