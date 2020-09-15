#include "mini_opt.hpp"

// temp
#include <iostream>

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
  const std::size_t x_size = p_.G.rows();
  const std::size_t num_slacks = p_.constraints.size();
  const std::size_t num_multipliers = p_.constraints.size() + p_.A_eq.rows();
  variables_.resize(x_size + num_slacks + num_multipliers);

  // if a guess was provided, copy it in
  if (x_guess.size() > 0) {
    ASSERT(x_guess.rows() == p_.G.rows());
    variables_.head(x_size) = x_guess;
  } else {
    // otherwise guess zero
    variables_.head(x_size).setZero();
  }

  // TODO(gareth): A better initialization strategy for these?
  // Could assume constraints are active, in which case we compute lambda from the KKT conditions.
  variables_.tail(num_slacks + num_multipliers).setConstant(1);

  // Allocate space for solving
  const std::size_t reduced_system_size = x_size + p_.A_eq.rows();
  H_.resize(reduced_system_size, reduced_system_size);
  H_.setZero();
  H_inv_.resizeLike(H_);

  // Use the total size here
  r_.resize(variables_.size());
  r_dual_aug_.resize(x_size);

  // Space for the output of all variables
  delta_.resizeLike(variables_);
}

void QPInteriorPointSolver::Iterate(const double sigma) {
  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();

  // fill out `r_`
  EvaluateKKTConditions();

  // evaluate the complementarity condition
  double mu = 0;
  if (M > 0) {
    const auto r_comp = r_.segment(N, M);
    mu = r_comp.sum() / static_cast<double>(M);
  }

  // solve the system, multiply by sigma to get equation 19.19
  SolveForUpdate(sigma * mu);

  // compute step size
  const double alpha = ComputeAlpha();

  if (logger_callback_) {
    // pass progress to the logger callback, TODO(gareth): More relevant things here?
    logger_callback_(r_.squaredNorm(), mu, alpha);
  }

  // update
  variables_.noalias() += delta_ * alpha;
}

// TODO(gareth): All the accessing by index is a little gross, could maybe split
// up the variables into separate vectors? I do like having the full state as one object.
QPInteriorPointSolver::ConstVectorBlock QPInteriorPointSolver::x_block() const {
  const std::size_t N = p_.G.rows();
  return variables_.head(N);
}

QPInteriorPointSolver::ConstVectorBlock QPInteriorPointSolver::s_block() const {
  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();
  return variables_.segment(N, M);
}

QPInteriorPointSolver::ConstVectorBlock QPInteriorPointSolver::y_block() const {
  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();
  const std::size_t K = p_.A_eq.rows();
  return variables_.segment(N + M, K);
}

QPInteriorPointSolver::ConstVectorBlock QPInteriorPointSolver::z_block() const {
  const std::size_t M = p_.constraints.size();
  return variables_.tail(M);
}

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
  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();
  const std::size_t K = p_.A_eq.rows();

  // const-block expressions for these, for convenience
  const auto x = variables_.head(N);
  const auto s = variables_.segment(N, M);
  const auto y = variables_.segment(N + M, K);
  const auto z = variables_.tail(M);

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
  const auto r_d = r_.head(N);
  const auto r_comp = r_.segment(N, M);
  const auto r_pe = r_.segment(N + M, K);
  const auto r_pi = r_.tail(M);

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
  auto dx = delta_.head(N);
  auto ds = delta_.segment(N, M);
  auto dy = delta_.segment(N + M, K);
  auto dz = delta_.tail(M);

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
  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();
  const std::size_t K = p_.A_eq.rows();

  // TODO(gareth): Some shorthand for getting these blocks?
  const auto x = variables_.head(N);
  const auto s = variables_.segment(N, M);
  const auto y = variables_.segment(N + M, K);
  const auto z = variables_.tail(M);

  // create the right-hand side
  auto r_d = r_.head(N);
  auto r_comp = r_.segment(N, M);
  auto r_pe = r_.segment(N + M, K);
  auto r_pi = r_.tail(M);

  // build the dual objective first
  r_d.noalias() = p_.G.selfadjointView<Eigen::Lower>() * x + p_.c;
  if (K > 0) {
    r_d.noalias() -= p_.A_eq.transpose() * y;
    // equality constraints
    r_pe.noalias() = p_.A_eq * x + p_.b_eq;
  }

  // contributions from inequality constraints, there is some redundant work here
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    r_d[c.variable] -= c.a * z[i];
    r_pi[i] = c.a * x[c.variable] + c.b - s[i];
    r_comp[i] = s[i] * z[i];
  }
}

// TODO(gareth): Should implement selection of alphas to minimize objective.
// See formula 16.66 in Nocedal and Wright
double QPInteriorPointSolver::ComputeAlpha() const {
  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();
  if (M > 0) {
    const double alpha_s = ComputeAlpha(variables_.segment(N, M), delta_.segment(N, M));
    const double alpha_z = ComputeAlpha(variables_.tail(M), delta_.tail(M));
    // We just take the min, although having separate alphas can supposedly improve convergence.
    return std::min(alpha_s, alpha_z);
  }
  return 1.0;
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

  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();
  const std::size_t K = p_.A_eq.rows();

  const auto x = variables_.head(N);
  const auto s = variables_.segment(N, M);
  const auto y = variables_.segment(N + M, K);
  const auto z = variables_.tail(M);

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

  auto r_d = r->head(N);
  auto s_inv_r_comp = r->segment(N, M);
  auto r_pe = r->segment(N + M, K);
  auto r_pi = r->tail(M);

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

}  // namespace mini_opt
