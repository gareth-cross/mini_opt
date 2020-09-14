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
  ASSERT(p_.c.rows() == x_guess.rows());
  ASSERT(p_.A_eq.rows() == p_.b_eq.rows());

  // If equality constraints were specified, we must be able to multiply A*x
  if (p_.A_eq.size() > 0) {
    ASSERT(p_.A_eq.cols() == x_guess.rows());
  } else {
    ASSERT(p_.A_eq.rows() == 0);
    ASSERT(p_.b_eq.rows() == 0);
  }

  // Order is [slacks (s), equality multipliers(y), inequality multiplers (lambda)]
  const std::size_t x_size = x_guess.rows();
  const std::size_t num_slacks = p_.constraints.size();
  const std::size_t num_multipliers = p_.constraints.size() + p_.A_eq.rows();
  variables_.resize(x_size + num_slacks + num_multipliers);
  variables_.head(x_size) = x_guess;

  // TODO(gareth): A better initialization strategy for these?
  variables_.tail(num_slacks + num_multipliers).setConstant(1);

  // Allocate space for solving
  const std::size_t reduced_system_size = x_size + p_.A_eq.rows();
  H_.resize(reduced_system_size, reduced_system_size);
  H_.setZero();
  H_inv_.resizeLike(H_);

  // Use the total size here
  r_.resize(variables_.size());

  // Space for the output of all variables
  delta_.resizeLike(variables_);
}

void QPInteriorPointSolver::Iterate(const double sigma) {
  (void)sigma;
  // const std::size_t N = primal_variables_.size();
  // const std::size_t M = dual_variables_.size() / 2;

  //// get const blocks to access the dual variables
  // const auto y = dual_variables_.head(M);
  // const auto lambda = dual_variables_.tail(M);

  //// compute the residuals, these are shared
  // auto r_dual = r_.head(N);
  // auto r_primal = r_.segment(N, M);
  // auto r_complementary = r_.tail(M);

  //// contribution from the quadratic cost
  // r_dual.noalias() = p_.G * primal_variables_ + p_.c;

  //// contribution from inequality constraints
  // for (std::size_t i = 0; i < p_.constraints.size(); ++i) {
  //  const LinearInequalityConstraint& c = p_.constraints[i];
  //  // add to dual `r_d`:
  //  r_dual[c.variable] -= c.a * lambda[i];
  //  // add to primal `r_p`:
  //  r_primal[i] = c.a * primal_variables_[c.variable] + c.b - y[i];
  //}

  //// diag(y) * lambda = diag(lambda) * y
  //// we don't subtract the mu * sigma component yet
  // r_complementary = y.array() * lambda.array();

  //// initialize top left to the quadratic term
  // H_.topLeftCorner(N, N) = p_.G;

  // if (solve_method_ == SolveMethod::FULL_SYSTEM_PARTIAL_PIV_LU) {
  //  IterateFullPivLU(sigma);
  //} else if (solve_method_ == SolveMethod::ELIMINATE_DUAL_CHOLESKY) {
  //  IterateEliminateCholesky();
  //}
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
 * We first eliminate the second row using:
 *
 * p_s = \Sigma^-1 * (-diag(s)^-1 * r_comp - I * p_z)
 *
 * Which reduces the system to:
 *
 * [[G    A_e^T  A_i^T    ]                [r_d;
 *  [A_e  0      0        ]    * delta = -  r_pe;
 *  [A_i  0     -\Sigma^-1]]                r_pi + \Sigma^-1 * diag(s)^-1 * r_comp]
 *
 * Then we reduce it again using:
 *
 * p_z = \Sigma * (-A_i * -p_x - r_pi) - diag(s)^-1 * r_comp
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
 */
void QPInteriorPointSolver::IterateEliminateCholesky() {
  const std::size_t N = p_.G.rows();
  const std::size_t M = p_.constraints.size();
  const std::size_t K = p_.A_eq.rows();

  // const-block expressions for these, for convenience
  const auto x = variables_.head(N);
  const auto s = variables_.segment(N, M);
  const auto y = variables_.segment(N + M, K);
  const auto z = variables_.tail(M);

  // build the left-hand side (we only need lower triangular)
  H_.topLeftCorner(N, N).triangularView<Eigen::Lower>() = p_.G.triangularView<Eigen::Lower>();
  if (K > 0) {
    H_.bottomLeftCorner(K, N) = p_.A_eq;
  }
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    H_(c.variable, c.variable) += c.a * (z[i] / s[i]) * c.a;
  }

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
    r_comp[i] = s[i] * z[i];  //  don't subtract mu, yet
  }

  //std::cout << r_.transpose() << std::endl;

  // apply the variable elimination, which updates r_d
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    r_d[c.variable] += c.a * (z[i] / s[i]) * r_pi[i];
    r_d[c.variable] += c.a * r_comp[i] / s[i];
  }

  //std::cout << "\n" << H_.format(kMatrixFmt) << std::endl;

  // factorize
  LDLT<MatrixXd, Eigen::Lower> ldlt(H_);
  if (ldlt.info() != Eigen::ComputationInfo::Success) {
    throw std::runtime_error("Failed to solve");  // todo more detail
  }

  // compute H^-1
  H_inv_.setIdentity();
  ldlt.solveInPlace(H_inv_);

  // compute [px, -py]
  auto dx = delta_.head(N);
  auto ds = delta_.segment(N, M);
  auto dy = delta_.segment(N + M, K);
  auto dz = delta_.tail(M);

  dx.noalias() = H_inv_.block(0, 0, N, N) * -r_d;
  if (K > 0) {
    dx.noalias() += H_inv_.block(0, N, N, K) * -r_pe;
    // negate here since py is negative in the solution vector (this flips the sign back to normal)
    dy.noalias() = H_inv_.block(N, 0, K, N) * r_d;
    dy.noalias() += H_inv_.block(N, N, K, K) * r_pe;
  }

  // Go back and solve for dz and ds
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = p_.constraints[i];
    ds[i] = c.a * dx[c.variable] + r_pi[i];
    dz[i] = -(z[i] / s[i]) * (ds[i]) - (1 / s[i]) * r_comp[i];
  }
}

/*
 * Just build this directly
 *
 * [[G       0        A_e^T     A_i^T]                 [r_d;
 *  [0       \Sigma   0        -I    ]     * delta = -  diag(s)^-1 * r_comp;
 *  [A_e     0        0         0    ]                  r_pe;
 *  [A_i    -I        0         0    ]]                 r_pi]
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
