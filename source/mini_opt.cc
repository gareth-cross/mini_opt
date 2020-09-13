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

QPInteriorPointSolver::QPInteriorPointSolver(const QP& problem, const Eigen::VectorXd& x_guess,
                                             const SolveMethod& solve_method)
    : problem_(problem), solve_method_(solve_method) {
  ASSERT(problem_.G.rows() == problem_.G.cols(), "G must be square, got dimensions [%i, %i]",
         problem_.G.rows(), problem_.G.cols());
  ASSERT(problem_.G.rows() == problem_.c.rows(), "Dimension of c (%i) does not match G (%i)",
         problem_.c.rows(), problem_.G.rows());
  ASSERT(problem_.c.rows() == x_guess.rows(), "Dimension of c (%i) does not match x_guess (%i)",
         problem_.c.rows(), x_guess.rows());

  primal_variables_.resize(problem_.G.rows());
  primal_variables_ = x_guess;

  // Order is [slacks (y), langrange multiplers (lambda)]
  // TODO(gareth): A better initialization strategy for these?
  dual_variables_.resize(problem_.constraints.size() * 2);
  dual_variables_.setConstant(1);

  // Allocate space for solving
  const std::size_t total_size = primal_variables_.rows() + dual_variables_.rows();
  H_.resize(total_size, total_size);
  H_.setZero();
  r_.resize(total_size);
}

void QPInteriorPointSolver::Iterate(const double sigma) {
  const std::size_t N = primal_variables_.size();
  const std::size_t M = dual_variables_.size() / 2;

  // get const blocks to access the dual variables
  const auto y = dual_variables_.head(M);
  const auto lambda = dual_variables_.tail(M);

  // compute the residuals, these are shared
  auto r_dual = r_.head(N);
  auto r_primal = r_.segment(N, M);
  auto r_complementary = r_.tail(M);

  // contribution from the quadratic cost
  r_dual.noalias() = problem_.G * primal_variables_ + problem_.c;

  // contribution from inequality constraints
  for (std::size_t i = 0; i < problem_.constraints.size(); ++i) {
    const LinearInequalityConstraint& c = problem_.constraints[i];
    // add to dual `r_d`:
    r_dual[c.variable] -= c.a * lambda[i];
    // add to primal `r_p`:
    r_primal[i] = c.a * primal_variables_[c.variable] + c.b - y[i];
  }

  // diag(y) * lambda = diag(lambda) * y
  // we don't subtract the mu * sigma component yet
  r_complementary = y.array() * lambda.array();

  // initialize top left to the quadratic term
  H_.topLeftCorner(N, N) = problem_.G;

  if (solve_method_ == SolveMethod::FULL_SYSTEM_PARTIAL_PIV_LU) {
    IterateFullPivLU(sigma);
  } else if (solve_method_ == SolveMethod::ELIMINATE_DUAL_CHOLESKY) {
    IterateEliminateDualCholesky();
  }
}

/*
 * We use the full H matrix, and fill it in as follows:
 *
 *  [[G   0               -A^T]
 *   [A  -I                  0]   (Equation 16.67)
 *   [0   diag(lambda) diag(y)]
 *
 * Top left is filled in by the caller already. I am leaving this implementation for
 * unit testing as a comparison for the faster cholesky method.
 */
void QPInteriorPointSolver::IterateFullPivLU(const double sigma) {
  ASSERT(H_.rows() > 0, "Must have already been initialized");
  const std::size_t N = primal_variables_.size();
  const std::size_t M = problem_.constraints.size();

  // shorthand accessors
  const auto y = dual_variables_.head(M);
  const auto lambda = dual_variables_.tail(M);

  // Add contributions from the inequality constraints.
  // This H is quite sparse, because we have only diagonal A matrices in my implementation.
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = problem_.constraints[i];
    H_(c.variable, N + M + i) = -c.a;  // -A^T
    H_(N + i, c.variable) = c.a;       // A
    H_(N + i, N + i) = -1.0;           // -Identity
    H_(N + M + i, N + i) = lambda[i];  // diag(lambda)
    H_(N + M + i, N + M + i) = y[i];   // diag(y)
  }

  // factorize
  const PartialPivLU<Eigen::MatrixXd> piv_lu(H_);
  constexpr double kDetTol = 1.0e-9;
  if (std::abs(piv_lu.determinant()) < kDetTol) {
    // TODO(gareth): Make this tolerance a parameter.
    throw std::runtime_error("Could not invert matrix to compute newton step");
  }

  // compute the complementarity measure
  const double mu = y.dot(lambda) / static_cast<double>(M);

  // TODO(gareth): get rid of this
  auto r_comp = r_.tail(M);
  r_comp.array() -= mu * sigma;

  std::cout << "determinant = " << piv_lu.determinant() << std::endl;

  const Eigen::VectorXd update = piv_lu.solve(-r_);
  std::cout << "update no scale = " << update.transpose() << std::endl;

  // line search for alpha
  double alpha = 1.0;
  for (std::size_t i = 0; i < 2 * M; ++i) {
    const double updated_dv = dual_variables_[i] + update[N + i];
    if (updated_dv <= 0 && std::abs(update[N + i]) > 0) {
      // clamp so that lambda, s >= 0
      const double candidate_alpha = -0.995 * dual_variables_[i] / update[i + N];
      if (candidate_alpha < alpha) {
        alpha = candidate_alpha;
      }
    }
  }

  std::cout << "alpha = " << alpha << std::endl;
  std::cout << "update = \n" << (update * alpha).transpose().format(kMatrixFmt) << std::endl;

  primal_variables_ += update.head(N) * alpha;
  dual_variables_ += update.tail(M * 2) * alpha;

  std::cout << "x = \n" << primal_variables_.transpose().format(kMatrixFmt) << std::endl;
  std::cout << "y, lambda = \n" << dual_variables_.transpose().format(kMatrixFmt) << std::endl;
}

void QPInteriorPointSolver::IterateEliminateDualCholesky() {}

}  // namespace mini_opt
