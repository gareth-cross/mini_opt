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
    : problem_(problem) {
  ASSERT(problem_.G.rows() == problem_.G.cols(), "G must be square, got dimensions [%i, %i]",
         problem_.G.rows(), problem_.G.cols());
  ASSERT(problem_.G.rows() == problem_.c.rows(), "Dimension of c (%i) does not match G (%i)",
         problem_.c.rows(), problem_.G.rows());
  ASSERT(problem_.c.rows() == x_guess.rows(), "Dimension of c (%i) does not match x_guess (%i)",
         problem_.c.rows(), x_guess.rows());

  primal_variables_.resize(problem_.G.rows());
  primal_variables_ = x_guess;
  dual_variables_.resize(problem_.constraints.size() * 2);
  dual_variables_.setConstant(1);
}

void QPInteriorPointSolver::Iterate(const double sigma) {
  // build the linear system
  // todo: leverage sparsity
  const std::size_t N = primal_variables_.size();
  const std::size_t M = dual_variables_.size() / 2;
  const std::size_t num_rows = M * 2 + N;

  MatrixXd H(num_rows, num_rows);
  VectorXd r(num_rows);

  H.setZero();
  r.setZero();

  // add contributions from the quadratic cost
  H.topLeftCorner(N, N) = problem_.G;
  r.head(N) = problem_.G * primal_variables_ + problem_.c;

  // compute complemetary measure
  const double mu = dual_variables_.head(M).dot(dual_variables_.tail(M)) / M;

  // add contributions from the inequality constrainst
  for (std::size_t i = 0; i < M; ++i) {
    const LinearInequalityConstraint& c = problem_.constraints[i];

    H(c.variable, N + M + i) = -c.a;
    H(N + i, c.variable) = c.a;
    H(N + i, N + i) = -1.0;
    H(N + M + i, N + i) = dual_variables_[M + i];
    H(N + M + i, N + M + i) = dual_variables_[i];

    // add to dual `r_d`:
    r(c.variable) -= c.a * dual_variables_[M + i];
    // add to primal `r_p`:
    r(N + i) = c.a * primal_variables_[c.variable] - c.b - dual_variables_[i];
    // complementary measure:
    r(N + M + i) = dual_variables_[i] * dual_variables_[M + i] - sigma * mu;
  }

  std::cout << "h = \n" << H.format(kMatrixFmt) << std::endl;
  std::cout << "r = \n" << r.format(kMatrixFmt) << std::endl;

  // solve the system
  PartialPivLU<Eigen::MatrixXd> piv_lu(H);
  std::cout << "determinant = " << piv_lu.determinant() << std::endl;

  const Eigen::VectorXd update = piv_lu.solve(-r);
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

  primal_variables_.head(N) += update.head(N) * alpha;
  dual_variables_ += update.tail(M * 2) * alpha;
  std::cout << "x = \n" << primal_variables_.transpose().format(kMatrixFmt) << std::endl;
  std::cout << "y, lambda = \n" << dual_variables_.transpose().format(kMatrixFmt) << std::endl;
}

}  // namespace mini_opt
