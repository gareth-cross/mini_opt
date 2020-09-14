// Copyright 2020 Gareth Cross
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Vector>
#include <array>
#include <memory>

#include "assertions.hpp"

/*
 * The reference for this implementation is:
 *
 *   "Numerical Optimization, Second Edition", Jorge Nocedal and Stephen J. Wright
 *
 * Any equation numbers you see refer to this book, unless otherwise stated.
 *
 * TODO(gareth): Template everything for double or float? For now double suits me.
 */
namespace mini_opt {

// Base type for residuals in case we want more than one.
struct ResidualBase {
  using unique_ptr = std::unique_ptr<ResidualBase>;

  // We will be storing these through pointer to the base class.
  virtual ~ResidualBase();

  // Get the error.
  virtual double Error(const Eigen::VectorXd& params) const = 0;

  // Update a system of equations Hx=b by writing to `H` and `b`.
  virtual void UpdateSystem(const Eigen::VectorXd& params, Eigen::MatrixXd* const H,
                            Eigen::VectorXd* const b) const = 0;
};

// Simple statically sized residual.
template <size_t ResidualDim, size_t NumParams>
struct Residual : public ResidualBase {
  using ParamType = Eigen::Matrix<double, NumParams, 1>;
  using ResidualType = Eigen::Matrix<double, ResidualDim, 1>;
  using JacobianType = Eigen::Matrix<double, ResidualDim, NumParams>;

  // Variables we are touching, one per column in the jacobian.
  std::array<int, NumParams> index;

  // Function that evaluates the residual given the params, and returns an error vector and
  // optionally the jacobian via the output argument.
  std::function<ResidualType(const ParamType& params, JacobianType* const J_out)> function;

  // Map params from the global state vector to those required for this function, and
  // then evaluate the function.
  double Error(const Eigen::VectorXd& params) const override;

  // Map params from the global state vector to those required for this function, and
  // then evaluate the function and its derivative. Update the linear system [H|b] w/
  // the result.
  void UpdateSystem(const Eigen::VectorXd& params, Eigen::MatrixXd* const H,
                    Eigen::VectorXd* const b) const override;

 private:
  // Copy out the params that matter for this function.
  ParamType GetParamSlice(const Eigen::VectorXd& params) const;
};

/*
 * Describes a linear (technically affine) inequality constraint.
 * The constraint is specified in the form:
 *
 *    a * x[variable] + b >= 0
 *
 * TODO(gareth): Generalize beyond diagonal A-matrix?
 */
struct LinearInequalityConstraint {
  // Index of the variable this refers to.
  int variable;
  // Constraint coefficients.
  double a;
  double b;

  // True if x is feasible.
  bool IsFeasible(double x) const;

  // Ctor
  LinearInequalityConstraint(int variable, double a, double b) : variable(variable), a(a), b(b) {}

  LinearInequalityConstraint() = default;
};

/*
 * Problem specification for a QP:
 *
 *  minimize x^T * G * x + c^T * c
 *
 *  st. A_e * x + b_e == 0
 *  st. a_i * x + b)_i >= 0
 */
struct QP {
  Eigen::MatrixXd G;
  Eigen::VectorXd c;

  // Optional equality constraints in form Ax + b = 0
  Eigen::MatrixXd A_eq;
  Eigen::VectorXd b_eq;

  // Diagonal inequality constraints.
  std::vector<LinearInequalityConstraint> constraints;
};

/*
 * Minimize quadratic cost function with inequality constraints using interior point method.
 */
struct QPInteriorPointSolver {
  // Note we don't copy the problem, it must remain in scope for the duration of the solver.
  QPInteriorPointSolver(const QP& problem, const Eigen::VectorXd& x_guess);

  void Iterate();

  // Print state to a string, for unit tests.
  std::string StateToString() const;

 private:
  const QP& p_;

  // Storage for the variables: (x, s, y, z)
  Eigen::VectorXd variables_;

  // Re-usable storage for the linear system and residuals
  Eigen::MatrixXd H_;
  Eigen::VectorXd r_;
  Eigen::MatrixXd H_inv_;

  // Solution vector at each iteration
  Eigen::VectorXd delta_;

  // Solve the augmented linear system, which is done by eliminating p_s, and p_z and then
  // solving for p_x and p_y.
  void SolveForUpdate(const double mu);

  double ComputeAlpha() const;

  // Compute the `alpha` step size.
  // Returns alpha such that (val[i] + d_val[i]) >= val[i] * (1 - tau)
  double ComputeAlpha(const Eigen::VectorBlock<const Eigen::VectorXd>& val,
                      const Eigen::VectorBlock<const Eigen::VectorXd>& d_val) const;

  // For unit test, allow construction of the full linear system required for Newton step.
  void BuildFullSystem(Eigen::MatrixXd* const H, Eigen::VectorXd* const r) const;

  friend class QPSolverTest;
};

/*
 * Describes a simple [non-]linear least squares problem. The primary cost is a sum-of
 * squares.
 *
 * Supports simple linear inequality constraints on the variables.
 *
 * More formally:
 *
 *  min: f_0(x)  [where f_0(x) = (1/2) * h(x)^T * h(x)]
 *
 *  Subject to: diag(a) * x >= b
 *
 * Note that we actually iteratively minimize the first order approximation of f(x):
 *
 *  h(x + dx) = h(x) + J * dx
 *
 * Such that: f_0(x) = (1/2) * h(x)^T * h(x) + (J * dx)^T * h(x) + (J * dx)^T * (J * dx)
 *
 * So in effect, we are solving a quadratic approximation of the nonlinear cost
 * with diagonal inequality constraints on the decision variables.
 */
struct Problem {
  using unique_ptr = std::unique_ptr<Problem>;

  // The errors that form the sum of squares part of the cost function.
  std::vector<ResidualBase::unique_ptr> costs;

  // Linear inequality constraints.
  std::vector<LinearInequalityConstraint> inequality_constraints;
};

//
// Template implementations.
// TODO(gareth): Could put these in a separate header.
//

template <size_t ResidualDim, size_t NumParams>
typename Residual<ResidualDim, NumParams>::ParamType
Residual<ResidualDim, NumParams>::GetParamSlice(const Eigen::VectorXd& params) const {
  ParamType sliced;
  static_assert(ParamType::RowsAtCompileTime == NumParams, "");
  for (std::size_t local = 0; local < NumParams; ++local) {
    const int i = index[local];
    ASSERT(i >= 0);
    ASSERT(i < params.rows(), "Index %i exceeds the # of provided params, which is %i", i,
           params.rows());
    sliced[local] = params[i];
  }
  return sliced;
}

template <size_t ResidualDim, size_t NumParams>
double Residual<ResidualDim, NumParams>::Error(const Eigen::VectorXd& params) const {
  const ParamType relevant_params = GetParamSlice(params);
  const ResidualType err = function(relevant_params, nullptr);
  return err.squaredNorm();
}

// TODO(gareth): Probably faster to associate a dimension to each variable,
// in the style of GTSAM, so that we can do block-wise updates. For now this
// suits the small problem size I am doing.
template <size_t ResidualDim, size_t NumParams>
void Residual<ResidualDim, NumParams>::UpdateSystem(const Eigen::VectorXd& params,
                                                    Eigen::MatrixXd* const H,
                                                    Eigen::VectorXd* const b) const {
  ASSERT(H != nullptr);
  ASSERT(b != nullptr);
  ASSERT(H->rows() == H->cols());
  ASSERT(b->rows() == H->rows());

  // Collect params.
  const ParamType relevant_params = GetParamSlice(params);

  // Evaluate the function and its derivative.
  JacobianType J;
  const ResidualType r = function(relevant_params, &J);

  // Add contributions to the hessian, only lower part.
  constexpr int N = static_cast<int>(NumParams);
  for (int row_local = 0; row_local < N; ++row_local) {
    // get index mapping into the full system
    const int row_global = index[row_local];
    ASSERT(row_global < H->rows(), "Index %i exceeds the bounds of the hessian (rows = %i)",
           row_global, H->rows());
    for (int col_local = 0; col_local <= row_local; ++col_local) {
      // pull and check the column index as well
      const int col_global = index[col_local];
      ASSERT(col_global < H->rows(), "Index %i exceeds the bounds of the hessian (rows = %i)",
             col_global, H->rows());

      // each param is a single column, so we can just do dot product
      const double JtT = J.col(row_local).dot(J.col(col_local));
      // swap so we only update the lower triangular part
      if (col_global <= row_global) {
        H->operator()(row_global, col_global) += JtT;
      } else {
        H->operator()(col_global, row_global) += JtT;
      }
    }
    // Also update the right hand side vector `b`.
    b->operator()(row_global) += J.col(row_local).dot(r);
  }
}

}  // namespace mini_opt
