// Copyright 2020 Gareth Cross
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Vector>
#include <array>
#include <memory>

#include "assertions.hpp"

// TODO(gareth): Template these for double or float? For now double suits me.
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
