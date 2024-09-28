// Copyright 2021 Gareth Cross
#pragma once
#include <Eigen/Core>
#include <array>
#include <memory>
#include <vector>

#include "mini_opt/assertions.hpp"

// Provide mechanisms of specifying non-linear residuals.
namespace mini_opt {

// Base type for residuals in case we want more than one.
struct ResidualBase {
  using unique_ptr = std::unique_ptr<ResidualBase>;

  // We will be storing these through pointer to the base class.
  virtual ~ResidualBase();

  // Dimension of the residual vector.
  virtual int Dimension() const = 0;

  // Get the error vector: h(x)
  virtual void ErrorVector(const Eigen::VectorXd& params,
                           Eigen::VectorBlock<Eigen::VectorXd> b_out) const = 0;

  // Update a system of equations Hx=b by writing to `H` and `b`.
  // Returns the value of `Error` as well (the constant part of the quadratic).
  virtual double UpdateHessian(const Eigen::VectorXd& params, Eigen::MatrixXd* H,
                               Eigen::VectorXd* b) const = 0;

  // Output the jacobian for the linear system: J * dx + b
  // `J_out` and `b_out` are set to the correct rows of a larger matrix.
  virtual void UpdateJacobian(const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
                              Eigen::VectorBlock<Eigen::VectorXd> b_out) const = 0;

  // Helper for tests for computing L2 error.
  double QuadraticError(const Eigen::VectorXd& params) const;
};

// Helper for declaring either vector or array, depending on whether size is known at compile time.
namespace internal {
template <int N>
struct IndexType {
  using type = std::array<int, static_cast<std::size_t>(N)>;
};
template <>
struct IndexType<Eigen::Dynamic> {
  using type = std::vector<int>;
};
}  // namespace internal

/*
 * Simple statically sized residual. The parameters may be dynamic.
 */
template <int ResidualDim, int NumParams>
struct Residual : public ResidualBase {
  using ParamType = Eigen::Matrix<double, NumParams, 1>;
  using ResidualType = Eigen::Matrix<double, ResidualDim, 1>;
  using JacobianType = Eigen::Matrix<double, ResidualDim, NumParams>;
  using IndexType = typename internal::IndexType<NumParams>::type;
  using FunctionType =
      std::function<ResidualType(const ParamType& params, JacobianType* const J_out)>;

  // Variables we are touching, one per column in the jacobian.
  IndexType index;

  // Function that evaluates the residual given the params, and returns an error vector and
  // optionally the jacobian via the output argument.
  FunctionType function;

  Residual() = default;

  // Construct from members by copy.
  Residual(const IndexType& index, const FunctionType& func) : index(index), function(func) {}

  // Return constant dimension.
  int Dimension() const override { return ResidualDim; }

  // Map params from the global state vector to those required for this function, and
  // then evaluate the function.
  void ErrorVector(const Eigen::VectorXd& params,
                   Eigen::VectorBlock<Eigen::VectorXd> b_out) const override;

  // Map params from the global state vector to those required for this function, and
  // then evaluate the function and its derivative. Update the linear system [H|b] w/
  // the result.
  double UpdateHessian(const Eigen::VectorXd& params, Eigen::MatrixXd* H,
                       Eigen::VectorXd* b) const override;

  // Implementation of abstract method UpdateJacobian.
  void UpdateJacobian(const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
                      Eigen::VectorBlock<Eigen::VectorXd> b_out) const override;

 private:
  // Copy out the params that matter for this function.
  ParamType GetParamSlice(const Eigen::VectorXd& params) const;
};

//
// Template implementations.
//

// Turn off warnings about constant if statements.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif  // _MSC_VER

// Helper to read a sparse set of values out of a larger matrix.
template <int N>
void ReadSparseValues(const Eigen::VectorXd& input,
                      const typename internal::IndexType<N>::type& index,
                      Eigen::Matrix<double, N, 1>* output) {
  F_ASSERT(output != nullptr);
  if (N == Eigen::Dynamic) {
    output->resize(index.size());
  }
  for (std::size_t local = 0; local < index.size(); ++local) {
    const int i = index[local];
    F_ASSERT_GE(i, 0);
    F_ASSERT_LT(i, input.rows(), "Index exceeds the number of params");
    output->operator[](local) = input[i];
  }
}

template <int ResidualDim, int NumParams>
typename Residual<ResidualDim, NumParams>::ParamType
Residual<ResidualDim, NumParams>::GetParamSlice(const Eigen::VectorXd& params) const {
  // TODO(gareth): Create cached storage for this.
  ParamType sliced;
  ReadSparseValues<NumParams>(params, index, &sliced);
  return sliced;
}

template <int ResidualDim, int NumParams>
void Residual<ResidualDim, NumParams>::ErrorVector(
    const Eigen::VectorXd& params, Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
  F_ASSERT_EQ(b_out.rows(), Dimension(), "Output vector is wrong dimension");
  const ParamType relevant_params = GetParamSlice(params);
  b_out = function(relevant_params, nullptr);
}

// TODO(gareth): Probably faster to associate a dimension to each variable,
// in the style of GTSAM, so that we can do block-wise updates. For now this
// suits the small problem size I am doing.
template <int ResidualDim, int NumParams>
double Residual<ResidualDim, NumParams>::UpdateHessian(const Eigen::VectorXd& params,
                                                       Eigen::MatrixXd* const H,
                                                       Eigen::VectorXd* const b) const {
  F_ASSERT(H != nullptr);
  F_ASSERT(b != nullptr);
  F_ASSERT_EQ(H->rows(), H->cols());
  F_ASSERT_EQ(b->rows(), H->rows());

  // Collect params.
  const ParamType relevant_params = GetParamSlice(params);

  // Evaluate the function and its derivative.
  JacobianType J;
  if (NumParams == Eigen::Dynamic) {
    J.resize(ResidualDim, index.size());
  }
  const ResidualType r = function(relevant_params, &J);

  // Add contributions to the hessian, only lower part.
  const int N = static_cast<int>(index.size());
  for (int row_local = 0; row_local < N; ++row_local) {
    // get index mapping into the full system
    const int row_global = index[row_local];
    F_ASSERT_LT(row_global, H->rows(), "Index exceeds bounds of hessian");
    for (int col_local = 0; col_local <= row_local; ++col_local) {
      // because col_local <= row_local, we already checked this global index
      const int col_global = index[col_local];
      // each param is a single column, so we can just do dot product
      const double JtT = J.col(row_local).dot(J.col(col_local));
      // swap so we only update the lower triangular part
      if (col_global <= row_global) {
        H->operator()(row_global, col_global) += JtT;
      } else {
        H->operator()(col_global, row_global) += JtT;
      }
    }
    // Also update the right-hand side vector `b`.
    b->operator()(row_global) += J.col(row_local).dot(r);
  }
  return 0.5 * r.squaredNorm();
}

// This version takes blocks, so we can write directly into A_eq.
template <int ResidualDim, int NumParams>
void Residual<ResidualDim, NumParams>::UpdateJacobian(
    const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
    Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
  F_ASSERT_EQ(ResidualDim, b_out.rows());
  F_ASSERT_EQ(ResidualDim, J_out.rows());
  // Collect params.
  const ParamType relevant_params = GetParamSlice(params);

  // Evaluate, and copy jacobian back using indices.
  JacobianType J;
  if (NumParams == Eigen::Dynamic) {
    J.resize(ResidualDim, index.size());
  }
  b_out.noalias() = function(relevant_params, &J);

  for (int col_local = 0; col_local < static_cast<int>(index.size()); ++col_local) {
    const int col_global = index[col_local];
    F_ASSERT_LT(col_global, J_out.cols(), "Index exceeds the size of the Jacobian");
    J_out.col(col_global).noalias() = J.col(col_local);
  }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

}  // namespace mini_opt
