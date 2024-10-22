// Copyright 2021 Gareth Cross
#pragma once
#include <array>
#include <memory>
#include <vector>

#include <fmt/ostream.h>
#include <Eigen/Core>

#include "mini_opt/assertions.hpp"

// Provide mechanisms of specifying non-linear residuals.
namespace mini_opt {

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

// Type for residual in the optimization. Stores a type-erased callable provided by the user.
class Residual final {
 public:
  // Dimension of the residual vector.
  int Dimension() const {
    F_ASSERT(impl_);
    return impl_->Dimension();
  }

  // Get the error vector: h(x)
  virtual void ErrorVector(const Eigen::VectorXd& params,
                           Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
    F_ASSERT(impl_);
    return impl_->ErrorVector(params, b_out);
  }

  // Update a system of equations Hx=b by writing to `H` and `b`.
  // Returns the value of `Error` as well (the constant part of the quadratic).
  virtual double UpdateHessian(const Eigen::VectorXd& params, Eigen::MatrixXd* H,
                               Eigen::VectorXd* b) const {
    F_ASSERT(impl_);
    return impl_->UpdateHessian(params, H, b);
  }

  // Output the jacobian for the linear system: J * dx + b
  // `J_out` and `b_out` are set to the correct rows of a larger matrix.
  virtual void UpdateJacobian(const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
                              Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
    F_ASSERT(impl_);
    return impl_->UpdateJacobian(params, J_out, b_out);
  }

  // Helper for tests for computing L2 error.
  double QuadraticError(const Eigen::VectorXd& params) const;

 private:
  // Specify the abstract interface of a residual.
  class Concept {
   public:
    virtual ~Concept() = default;
    virtual int Dimension() const noexcept = 0;
    virtual void ErrorVector(const Eigen::VectorXd& params,
                             Eigen::VectorBlock<Eigen::VectorXd> b_out) const = 0;
    virtual double UpdateHessian(const Eigen::VectorXd& params, Eigen::MatrixXd* H,
                                 Eigen::VectorXd* b) const = 0;
    virtual void UpdateJacobian(const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
                                Eigen::VectorBlock<Eigen::VectorXd> b_out) const = 0;
  };

  // A concrete implementation of a residual.
  template <int ResidualDim, int ParamDim, typename F>
  class Model final : public Concept {
   public:
    using ParamType = Eigen::Matrix<double, ParamDim, 1>;
    using ResidualType = Eigen::Matrix<double, ResidualDim, 1>;
    using JacobianType = Eigen::Matrix<double, ResidualDim, ParamDim>;
    using IndexType = typename internal::IndexType<ParamDim>::type;

    template <typename I, typename U>
    Model(I&& index, U&& func) : index_(std::forward<I>(index)), func_(std::forward<U>(func)) {}

    int Dimension() const noexcept override { return ResidualDim; }

    void ErrorVector(const Eigen::VectorXd& params,
                     Eigen::VectorBlock<Eigen::VectorXd> b_out) const override;

    // TODO: This could accept blocks as well.
    double UpdateHessian(const Eigen::VectorXd& params, Eigen::MatrixXd* H,
                         Eigen::VectorXd* b) const override;

    void UpdateJacobian(const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
                        Eigen::VectorBlock<Eigen::VectorXd> b_out) const override;

   private:
    // Copy out the params that matter for this function.
    ParamType GatherParams(const Eigen::VectorXd& params) const;

    IndexType index_;
    F func_;
  };

  explicit Residual(std::unique_ptr<Concept> impl) noexcept : impl_(std::move(impl)) {}

  template <int ResidualDim, int ParamDim, typename Index, typename F>
  friend Residual MakeResidual(Index&&, F&&);
  template <int R, int P, typename F>
  friend Residual MakeResidual(std::initializer_list<int> index, F&& func);

  // TODO: Do small buffer optimization here for residuals <= 32 bytes.
  std::unique_ptr<Concept> impl_;
};

// Construct a residual that ingests `P` params, and produces an `R` dimensional error vector.
// TODO: We can reduce R & P by inspection of `F`.
template <int R, int P, typename Index, typename F>
Residual MakeResidual(Index&& index, F&& func) {
  using FuncType = std::remove_const_t<std::remove_reference_t<F>>;
  return Residual(std::make_unique<Residual::Model<R, P, FuncType>>(std::forward<Index>(index),
                                                                    std::forward<F>(func)));
}

// Variant of `MakeResidual` that accepts an initializer list of integers.
template <int R, int P, typename F>
Residual MakeResidual(std::initializer_list<int> index, F&& func) {
  using FuncType = std::remove_const_t<std::remove_reference_t<F>>;
  if constexpr (P == Eigen::Dynamic) {
    return Residual(
        std::make_unique<Residual::Model<R, P, FuncType>>(index, std::forward<F>(func)));
  } else {
    F_ASSERT_EQ(index.size(), static_cast<std::size_t>(P));
    // The index is std::array, which cannot be constructed from initializer list or iterators.
    std::array<int, P> index_copied{};
    std::copy_n(index.begin(), index.size(), index_copied.begin());
    return Residual(
        std::make_unique<Residual::Model<R, P, FuncType>>(index_copied, std::forward<F>(func)));
  }
}

//
// Template implementations.
//

// Helper to read a sparse set of values out of a larger matrix.
template <int N>
void GatherValues(const Eigen::VectorXd& input, const typename internal::IndexType<N>::type& index,
                  Eigen::Matrix<double, N, 1>* output) {
  F_ASSERT(output != nullptr);
  if constexpr (N == Eigen::Dynamic) {
    output->resize(index.size());
  }
  for (std::size_t local = 0; local < index.size(); ++local) {
    const int i = index[local];
    F_ASSERT_GE(i, 0);
    F_ASSERT_LT(i, input.rows(), "Index exceeds the number of params");
    output->operator[](local) = input[i];
  }
}

template <int ResidualDim, int ParamsDim, typename F>
typename Residual::Model<ResidualDim, ParamsDim, F>::ParamType
Residual::Model<ResidualDim, ParamsDim, F>::GatherParams(const Eigen::VectorXd& params) const {
  // TODO(gareth): Create cached storage for this.
  ParamType p;
  GatherValues<ParamsDim>(params, index_, &p);
  return p;
}

template <int ResidualDim, int ParamsDim, typename F>
void Residual::Model<ResidualDim, ParamsDim, F>::ErrorVector(
    const Eigen::VectorXd& params, Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
  F_ASSERT_EQ(b_out.rows(), Dimension(), "Output vector is wrong dimension");
  const ParamType relevant_params = GatherParams(params);
  b_out = func_(relevant_params, nullptr);
}

// TODO(gareth): Probably faster to associate a dimension to each variable,
// in the style of GTSAM, so that we can do block-wise updates. For now this
// suits the small problem size I am doing.
template <int ResidualDim, int ParamsDim, typename F>
double Residual::Model<ResidualDim, ParamsDim, F>::UpdateHessian(const Eigen::VectorXd& params,
                                                                 Eigen::MatrixXd* const H,
                                                                 Eigen::VectorXd* const b) const {
  F_ASSERT(H != nullptr);
  F_ASSERT(b != nullptr);
  F_ASSERT_EQ(H->rows(), H->cols());
  F_ASSERT_EQ(b->rows(), H->rows());

  // Collect params.
  const ParamType relevant_params = GatherParams(params);

  // Evaluate the function and its derivative.
  JacobianType J;
  if constexpr (ParamsDim == Eigen::Dynamic) {
    J.resize(ResidualDim, index_.size());
  }
  const ResidualType r = func_(relevant_params, &J);

  // Add contributions to the hessian, only lower part.
  const int N = static_cast<int>(index_.size());
  for (int row_local = 0; row_local < N; ++row_local) {
    // get index mapping into the full system
    const int row_global = index_[row_local];
    F_ASSERT_LT(row_global, H->rows(), "Index exceeds bounds of hessian");
    for (int col_local = 0; col_local <= row_local; ++col_local) {
      // because col_local <= row_local, we already checked this global index
      const int col_global = index_[col_local];
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
template <int ResidualDim, int ParamsDim, typename F>
void Residual::Model<ResidualDim, ParamsDim, F>::UpdateJacobian(
    const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
    Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
  F_ASSERT_EQ(ResidualDim, b_out.rows());
  F_ASSERT_EQ(ResidualDim, J_out.rows());
  // Collect params.
  const ParamType relevant_params = GatherParams(params);

  // Evaluate, and copy jacobian back using indices.
  JacobianType J;
  if constexpr (ParamsDim == Eigen::Dynamic) {
    J.resize(ResidualDim, index_.size());
  }
  b_out.noalias() = func_(relevant_params, &J);

  for (int col_local = 0; col_local < static_cast<int>(index_.size()); ++col_local) {
    const int col_global = index_[col_local];
    F_ASSERT_LT(col_global, J_out.cols(), "Index exceeds the size of the Jacobian");
    J_out.col(col_global).noalias() = J.col(col_local);
  }
}

}  // namespace mini_opt
