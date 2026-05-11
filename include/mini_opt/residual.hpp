// Copyright 2021 Gareth Cross
#pragma once
#include <array>
#include <memory>

#include <Eigen/Core>

#include "mini_opt/assertions.hpp"
#include "mini_opt/key.hpp"
#include "mini_opt/residual_impl.hpp"
#include "mini_opt/scatter.hpp"
#include "mini_opt/values.hpp"

// Provide mechanisms of specifying non-linear residuals.
namespace mini_opt {

// Type for residual in the optimization.
class residual final {
 public:
  explicit residual(std::unique_ptr<residual_concept> impl) noexcept : impl_(std::move(impl)) {}

  // Dimension of the residual vector.
  int residual_dimension() const { return impl_->residual_dimension(); }

  // Get the error vector: f(x)
  void error_vector(const values& v, Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
    return impl_->error_vector(v, b_out);
  }

  // Update a system of equations Hx=b by writing to `H` and `b`.
  // Returns the value of `Error` as well (the constant part of the quadratic).
  double update_hessian(const values& v, const scatter& scatter, Eigen::MatrixXd* H,
                        Eigen::VectorXd* b) const {
    return impl_->update_hessian(v, scatter, H, b);
  }

  // Output the jacobian for the linear system: J * dx + b
  // `J_out` and `b_out` are set to the correct rows of a larger matrix.
  void update_jacobian(const values& v, const scatter& scatter, Eigen::Block<Eigen::MatrixXd> J_out,
                       Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
    return impl_->update_jacobian(v, scatter, J_out, b_out);
  }

  // Helper for tests for computing L2 error.
  double error(const values& v) const {
    MINI_OPT_ASSERT(impl_);
    Eigen::VectorXd err;
    err.resize(residual_dimension());
    impl_->error_vector(v, err.head(residual_dimension()));
    return 0.5 * err.squaredNorm();
  }

 private:
  template <typename F, typename... Ks>
  friend residual make_residual(F&& func, Ks&&... keys);

  std::unique_ptr<residual_concept> impl_;
};

// Construct a residual that ingests `P` params, and produces an `R` dimensional error vector.
template <typename F, typename... Ks>
residual make_residual(F&& func, Ks&&... keys) {
  static_assert(std::conjunction_v<is_valid_key<Ks>...>);
  using FuncType = std::remove_const_t<std::remove_reference_t<F>>;

  using func_traits = function_traits<FuncType>;
  static_assert(func_traits::arity == sizeof...(Ks) * 2,
                "Number of keys must be half the number of function arguments");

  return residual(std::make_unique<fixed_size_residual<FuncType>>(
      std::array<key, sizeof...(Ks)>{{key(std::forward<Ks>(keys))...}}, std::forward<F>(func)));
}

}  // namespace mini_opt
