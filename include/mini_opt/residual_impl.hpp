#pragma once
#include <Eigen/Core>
#include <optional>

#include "mini_opt/function_traits.hpp"
#include "mini_opt/index_sequence_for.hpp"
#include "mini_opt/scatter.hpp"
#include "mini_opt/values.hpp"

namespace mini_opt {

// Helper for declaring either vector or array, depending on whether size is known at compile time.
namespace internal {

template <typename... Ts>
constexpr std::integer_sequence<int, sizeof...(Ts)> get_tangent_dims_at_compile_time(
    type_list<Ts...>) {
  return std::integer_sequence<int, manifold_trait<std::remove_cvref_t<Ts>>::tangent_dim...>{};
}

template <typename... Ts>
std::array<int, sizeof...(Ts)> get_tangent_dims_at_runtime(const std::tuple<const Ts&...>& args) {
  return std::apply(
      [](const auto&... args) {
        return std::array<int, sizeof...(Ts)>{
            manifold_trait<std::remove_cvref_t<decltype(args)>>::tangent_dimension(args)...};
      },
      args);
}

template <int Rows, int Cols>
auto map_from_eigen_matrix(Eigen::Matrix<double, Rows, Cols>& m) {
  return Eigen::Map<Eigen::Matrix<double, Rows, Cols>>(m.data(), m.rows(), m.cols());
}

}  // namespace internal

// Get the Jacobian type for a given residual dimension and parameter type.
template <int ResidualDim, typename T>
using jacobian_type_t =
    Eigen::Matrix<double, ResidualDim, manifold_trait<std::remove_cvref_t<T>>::tangent_dim>;

template <int ResidualDim, typename T>
using jacobian_map_type_t = Eigen::Map<
    Eigen::Matrix<double, ResidualDim, manifold_trait<std::remove_cvref_t<T>>::tangent_dim>>;

template <int ResidualDim, typename T>
struct jacobian_type_list;

template <int ResidualDim, typename... Ts>
struct jacobian_type_list<ResidualDim, type_list<Ts...>> {
  using type = type_list<jacobian_type_t<ResidualDim, Ts>...>;
};

template <int ResidualDim, typename T>
using jacobian_type_list_t = typename jacobian_type_list<ResidualDim, T>::type;

template <int ResidualDim, typename... Ts>
struct jacobian_map_type_list;

template <int ResidualDim, typename... Ts>
struct jacobian_map_type_list<ResidualDim, type_list<Ts...>> {
  using type = type_list<jacobian_map_type_t<ResidualDim, Ts>...>;
};

template <int ResidualDim, typename... Ts>
using jacobian_map_tuple_type = std::tuple<jacobian_map_type_t<ResidualDim, Ts>...>;

template <int ResidualDim, std::size_t N>
using jacobian_array_type =
    std::array<Eigen::Map<const Eigen::Matrix<double, ResidualDim, Eigen::Dynamic>>, N>;

// Create a tuple of Jacobian maps, where all maps are initialized to nullptr.
template <int ResidualDim, typename... Ts>
jacobian_map_tuple_type<ResidualDim, Ts...> make_empty_jacobian_map_tuple(type_list<Ts...>) {
  return index_sequence_for(
      []<std::size_t I>(std::integral_constant<std::size_t, I>) {
        using T = type_list_element_t<I, type_list<Ts...>>;
        return jacobian_map_type_t<ResidualDim, T>(nullptr);
      },
      std::make_index_sequence<sizeof...(Ts)>());
}

// Gather a fixed number of values into a tuple.
template <typename Container, typename... Ts>
std::tuple<const Ts&...> gather_values(const values& v, const Container& keys, type_list<Ts...>) {
  MINI_OPT_ASSERT_EQ(keys.size(), sizeof...(Ts), "Number of keys must match the number of types");
  return index_sequence_for(
      [&v, &keys]<std::size_t I>(std::integral_constant<std::size_t, I>) -> const auto& {
        using T = type_list_element_t<I, type_list<Ts...>>;
        return v.at<T>(keys[I]);
      },
      std::index_sequence_for<Ts...>{});
}

template <int R, typename T>
struct function_evaluator;

// Evaluate a user-provided function and capture its output value and jacobians.
template <int R, typename... ParamTypes>
struct function_evaluator<R, type_list<ParamTypes...>> {
  using jacobian_list = jacobian_type_list_t<R, type_list<ParamTypes...>>;

  static constexpr std::size_t num_params = sizeof...(ParamTypes);

  // Residual vector.
  Eigen::Matrix<double, R, 1> f;

  // Jacobians.
  tuple_from_type_list_t<jacobian_list> J;

  // Jacobian dimensions.
  std::array<int, num_params> jacobian_dims;

  template <typename F>
  void compute(const std::tuple<const ParamTypes&...>& params, F&& func) {
    // Get the runtime sizes of the jacobians for each parameter.
    jacobian_dims = internal::get_tangent_dims_at_runtime(params);

    index_sequence_for(
        [&]<std::size_t I>(std::integral_constant<std::size_t, I>) {
          auto& J_i = std::get<I>(J);
          if constexpr (std::remove_reference_t<decltype(J_i)>::ColsAtCompileTime ==
                        Eigen::Dynamic) {
            J_i.resize(R, jacobian_dims[I]);
          }
        },
        std::make_index_sequence<num_params>{});

    const auto J_maps = std::apply(
        [](auto&... J_i) { return std::make_tuple(internal::map_from_eigen_matrix(J_i)...); }, J);

    // Invoke the residual itself. Pass maps for the jacobians.
    f = std::apply(
        [&](const auto&... args) -> Eigen::Matrix<double, R, 1> { return func(args...); },
        std::tuple_cat(params, J_maps));
  }

  template <typename F>
  void compute(const values& v, const std::array<key, num_params>& keys, F&& func) {
    compute(gather_values(v, keys, type_list<ParamTypes...>{}), std::forward<F>(func));
  }

  template <typename F>
  void compute_no_jacobians(const std::tuple<const ParamTypes&...>& params, F&& func) {
    f = std::apply(
        [&](const auto&... args) -> Eigen::Matrix<double, R, 1> { return func(args...); },
        std::tuple_cat(params, make_empty_jacobian_map_tuple<R>(type_list<ParamTypes...>{})));
  }

  template <typename F>
  void compute_no_jacobians(const values& v, const std::array<key, num_params>& keys, F&& func) {
    compute_no_jacobians(gather_values(v, keys, type_list<ParamTypes...>{}), std::forward<F>(func));
  }

  // Get array of dynamically sized maps, one per output jacobian.
  jacobian_array_type<R, num_params> jacobian_array() const {
    return std::apply(
        [](const auto&... J_i) {
          return jacobian_array_type<R, num_params>{
              Eigen::Map<const Eigen::Matrix<double, R, Eigen::Dynamic>>{J_i.data(), J_i.rows(),
                                                                         J_i.cols()}...};
        },
        this->J);
  }
};

class residual_concept {
 public:
  virtual ~residual_concept() = default;

  virtual int residual_dimension() const noexcept = 0;

  virtual void error_vector(const values& v, Eigen::VectorBlock<Eigen::VectorXd> b_out) const = 0;

  virtual double update_hessian(const values& v, const scatter& scatter, Eigen::MatrixXd* H,
                                Eigen::VectorXd* b) const = 0;

  virtual void update_jacobian(const values& v, const scatter& scatter,
                               Eigen::Block<Eigen::MatrixXd> J_out,
                               Eigen::VectorBlock<Eigen::VectorXd> b_out) const = 0;
};

// A concrete implementation of a residual.
template <typename F>
class fixed_size_residual final : public residual_concept {
 public:
  using return_type = function_traits<F>::return_type;

  static_assert(inherits_matrix_base_v<return_type> && return_type::ColsAtCompileTime == 1,
                "Return values must be vectors.");

  static constexpr int residual_dimension_at_compile_time = return_type::RowsAtCompileTime;
  static_assert(residual_dimension_at_compile_time != Eigen::Dynamic,
                "Residual dimension must be known at compile time.");

  using arg_types = function_traits<F>::args_list;
  static constexpr std::size_t num_args = type_list_size_v<arg_types>;

  static_assert(num_args >= 1 && num_args % 2 == 0,
                "Residuals must have an even number of arguments.");

  using param_types = type_list_take_n_t<num_args / 2, arg_types>;
  using intrinsic_param_types = type_list_map_t<std::remove_cvref_t, param_types>;

  static constexpr std::size_t num_input_values = type_list_size_v<intrinsic_param_types>;

  using keys_container_type = std::array<key, num_input_values>;

  fixed_size_residual(keys_container_type keys, F func) noexcept
      : keys_(std::move(keys)), func_(std::move(func)) {}

  int residual_dimension() const noexcept override { return residual_dimension_at_compile_time; }

  void error_vector(const values& v, Eigen::VectorBlock<Eigen::VectorXd> b_out) const override;

  double update_hessian(const values& v, const scatter& scatter, Eigen::MatrixXd* H,
                        Eigen::VectorXd* b) const override;

  void update_jacobian(const values& v, const scatter& scatter, Eigen::Block<Eigen::MatrixXd> J_out,
                       Eigen::VectorBlock<Eigen::VectorXd> b_out) const override;

 private:
  keys_container_type keys_;
  F func_;
};

//
// Template implementations.
//

template <typename F>
void fixed_size_residual<F>::error_vector(const values& v,
                                          Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
  MINI_OPT_ASSERT_EQ(b_out.rows(), residual_dimension(), "Output vector is wrong dimension");
  function_evaluator<residual_dimension_at_compile_time, intrinsic_param_types> eval{};
  eval.compute_no_jacobians(v, keys_, func_);
  b_out = eval.f;
}

template <typename F>
double fixed_size_residual<F>::update_hessian(const values& v, const scatter& scatter,
                                              Eigen::MatrixXd* const H,
                                              Eigen::VectorXd* const b) const {
  MINI_OPT_ASSERT(H != nullptr);
  MINI_OPT_ASSERT(b != nullptr);
  MINI_OPT_ASSERT_EQ(H->rows(), H->cols(), "Hessian matrix is not square");
  MINI_OPT_ASSERT_EQ(b->rows(), H->rows(), "Output vector is wrong dimension");

  function_evaluator<residual_dimension_at_compile_time, intrinsic_param_types> eval{};
  eval.compute(v, keys_, func_);

  std::array<int, num_input_values> output_pos{};
  for (std::size_t i = 0; i < num_input_values; ++i) {
    output_pos[i] = scatter.at(keys_[i]);
  }

  const auto J_array = eval.jacobian_array();

  // Add contributions to the hessian, only lower triangular part:
  for (int r = 0; r < static_cast<int>(num_input_values); ++r) {
    const int row_global = output_pos[r];
    for (int c = 0; c <= r; ++c) {
      const int col_global = output_pos[c];

      // lower triangular part (row >= col)
      if (row_global >= col_global) {
        H->block(row_global, col_global, eval.jacobian_dims[r], eval.jacobian_dims[c]).noalias() +=
            J_array[r].transpose() * J_array[c];
      } else {
        H->block(col_global, row_global, eval.jacobian_dims[c], eval.jacobian_dims[r]).noalias() +=
            J_array[c].transpose() * J_array[r];
      }
    }
    b->segment(row_global, eval.jacobian_dims[r]).noalias() += J_array[r].transpose() * eval.f;
  }
  return 0.5 * eval.f.squaredNorm();
}

// This version takes blocks, so we can write directly into A_eq.
template <typename F>
void fixed_size_residual<F>::update_jacobian(const values& v, const scatter& scatter,
                                             Eigen::Block<Eigen::MatrixXd> J_out,
                                             Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
  MINI_OPT_ASSERT_EQ(residual_dimension_at_compile_time, b_out.rows());
  MINI_OPT_ASSERT_EQ(residual_dimension_at_compile_time, J_out.rows());

  function_evaluator<residual_dimension_at_compile_time, intrinsic_param_types> eval{};
  eval.compute(v, keys_, func_);
  const auto J_array = eval.jacobian_array();

  for (int c = 0; c < static_cast<int>(num_input_values); ++c) {
    const int col_global = scatter.at(keys_[c]);
    MINI_OPT_ASSERT_LT(col_global, J_out.cols(), "Index exceeds the size of the Jacobian");

    J_out.middleCols(col_global, eval.jacobian_dims[c]) = J_array[c];
  }
  b_out.noalias() = eval.f;
}

};  // namespace mini_opt
