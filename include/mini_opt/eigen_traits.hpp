#pragma once
#include <Eigen/Core>
#include <type_traits>

namespace mini_opt {

namespace detail {
constexpr auto inherits_matrix_base_(...) -> std::false_type;
template <typename Derived>
constexpr auto inherits_matrix_base_(const Eigen::MatrixBase<Derived>&) -> std::true_type;

constexpr auto inherits_quaternion_base_(...) -> std::false_type;
template <typename Derived>
constexpr auto inherits_quaternion_base_(const Eigen::QuaternionBase<Derived>&) -> std::true_type;

}  // namespace detail

// Evaluates to std::true_type if `T` inherits from MatrixBase, otherwise std::false_type.
template <typename T>
using inherits_matrix_base = decltype(detail::inherits_matrix_base_(std::declval<const T>()));
template <typename T>
constexpr bool inherits_matrix_base_v = inherits_matrix_base<T>::value;

// Evaluates to `U` if `T` inherits from MatrixBase.
template <typename T, typename U = void>
using enable_if_inherits_matrix_base_t =
    std::enable_if_t<inherits_matrix_base_v<std::decay_t<T>>, U>;

// Enable if the type is a Vector3 at compile time.
template <typename T, typename U = void>
using enable_if_is_vector3_at_compile_time_t =
    std::enable_if_t<inherits_matrix_base_v<std::decay_t<T>> &&
                         std::decay_t<T>::RowsAtCompileTime == 3 &&
                         std::decay_t<T>::ColsAtCompileTime == 1,
                     U>;

// Evaluates to std::true_type if `T` inherits from QuaternionBase, otherwise std::false_type.
template <typename T>
using inherits_quaternion_base =
    decltype(detail::inherits_quaternion_base_(std::declval<const T>()));
template <typename T>
constexpr bool inherits_quaternion_base_v = inherits_quaternion_base<T>::value;

// Evaluates to `U` if `T` inherits from QuaternionBase.
template <typename T, typename U = void>
using enable_if_inherits_quaternion_base_t =
    std::enable_if_t<inherits_quaternion_base_v<std::decay_t<T>>, U>;

// Scalar type of an eigen expression.
template <typename Derived>
using scalar_type_t = typename Eigen::MatrixBase<Derived>::Scalar;

}  // namespace mini_opt
