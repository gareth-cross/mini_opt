// Copyright 2020 Gareth Cross
#pragma once
#include <Eigen/Geometry>
#include <type_traits>
#include "mini_opt/eigen_traits.hpp"

namespace mini_opt {

// Unspecified manifold traits.
template <typename T, typename = void>
struct manifold_trait;

template <typename T, typename = void>
struct implements_manifold_t : std::false_type {};
template <typename T>
struct implements_manifold_t<
    T,
    std::void_t<decltype(manifold_trait<T>::tangent_dim), typename manifold_trait<T>::scalar_type,
                typename manifold_trait<T>::tangent_vector,
                decltype(manifold_trait<T>::local_coordinates(std::declval<const T&>(),
                                                              std::declval<const T&>())),
                decltype(manifold_trait<T>::retract(
                    std::declval<const T&>(),
                    std::declval<const typename manifold_trait<T>::tangent_vector&>()))>>
    : std::true_type {};

template <typename T>
constexpr bool implements_manifold_v = implements_manifold_t<T>::value;

// Quaternion from rotation vector.
template <typename Derived,
          typename = enable_if_is_vector3_at_compile_time_t<Eigen::MatrixBase<Derived>>>
Eigen::Quaternion<scalar_type_t<Derived>> quaternion_exp(const Eigen::MatrixBase<Derived>& w_xpr) {
  using Scalar = scalar_type_t<Derived>;
  const Scalar angle = w_xpr.norm();
  // Fill out the quaternion.
  Eigen::Quaternion<Scalar> q;
  q.w() = std::cos(angle / 2);
  if (angle < static_cast<Scalar>(1.0e-9)) {
    q.vec() = w_xpr / 2;
    q.normalize();
  } else {
    const Scalar sinc_ha_2 = std::sin(angle / 2) / angle;
    q.vec() = w_xpr * sinc_ha_2;
  }
  return q;
}

// Operations on quaternions. Quaternions are optimized on so(3).
template <typename T>
struct manifold_trait<T, enable_if_inherits_quaternion_base_t<T>> {
  static constexpr int tangent_dim = 3;
  using self_type = T;
  using scalar_type = typename T::Scalar;
  using tangent_vector = Eigen::Matrix<scalar_type, tangent_dim, 1>;

  // y [-] x
  static tangent_vector local_coordinates(const self_type& y, const self_type& x) {
    const Eigen::AngleAxis<scalar_type> angle_axis(x.conjugate() * y);
    return angle_axis.angle() * angle_axis.axis();
  }

  // Map from the manifold to a vector in the tangent space of `x`.
  // x [+] dx
  static self_type retract(const self_type& x, const tangent_vector& dx) {
    return (x * quaternion_exp(dx)).normalized();
  }

  static constexpr int tangent_dimension(const T&) { return tangent_dim; }
};

// Traits on vector types.
template <typename T>
struct manifold_trait<T, enable_if_inherits_matrix_base_t<T>> {
  static_assert(Eigen::MatrixBase<T>::ColsAtCompileTime == 1, "Must be a vector");

  static constexpr int tangent_dim = Eigen::MatrixBase<T>::RowsAtCompileTime;
  using self_type = T;
  using scalar_type = typename T::Scalar;
  using tangent_vector = Eigen::Matrix<scalar_type, tangent_dim, 1>;

  static tangent_vector local_coordinates(const self_type& y, const self_type& x) { return y - x; }

  template <typename Derived>
  static tangent_vector retract(const T& x, const Eigen::MatrixBase<Derived>& dx) {
    return x + dx;
  }

  // Runtime dimension of vector.
  static int tangent_dimension(const T& x) { return static_cast<int>(x.rows()); }
};

// Traits on floats/doubles.
template <typename T>
struct manifold_trait<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  static constexpr int tangent_dim = 1;
  using self_type = T;
  using scalar_type = T;
  using tangent_vector = Eigen::Matrix<scalar_type, tangent_dim, 1>;

  // Defined so numericalDerivative works.
  static tangent_vector local_coordinates(const T& y, const T& x) { return tangent_vector{y - x}; }

  // Defined so numericalDerivative works.
  template <typename Derived>
  static T retract(const T& x, const Eigen::MatrixBase<Derived>& dx) {
    // If we can check at compile time, enforce this is a 1x1 matrix.
    static_assert(Eigen::MatrixBase<Derived>::RowsAtCompileTime == Eigen::Dynamic ||
                      Eigen::MatrixBase<Derived>::RowsAtCompileTime == 1,
                  "Must be a scalar");
    static_assert(Eigen::MatrixBase<Derived>::ColsAtCompileTime == Eigen::Dynamic ||
                      Eigen::MatrixBase<Derived>::ColsAtCompileTime == 1,
                  "Must be a scalar");
    return x + static_cast<T>(dx[0]);
  }

  static constexpr int tangent_dimension(const T&) { return tangent_dim; }
};

static_assert(implements_manifold_v<Eigen::Quaterniond>);
static_assert(implements_manifold_v<Eigen::Vector3d>);

}  // namespace mini_opt
