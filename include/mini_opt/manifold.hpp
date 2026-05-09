// Copyright 2020 Gareth Cross
#pragma once
#include <type_traits>
#include "mini_opt/eigen_traits.hpp"

namespace mini_opt {

// Unspecified manifold traits.
template <typename T, typename = void>
struct manifold_trait;

// Operations on quaternions.
template <typename T>
struct manifold_trait<T, enable_if_inherits_quaternion_base_t<T>> {
  // Quaternions are convertible to so(3).
  static constexpr int tangent_dim = 3;
  using Scalar = typename T::Scalar;
  using VectorType = Eigen::Matrix<Scalar, tangent_dim, 1>;

  // Map from the manifold to a vector in the tangent space of `x`.
  static VectorType From(const T& x, const T& y) { return RotationLog(x.conjugate() * y); }

  // Map a vector to the manifold in the tangent space of `x`.
  template <typename Derived>
  static T To(const T& x, const Eigen::MatrixBase<Derived>& v) {
    return x * QuaternionExp(v);
  }

  // Get runtime dimension of the tangent space of an object.
  static constexpr int TangentDimension(const T& x) {
    (void)x;  //  silence un-referenced parameter in MSVC
    return tangent_dim;
  }
};

// Traits on vector types.
template <typename T>
struct manifold_trait<T, enable_if_inherits_matrix_base_t<T>> {
  static_assert(Eigen::MatrixBase<T>::ColsAtCompileTime == 1, "Must be a vector");

  // Inherit dimensionality from the vector itself.
  static constexpr int tangent_dim = Eigen::MatrixBase<T>::RowsAtCompileTime;
  using Scalar = typename T::Scalar;
  using VectorType = Eigen::Matrix<Scalar, tangent_dim, 1>;

  // Defined so numericalDerivative works.
  static VectorType From(const VectorType& x, const VectorType& y) { return y - x; }

  // Defined so numericalDerivative works.
  template <typename Derived>
  static VectorType To(const T& x, const Eigen::MatrixBase<Derived>& v) {
    return x + v;
  }

  // Runtime dimension of vector.
  static int TangentDimension(const T& x) { return static_cast<int>(x.rows()); }
};

// Traits on floats/doubles.
template <typename T>
struct manifold_trait<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  static constexpr int tangent_dim = 1;
  using Scalar = T;
  using VectorType = Eigen::Matrix<Scalar, tangent_dim, 1>;

  // Defined so numericalDerivative works.
  static VectorType From(const T& x, const T& y) { return VectorType{y - x}; }

  // Defined so numericalDerivative works.
  template <typename Derived>
  static T To(const T& x, const Eigen::MatrixBase<Derived>& v) {
    // If we can check at compile time, enforce this is a 1x1 matrix.
    static_assert(Eigen::MatrixBase<Derived>::RowsAtCompileTime == Eigen::Dynamic ||
                      Eigen::MatrixBase<Derived>::RowsAtCompileTime == 1,
                  "Must be a scalar");
    static_assert(Eigen::MatrixBase<Derived>::ColsAtCompileTime == Eigen::Dynamic ||
                      Eigen::MatrixBase<Derived>::ColsAtCompileTime == 1,
                  "Must be a scalar");
    return x + static_cast<T>(v[0]);
  }

  // Runtime dimension of vector.
  static constexpr int TangentDimension(const T& x) {
    (void)x;
    return tangent_dim;
  }
};

}  // namespace mini_opt
