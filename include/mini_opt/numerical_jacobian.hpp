// Copyright 2020 Gareth Cross
#pragma once
#include "mini_opt/manifold.hpp"

// Turn off warning about constant if statements.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif  // _MSC_VER

namespace mini_opt {

/**
 * Numerically compute first derivative of f(x) via central difference. Uses the third-order
 * approximation, which has error in O(h^6).
 *
 * The function `func` is presumed to be centered on the linearization point `x`, such that only
 * the step increment `dx` (a scalar) is passed as an argument.
 *
 * References:
 * http://www.rsmas.miami.edu/personal/miskandarani/Courses/MSC321/lectfiniteDifference.pdf
 * https://en.wikipedia.org/wiki/Finite_difference_coefficient
 */
template <typename Scalar, typename Function>
auto numerical_derivative2(const Scalar dx, Function func) -> decltype(func(dx)) {
  using ResultType = decltype(func(dx));
  const Scalar dx2 = dx * 2;
  const Scalar dx3 = dx * 3;
  const ResultType c1 = func(dx) - func(-dx);
  const ResultType c2 = func(dx2) - func(-dx2);
  const ResultType c3 = func(dx3) - func(-dx3);
  return (c1 * 45 - c2 * 9 + c3) / (60 * dx);
}

/**
 * Version of NumericalDerivative2 that accepts linearization point `x` and step-size `h`
 * separately. This variant does not require that `func` be centered around the linearization point.
 */
template <typename Scalar, typename Function>
auto numerical_derivative(const Scalar x, const Scalar h, Function func) -> decltype(func(x)) {
  return numerical_derivative2(h, [&](const Scalar dx) { return func(x + dx); });
}

/**
 * Numerically compute the jacobian of vector function `y = f(x)` via the central-difference. `func`
 * accepts type `XExpr` and returns type `YExpr`, both of which may be manifolds. This method uses
 * the Manifold<> trait to determine how make the manifold locally euclidean.
 */
template <typename XExpr, typename Function>
auto numerical_jacobian(const XExpr& x, Function func, const double h = 0.01) {
  using YExpr = typename std::decay<decltype(func(x))>::type;
  constexpr int DimX = mini_opt::manifold_trait<XExpr>::tangent_dim;
  constexpr int DimY = mini_opt::manifold_trait<YExpr>::tangent_dim;
  using Scalar = typename mini_opt::manifold_trait<XExpr>::Scalar;

  // Compute the output expression at the linearization point.
  const YExpr y_0 = func(x);

  // Possibly allocate for the result, since dimensions may be dynamic.
  Eigen::Matrix<Scalar, DimY, DimX> J;
  if (DimX == Eigen::Dynamic || DimY == Eigen::Dynamic) {
    J.resize(mini_opt::manifold_trait<YExpr>::TangentDimension(y_0),
             mini_opt::manifold_trait<XExpr>::TangentDimension(x));
  }

  // Pre-allocate `delta` once and re-use it.
  Eigen::Matrix<Scalar, DimX, 1> delta;
  if (DimX == Eigen::Dynamic) {
    delta.resize(mini_opt::manifold_trait<XExpr>::TangentDimension(x));
  }

  for (int j = 0; j < mini_opt::manifold_trait<XExpr>::TangentDimension(x); ++j) {
    // Take derivative wrt dimension `j` of X
    const auto wrapped = [&](const Scalar dx) {
      // apply perturbation in the tangent space
      delta.setZero();
      delta[j] = dx;
      // Perform the operation: x [+] f(dx), where [+] is the manifold composition.
      const auto x_oplus_dx = mini_opt::manifold_trait<XExpr>::To(x, delta);
      const auto y = func(x_oplus_dx);
      // determine the perturbation in y: dy = f^-1(y^-1 [+] y)
      // where f() maps to and from the manifold
      return mini_opt::manifold_trait<YExpr>::From(y_0, y);
    };
    J.col(j) = numerical_derivative2(static_cast<Scalar>(h), wrapped);
  }
  return J;
}

}  // namespace mini_opt

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
