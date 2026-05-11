#pragma once
#include <Eigen/Core>

#include "mini_opt/key.hpp"

namespace mini_opt {

/*
 * Describes a linear (technically affine) inequality constraint.
 *
 * The constraint is specified in the form:
 *
 *    a * x[variable] + b >= 0
 */
struct linear_inequality {
  // Key of the variable this refers to.
  key variable;
  int index;

  // Constraint coefficients.
  double a;
  double b;

  // Construct with index and coefficients.
  linear_inequality(key variable, int index, double a, double b) noexcept
      : variable(variable), index(index), a(a), b(b) {}

  // Shift to a new linearization point.
  // a*(x + dx) + b >= 0  -->  a*dx + (ax + b) >= 0
  linear_inequality shift_to(double x) const noexcept { return {variable, index, a, a * x + b}; }
};

/*
 * Helper for specifying constraints in a more legible way.
 *
 * Allows you to write Var(index) >= alpha to specify the appropriate LinearInequalityConstraint.
 */
struct var {
  explicit var(key variable, int index) noexcept : variable_(variable), index_(index) {}

  // Specify constraint as <=
  linear_inequality operator<=(double value) const noexcept {
    return linear_inequality(variable_, index_, -1.0, value);
  }

  // Specify constraint as >=
  linear_inequality operator>=(double value) const noexcept {
    return linear_inequality(variable_, index_, 1.0, -value);
  }

 private:
  key variable_;
  int index_;
};

}  // namespace mini_opt
