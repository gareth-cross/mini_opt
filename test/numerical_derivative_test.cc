// Copyright 2020 Gareth Cross
#include "geometry_utils/numerical_derivative.hpp"

#include <cmath>

#include "geometry_utils/so3.hpp"
#include "test_utils.hpp"

namespace math {

// Shorthand for tests.
using Vector2d = Vector<double, 2>;
using Vector3d = Vector<double, 3>;
using Matrix2d = Matrix<double, 2, 2>;
using Matrix3d = Matrix<double, 3, 3>;

// TODO(gareth): Add more principled bounds on max error.
// Some of these tolerances may be somewhat pessimistic.
// We test w/ h = 0.01, which should produce error terms on the order of h^6 = 1e-12
TEST(NumericalDerivativeTest, TestCentralDiff) {
  // a linear equation
  const auto linear = [](double x) { return 2 * x + 0.5; };
  for (double h : {1.0, 0.1, 0.01, 0.001}) {
    // central difference should be exact (up to float error) for linear (independent of h)
    EXPECT_NEAR(2.0, NumericalDerivative(0.0, h, linear), tol::kPico);
  }

  // a second order polynomial
  const auto parabola = [](double x) { return 2 * (x * x) - x + 1.0; };
  const auto parabola_diff = [](double x) { return 4 * x - 1.0; };
  // compare to analytical solution
  for (double x : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
    EXPECT_NEAR(parabola_diff(x), NumericalDerivative(x, 0.01, parabola), tol::kNano);
  }

  // sinusoid
  const auto sinusoid = [](double x) { return std::sin(x); };
  const auto sinusoid_diff = [](double x) { return std::cos(x); };
  for (double x : Range(-M_PI, M_PI, 0.01)) {
    EXPECT_NEAR(sinusoid_diff(x), NumericalDerivative(x, 0.01, sinusoid), tol::kPico);
  }

  // exponential
  const auto exp = [](double x) { return std::exp(x); };
  for (double x : {-10.0, -1.0, -0.1, 0.0, 1.0, 10.0}) {
    // higher tolerance here for x=10
    EXPECT_NEAR(exp(x), NumericalDerivative(x, 0.01, exp), tol::kNano);
  }
}

TEST(NumericalDerivativeTest, TestJacobianLinear) {
  // a simple first order system of size 2
  // clang-format off
  const Matrix2d A = (Matrix2d() <<
                      1.0, 0.2,
                      0.0, 0.5).finished();
  // clang-format on
  const auto linear = [&](const Vector2d& x) -> Vector2d { return A * x + Vector2d(0.1, -0.3); };
  for (double x1 : {-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0}) {
    for (double x2 : {-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0}) {
      // should be exact up to float error
      EXPECT_EIGEN_NEAR(A, NumericalJacobian(Vector2d(x1, x2), linear), tol::kPico);
    }
  }
}

TEST(NumericalDerivativeTest, TestJacobianNonlinear) {
  // a vector in polar coordinates, [theta, r]
  const auto rot_2d = [](const Vector2d& x) -> Vector2d {
    return Vector2d(std::cos(x[0]), std::sin(x[0])) * x[1];
  };
  const auto rot_2d_diff = [](const Vector2d& x) -> Matrix2d {
    // clang-format off
    return (Matrix2d() <<
            -std::sin(x[0]) * x[1], std::cos(x[0]),
             std::cos(x[0]) * x[1], std::sin(x[0])).finished();
    // clang-format on
  };
  for (double theta : Range(-M_PI, M_PI, 0.05)) {
    for (double r : Range(-5.0, 5.0, 0.5)) {
      const Vector2d x(theta, r);
      EXPECT_EIGEN_NEAR(rot_2d_diff(x), NumericalJacobian(x, rot_2d), tol::kPico);
    }
  }

  // the unit sphere, [theta, phi]
  const auto sphere_3d = [](const Vector2d& x) -> Vector3d {
    return Vector3d(std::sin(x[0]) * std::cos(x[1]), std::sin(x[0]) * std::sin(x[1]),
                    std::cos(x[0]));
  };
  const auto sphere_3d_diff = [](const Vector2d& x) -> Matrix<double, 3, 2> {
    // clang-format off
    return (Matrix<double, 3, 2>() <<
            std::cos(x[0]) * std::cos(x[1]), -std::sin(x[0]) * std::sin(x[1]),
            std::cos(x[0]) * std::sin(x[1]),  std::sin(x[0]) * std::cos(x[1]),
            -std::sin(x[0]), 0.0).finished();
    // clang-format on
  };
  for (double theta : Range(-M_PI, M_PI, 0.05)) {
    for (double phi : Range(-M_PI / 2, M_PI / 2, 0.05)) {
      const Vector2d x(theta, phi);
      EXPECT_EIGEN_NEAR(sphere_3d_diff(x), NumericalJacobian(x, sphere_3d), tol::kPico);
    }
  }
}

TEST(NumericalDerivativeTest, TestQuaternionCompose) {
  // Test using quaternion traits in numericalJacobian
  const Quaternion<double> q1 = QuaternionExp(Vector3d(-0.1, 0.5, 0.7));
  const Quaternion<double> q2 = QuaternionExp(Vector3d(0.2, 0.01, -0.3));
  // take wrt q2
  const Matrix3d J_numerical_right =
      NumericalJacobian(q2, [&](const Quaternion<double>& q2) { return q1 * q2; });
  EXPECT_EIGEN_NEAR(Matrix3d::Identity(), J_numerical_right, tol::kNano);
  // take wrt q1
  const Matrix3d J_numerical_left =
      NumericalJacobian(q1, [&](const Quaternion<double>& q1) { return q1 * q2; });
  // should be the adjoint of q2^-1
  EXPECT_EIGEN_NEAR(q2.conjugate().matrix(), J_numerical_left, tol::kNano);
}

// Check we can compute the expected angular velocity on `a_R_b`, implied
// by two angular velocities observed in `a` and `b`.
TEST(NumericalDerivativeTest, TestLeftRightAngularVelocity) {
  // make up two rotations
  const Quaternion<double> ref_R_a = QuaternionExp(Vector3d(-0.4, 0.6, 0.2));
  const Quaternion<double> ref_R_b = QuaternionExp(Vector3d(0.8, 0.5, -0.6));
  // create two angular velocities
  const Vector3d w_a{-0.1, 0.2, -0.3};
  const Vector3d w_b{0.02, 0.8, 0.2};
  // add them on the left and right and compute a_R_b
  const auto integrate_ab = [&](const double& t) {
    const Quaternion<double> ref_R_a_int = ref_R_a * QuaternionExp(w_a * t);
    const Quaternion<double> ref_R_b_int = ref_R_b * QuaternionExp(w_b * t);
    return ref_R_a_int.conjugate() * ref_R_b_int;
  };
  const Vector3d J_time = NumericalJacobian(0.0, integrate_ab);
  const Vector3d J_time_expected = -(ref_R_b.conjugate() * ref_R_a * w_a) + w_b;
  EXPECT_EIGEN_NEAR(J_time_expected, J_time, tol::kPico);
}

// Test numerical jacobian with a dynamic sized vector.
TEST(NumericalDerivativeTest, TestDynamicSize) {
  const Vector<double> x = (Vector<double, 6>() << -0.5, 0.2, 0.8, 1.2, 0.5, -0.2).finished();

  // dynamic input, static output
  {
    const Matrix<double, 3, Eigen::Dynamic> J_numerical =
        NumericalJacobian(x, [](const Vector<double>& x) -> Vector<double, 3> {
          Matrix<double, 3, 3> A;
          A.setZero();
          A.diagonal() = x.head<3>();
          return A * x.tail<3>();
        });
    const Matrix<double, 3, 3> J_1 = x.tail(3).asDiagonal();
    const Matrix<double, 3, 3> J_2 = x.head(3).asDiagonal();
    EXPECT_EIGEN_NEAR(J_1, J_numerical.block(0, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(J_2, J_numerical.block(0, 3, 3, 3), tol::kPico);
  }

  // both input and output are dynamic
  {
    const Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J_numerical =
        NumericalJacobian(x, [](const Vector<double>& x) -> Vector<double> {
          Matrix<double, 3, 3> A;
          A.setZero();
          A.diagonal() = x.head<3>();
          return A * x.tail<3>() * -0.25;
        });
    const Matrix<double, 3, 3> J_1 = x.tail(3).asDiagonal() * -0.25;
    const Matrix<double, 3, 3> J_2 = x.head(3).asDiagonal() * -0.25;
    EXPECT_EIGEN_NEAR(J_1, J_numerical.block(0, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(J_2, J_numerical.block(0, 3, 3, 3), tol::kPico);
  }
}

// Test a 1x1 jacobian.
TEST(NumericalDerivativeTest, TestScalarJacobian) {
  const double x = 1.123;
  const Matrix<double, 1, 1> J_numerical =
      NumericalJacobian(x, [](const double x) { return std::sin(x) - x; });
  ASSERT_NEAR(std::cos(x) - 1, J_numerical[0], tol::kPico);
}

}  // namespace math
