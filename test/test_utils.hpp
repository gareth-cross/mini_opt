// Copyright 2021 Gareth Cross
#pragma once
#include <gtest/gtest.h>

#include <Eigen/Dense>

// Numerical tolerances for tests.
namespace tol {
static constexpr double kDeci = 1.0e-1;
static constexpr double kCenti = 1.0e-2;
static constexpr double kMilli = 1.0e-3;
static constexpr double kMicro = 1.0e-6;
static constexpr double kNano = 1.0e-9;
static constexpr double kPico = 1.0e-12;
}  // namespace tol

// Print variable w/ name.
#define PRINT(x) printImpl(#x, x)

// Print a matrix.
#define PRINT_MATRIX(x) \
  { std::cout << #x << ":\n" << (x).eval().format(test_utils::kNumPyMatrixFmt) << "\n"; }

template <typename Xpr>
void printImpl(const std::string& name, Xpr xpr) {
  std::cout << name << "=" << xpr << std::endl;
}

// Define a test on a class.
#define TEST_FIXTURE(object, function) \
  TEST_F(object, function) { function(); }

// Macro to compare eigen matrices and print a nice error.
#define EXPECT_EIGEN_NEAR(a, b, tol) \
  EXPECT_PRED_FORMAT3(test_utils::ExpectEigenNear, (a).eval(), (b).eval(), tol)
#define ASSERT_EIGEN_NEAR(a, b, tol) \
  ASSERT_PRED_FORMAT3(test_utils::ExpectEigenNear, (a).eval(), (b).eval(), tol)

// TODO(gareth): This is all copy-pasta from other repos I have. Should put it in a
// common place.
namespace test_utils {

// Used by PRINT_MATRIX.
extern const Eigen::IOFormat kNumPyMatrixFmt;

// Compare two eigen matrices. Use EXPECT_EIGEN_NEAR()
template <typename Ta, typename Tb>
testing::AssertionResult ExpectEigenNear(const std::string& name_a, const std::string& name_b,
                                         const std::string& name_tol,
                                         const Eigen::MatrixBase<Ta>& a,
                                         const Eigen::MatrixBase<Tb>& b, double tolerance) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return testing::AssertionFailure()
           << "Dimensions of " << name_a << " and " << name_b << " do not match.";
  }
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      const double delta = a(i, j) - b(i, j);
      if (std::abs(delta) > tolerance || std::isnan(delta)) {
        const std::string index_str = "(" + std::to_string(i) + ", " + std::to_string(j) + ")";
        return testing::AssertionFailure()
               << "Matrix equality " << name_a << " == " << name_b << " failed because:\n"
               << name_a << index_str << " - " << name_b << index_str << " = " << delta << " > "
               << name_tol << "\nWhere " << name_a << " evaluates to:\n"
               << a << "\n and " << name_b << " evaluates to:\n"
               << b << "\n and " << name_tol << " evaluates to: " << tolerance << "\n";
      }
    }
  }
  return testing::AssertionSuccess();
}

/// Create a vector in the specified range.
/// Begins at `start` and increments by `step` until >= `end`.
std::vector<double> Range(double start, double end, double step);

/// Generate a random positive definite matrix.
Eigen::MatrixXd GenerateRandomPDMatrix(int size, int seed);

}  // namespace test_utils
