// Copyright 2021 Gareth Cross
#pragma once
#include <string_view>

#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <Eigen/Core>

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
#define PRINT(x)                                   \
  {                                                \
    fmt::print("{} = {}\n", #x, fmt::streamed(x)); \
    std::fflush(nullptr);                          \
  }

// Print a matrix.
#define PRINT_MATRIX(x) \
  { fmt::print("{}:\n{}\n", #x, fmt::streamed((x).eval().format(test_utils::kNumPyMatrixFmt))); }

// Define a test on a class.
#define TEST_FIXTURE(object, function) \
  TEST_F(object, function) { function(); }

// Macro to compare eigen matrices and print a nice error.
#define EXPECT_EIGEN_NEAR(a, b, tol) \
  EXPECT_PRED_FORMAT3(test_utils::expect_eigen_near, (a).eval(), (b).eval(), tol)
#define ASSERT_EIGEN_NEAR(a, b, tol) \
  ASSERT_PRED_FORMAT3(test_utils::expect_eigen_near, (a).eval(), (b).eval(), tol)

// TODO(gareth): This is all copy-pasta from other repos I have. Should put it in a
// common place.
namespace test_utils {

// Used by PRINT_MATRIX.
extern const Eigen::IOFormat kNumPyMatrixFmt;

// Compare two eigen matrices. Use EXPECT_EIGEN_NEAR()
template <typename Ta, typename Tb>
testing::AssertionResult expect_eigen_near(const std::string_view name_a,
                                           const std::string_view name_b,
                                           const std::string_view name_tol, const Ta& a,
                                           const Tb& b, const double tolerance) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return testing::AssertionFailure()
           << fmt::format("Dimensions of {} [{}, {}] and {} [{}, {}] do not match.", name_a,
                          a.rows(), a.cols(), name_b, b.rows(), b.cols());
  }
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      if (const double delta = a(i, j) - b(i, j);
          std::abs(delta) > tolerance || std::isnan(delta)) {
        return testing::AssertionFailure() << fmt::format(
                   "Matrix equality {a} == {b} failed because:\n"
                   "({a})({i}, {j}) - ({b})({i}, {j}) = {delta} > {tol}\n"
                   "Where {a} evaluates to:\n{a_eval}\nAnd {b} evaluates to:\n{b_eval}\n"
                   "And {name_tol} evaluates to: {tol}\n",
                   fmt::arg("a", name_a), fmt::arg("b", name_b), fmt::arg("i", i), fmt::arg("j", j),
                   fmt::arg("a_eval", fmt::streamed(a)), fmt::arg("b_eval", fmt::streamed(b)),
                   fmt::arg("name_tol", name_tol), fmt::arg("tol", tolerance),
                   fmt::arg("delta", delta));
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
