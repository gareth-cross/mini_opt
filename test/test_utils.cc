// Copyright 2021 Gareth Cross
#include "test_utils.hpp"

#include <random>

namespace test_utils {

const Eigen::IOFormat kNumPyMatrixFmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

std::vector<double> Range(double start, double end, double step) {
  std::vector<double> values;
  while (start < end) {
    values.push_back(start);
    start += step;
  }
  return values;
}

Eigen::MatrixXd GenerateRandomPDMatrix(int size, int seed) {
  std::default_random_engine engine{static_cast<unsigned int>(seed)};
  std::uniform_real_distribution<double> dist{-1, 1};
  Eigen::MatrixXd A(size, size);
  Eigen::VectorXd u(size);
  A.setZero();

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      u[j] = dist(engine);
    }
    A.selfadjointView<Eigen::Upper>().rankUpdate(u.transpose());
  }
  A.triangularView<Eigen::StrictlyUpper>() = A.triangularView<Eigen::StrictlyLower>().transpose();
  return A;
}

}  // namespace test_utils
