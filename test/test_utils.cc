// Copyright 2020 Gareth Cross
#include "test_utils.hpp"

namespace test_utils {

std::vector<double> Range(double start, double end, double step) {
  std::vector<double> values;
  while (start < end) {
    values.push_back(start);
    start += step;
  }
  return values;
}

}  // namespace test_utils
