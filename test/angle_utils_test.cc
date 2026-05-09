// Copyright 2020 Gareth Cross
#include "mini_opt/angle_utils.hpp"

#include "test_utils.hpp"

namespace mini_opt {

TEST(AngleUtilsTest, TestModPi) {
  // Test multiples of 2-pi.
  for (int i = -3; i <= 3; ++i) {
    const auto offset = 2 * M_PI * i;
    for (auto angle : {0., M_PI / 6, M_PI / 4, M_PI / 2}) {
      ASSERT_NEAR(angle, mod_pi(angle + offset), tol::kNano);
    }
    ASSERT_NEAR(M_PI, mod_pi(M_PI + offset), tol::kNano);
    ASSERT_NEAR(M_PI, mod_pi(-M_PI + offset), tol::kNano);
  }
}

// Test computing the difference between two angles.
TEST(AngleUtilsTest, TestComputeAngleDelta) {
  // Test multiples of 2-pi.
  for (int i = -3; i <= 3; ++i) {
    const auto offset = 2 * M_PI * i;
    ASSERT_NEAR(0, compute_angle_delta(0., 0. + offset), tol::kNano);
    ASSERT_NEAR(0, compute_angle_delta(2 * M_PI, 0. + offset), tol::kNano);
    ASSERT_NEAR(0, compute_angle_delta(M_PI, -M_PI + offset), tol::kNano);
    ASSERT_NEAR(0, compute_angle_delta(-M_PI, M_PI + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 6, compute_angle_delta(M_PI / 6, M_PI / 3 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 3, compute_angle_delta(-M_PI / 6, M_PI / 6 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 2, compute_angle_delta(M_PI / 4, 3 * M_PI / 4 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 2, compute_angle_delta(3 * M_PI / 4, 5 * M_PI / 4 + offset), tol::kNano);
    ASSERT_NEAR(-M_PI / 2, compute_angle_delta(5 * M_PI / 4, 3 * M_PI / 4 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 3, compute_angle_delta(5 * M_PI / 6 - offset, 7 * M_PI / 6 + offset),
                tol::kNano);
  }
}

}  // namespace mini_opt
