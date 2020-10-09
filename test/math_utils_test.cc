// Copyright 2020 Gareth Cross
#include "mini_opt/math_utils.hpp"

#include "test_utils.hpp"

namespace mini_opt {

TEST(MathUtilsTest, TestModPi) {
  ASSERT_EQ(0, ModPi(0.));
  ASSERT_EQ(M_PI / 3, ModPi(M_PI / 3));
  ASSERT_EQ(M_PI / 4, ModPi(M_PI / 4));
  ASSERT_NEAR(3 * M_PI / 4, ModPi(3 * M_PI / 4), tol::kPico);
  ASSERT_NEAR(3 * M_PI / 4, ModPi(3 * M_PI / 4 + 2 * M_PI), tol::kPico);
  ASSERT_NEAR(3 * M_PI / 4, ModPi(3 * M_PI / 4 + 4 * M_PI), tol::kPico);
  ASSERT_NEAR(0.0, ModPi(2 * M_PI), tol::kPico);
  ASSERT_NEAR(0.0, ModPi(4 * M_PI), tol::kPico);
  ASSERT_NEAR(0.0, ModPi(12 * M_PI), tol::kPico);
  ASSERT_NEAR(M_PI, ModPi(3 * M_PI), tol::kPico);
  ASSERT_NEAR(M_PI, ModPi(7 * M_PI), tol::kPico);
  ASSERT_NEAR(-3 * M_PI / 4, ModPi(5 * M_PI + M_PI / 4), tol::kPico);
  ASSERT_NEAR(-M_PI / 2, ModPi(5 * M_PI + M_PI / 2), tol::kPico);
  ASSERT_EQ(-0.0, ModPi(-0.0));
  ASSERT_EQ(-M_PI / 6, ModPi(-M_PI / 6));
  ASSERT_EQ(-M_PI / 2, ModPi(-M_PI / 2));
  ASSERT_NEAR(-3 * M_PI / 4, ModPi(-3 * M_PI / 4 - 2 * M_PI), tol::kPico);
  ASSERT_EQ(-M_PI, ModPi(-M_PI));
  ASSERT_NEAR(0.0, ModPi(-4 * M_PI), tol::kPico);
}

}  // namespace mini_opt