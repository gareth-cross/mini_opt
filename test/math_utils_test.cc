// Copyright 2021 Gareth Cross
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

TEST(MathUtilsTest, TestRadConversion) {
  ASSERT_EQ(0.0, RadToDeg(0.));
  ASSERT_NEAR(45.0, RadToDeg(M_PI / 4), tol::kPico);
  ASSERT_NEAR(90.0, RadToDeg(M_PI / 2), tol::kPico);
  ASSERT_NEAR(180.0, RadToDeg(M_PI), tol::kPico);
  ASSERT_EQ(0.0, DegToRad(0.));
  ASSERT_NEAR(M_PI / 3, DegToRad(60.), tol::kPico);
  ASSERT_NEAR(5 * M_PI / 3, DegToRad(5 * 60.), tol::kPico);
  for (double x = -360.0; x <= 360.0; x += 1.0) {
    ASSERT_NEAR(x, RadToDeg(DegToRad(x)), tol::kPico);
  }
}

}  // namespace mini_opt