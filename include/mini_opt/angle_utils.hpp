// Copyright 2020 Gareth Cross
#pragma once
#include <cmath>

namespace mini_opt {

// Map an angle to (-pi, pi].
template <typename Scalar>
Scalar mod_pi(Scalar angle) {
  static_assert(std::is_floating_point_v<Scalar>, "Must be floating point type");
  constexpr Scalar pi = static_cast<Scalar>(M_PI);
  constexpr Scalar two_pi = 2 * pi;
  angle = std::fmod(angle, two_pi);
  angle += (angle < 0) * two_pi;   //  Map to [0, 2pi]
  angle -= (angle > pi) * two_pi;  //  Map to (-pi, pi].
  return angle;
}

// Compute normalized difference between two angles, returning a delta in (-PI, PI].
template <typename Scalar>
Scalar compute_angle_delta(const Scalar a0, const Scalar a1) {
  return mod_pi<Scalar>(a1 - a0);
}

}  // namespace mini_opt
