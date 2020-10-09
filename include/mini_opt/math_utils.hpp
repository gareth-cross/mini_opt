// Copyright 2020 Gareth Cross
#pragma once
#include <cmath>
#include <type_traits>

namespace mini_opt {

// Wrap into range of [-pi, pi].
template <typename Scalar>
Scalar ModPi(Scalar x) {
  static_assert(std::is_floating_point_v<Scalar>, "Must be float");
  constexpr auto pi = static_cast<Scalar>(M_PI);
  if (x < 0) {
    return -ModPi(-x);
  }
  const Scalar x_mod_2pi = std::fmod(x, 2 * pi);
  if (x_mod_2pi > pi) {
    return x_mod_2pi - 2 * pi;
  }
  return x_mod_2pi;
}

}  // namespace mini_opt
