// Copyright 2021 Gareth Cross
#pragma once

namespace mini_opt {

// Wrap into range of [-pi, pi].
template <typename Scalar>
Scalar ModPi(Scalar x);

// Convert degrees to radians.
template <typename Scalar>
Scalar DegToRad(Scalar deg);

// Convert radians to degrees.
template <typename Scalar>
Scalar RadToDeg(Scalar rad);

}  // namespace mini_opt
