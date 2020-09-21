// Copyright 2020 Gareth Cross
#include <vector>

#include "so3.hpp"

// Code for representing and computing chains of transforms, such as you might
// find in a robotic actuator or skeleton.
namespace mini_opt {

/*
 * A simple 3D Pose type: rotation + translation.
 */
struct Pose {
  // Construct w/ rotation and translation.
  Pose(const math::Quaternion<double>& q, const math::Vector<double, 3>& t)
      : rotation(q), translation(t) {}

  Pose()
      : rotation(math::Quaternion<double>::Identity()),
        translation(math::Vector<double, 3>::Zero()) {}

  // Rotation.
  math::Quaternion<double> rotation;

  // Translation.
  math::Vector<double, 3> translation;

  // Multiply together.
  Pose operator*(const Pose& other) const {
    return Pose(rotation * other.rotation, translation + rotation * other.translation);
  }

  // Invert the pose. If this is A, returns B such that A*B = Identity
  Pose Inverse() const {
    const math::Quaternion<double> R_inv = rotation.inverse();
    return Pose(R_inv, R_inv * -translation);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

// Aligned std::vector for Eigen.
template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

/*
 * Store intermediate quantities that are generated when iterating down a sequence of
 * joints. This is used to compute the transform of an effector wrt the root of the
 * chain, as well as the derivatives of its orientation and translation wrt the intermediate
 * joints.
 *
 * Derivatives are stored in the right-tangent of SO(3). They are later converted to be
 * with respect to the joint angles themselves.
 *
 * The convention used here is that joint angles `i` express the rotation from frame `i`
 * to frame `i + 1`. If there are `N` transforms in link, there will be `N+1` frames including
 * the start (i = 0) and end (i=N+1).
 */
struct ChainComputationBuffer {
  // Derivatives of `root_R_effector` wrt tangent of SO(3).
  // Dimension is [3, N * 3]
  math::Matrix<double, 3, Eigen::Dynamic> orientation_D_tangent;

  // Derivatives of `root_t_effector` wrt tangent of SO(3).
  // Dimension is [3, N * 3]
  math::Matrix<double, 3, Eigen::Dynamic> translation_D_tangent;

  // Buffer for the end frame wrt all intermediate frames.
  // Dimension is N + 1
  AlignedVector<math::Quaternion<double>> i_R_end;

  // Buffer for translation of the end frame wrt all intermediate frames.
  // Dimension is N + 1
  math::Matrix<double, 3, Eigen::Dynamic> i_t_end;

  // Return the transform expressing the `end` frame wrt the `start` (i = 0) frame.
  Pose start_T_end() const;
};

/*
 * Fill `ChainComputationBuffer` instance by composing all the poses in a chain, computing
 * the derivatives in the process.
 *
 * If `links` is empty, we clear the buffer.
 */
void ComputeChain(const std::vector<Pose>& links, ChainComputationBuffer* const c);

}  // namespace mini_opt
