// Copyright 2020 Gareth Cross
#include <array>
#include <vector>

#include "geometry_utils/so3.hpp"

// Code for representing and computing chains of transforms, such as you might
// find in a robotic actuator or skeleton.
namespace mini_opt {

/*
 * A simple 3D Pose type: rotation + translation.
 *
 * TODO(gareth): Template on double vs. float.
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

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Aligned std::vector for Eigen.
template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

/**
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

/**
 * Fill `ChainComputationBuffer` instance by composing all the poses in a chain, computing
 * the derivatives in the process.
 *
 * If `links` is empty, we clear the buffer.
 */
void ComputeChain(const std::vector<Pose>& links, ChainComputationBuffer* const c);

/**
 *
 */
struct ActuatorLink {
  // Euler angles from the decomposed rotation.
  // Factorized w/ order XYZ.
  math::Vector<double, 3> rotation_xyz;

  // Translational part.
  math::Vector<double, 3> translation;

  // Mask of angles that are active in the optimization.
  std::array<uint8_t, 3> active;

  // Number of active angles.
  int ActiveCount() const {
    return static_cast<int>(
        std::count_if(active.begin(), active.end(), [](auto b) { return b > 0; }));
  }

  // Construct from Pose and mask.
  ActuatorLink(const Pose& pose, const std::array<uint8_t, 3>& mask)
      : rotation_xyz(math::EulerAnglesFromSO3(pose.rotation.conjugate())),
        translation(pose.translation),
        active(mask) {}

  // Return pose representing this transform, given the euler angles.
  Pose Compute(const math::Vector<double>& angles, const int position,
               math::Matrix<double, 3, Eigen::Dynamic>* const J_out) const;

  void FillJacobian(
      const Eigen::Block<const Eigen::Matrix<double, 3, Eigen::Dynamic>, 3, 3, true>&
          output_D_tangent,
      const Eigen::Block<const Eigen::Matrix<double, 3, Eigen::Dynamic>, 3, Eigen::Dynamic, true>&
          tangent_D_angles,
      Eigen::Block<Eigen::Matrix<double, 3, Eigen::Dynamic>, 3, Eigen::Dynamic, true> J_out) const;
};

// TODO(gareth): comments...
struct ActuatorChain {
  // Current poses in the chain.
  std::vector<ActuatorLink> links;

  // private:
  // Poses.
  std::vector<Pose> pose_buffer_;

  // Buffer of rotations derivatives.
  math::Matrix<double, 3, Eigen::Dynamic> rotation_D_angles_;

  // Cached products while doing computations.
  ChainComputationBuffer chain_buffer_;

  // Cached angles
  math::Vector<double> angles_cached_;

  // Iterate over the chain and compute the effector pose.
  // Derivatives wrt all the input angles are computed and cached locally.
  void Update(const math::Vector<double>& angles);

  // Return true if the angles have changed since the last time this was called.
  // Allows us to re-use intermediate values.
  bool ShouldUpdate(const math::Vector<double>& angles) const;

 public:
  // Compute rotation and translation of the effector.
  math::Vector<double, 3> ComputeEffectorPosition(
      const math::Vector<double>& angles,
      math::Matrix<double, 3, Eigen::Dynamic>* const J = nullptr);

  int TotalActive() const;
};

}  // namespace mini_opt
