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

  // Multiply by vector.
  math::Vector<double, 3> operator*(const math::Vector<double, 3>& v) const {
    return rotation * v + translation;
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
  math::Matrix<double, 3, Eigen::Dynamic> rotation_D_rotation;

  // Derivatives of `root_t_effector` wrt tangent of SO(3).
  // Dimension is [3, N * 3]
  math::Matrix<double, 3, Eigen::Dynamic> translation_D_rotation;

  // Derivatives of `root_t_effector` wrt translation vectors.
  // Dimension is [3, N * 3]
  math::Matrix<double, 3, Eigen::Dynamic> translation_D_translation;

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
void ComputeChain(const std::vector<Pose>& links, ChainComputationBuffer* c);

/**
 * Compute all the intermediate poses `start_T_i`, including `start_T_end`.
 */
std::vector<Pose> ComputeAllPoses(const ChainComputationBuffer& buffer);

/**
 * Store a single link in a chain of actuators. Each "actuator" executes a rotation
 * followed by a translation. Rotations are de-composed into euler angles, so that
 * as few as one of the angles may be optimized while the others are fixed.
 *
 * Normally we would want to optimize rotations on SO(3), but in this instance we
 * can choose the rotation frames so that the singularity is not an issue.
 */
struct ActuatorLink {
  // The pose of this link, in the original state (no parameter substitutions).
  Pose parent_T_child;

  // Mask of angles and translations that are active in the optimization.
  std::array<uint8_t, 6> active;

  // Euler angles from the decomposed rotation.
  // Factorized w/ order XYZ, only initialized if a rotation is active.
  math::Vector<double, 3> rotation_xyz;

  // Used to output derivative of SO(3) tangent wrt angle representation.
  using DerivativeBlock =
      Eigen::Block<math::Matrix<double, 3, Eigen::Dynamic>, 3, Eigen::Dynamic, true>;

  // Number of active parameters, between [0, 6].
  int ActiveCount() const;

  // Number of active rotational parameters, between [0, 3].
  int ActiveRotationCount() const;

  // Construct from Pose and mask.
  ActuatorLink(const Pose& pose, const std::array<uint8_t, 6>& mask);

  // Return pose representing this transform, given the parameters.
  // Derivative is only of the rotation part.
  Pose Compute(const math::Vector<double>& params, int position, DerivativeBlock J_out) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Store a chain of links, and compute the position of the effector as well as
// derivatives.
struct ActuatorChain {
 public:
  // Current poses in the chain.
  std::vector<ActuatorLink> links;

  // Total number of optimized variables in this chain (ie. angles that we can alter).
  int TotalActive() const;

  // Iterate over the chain and compute the effector pose.
  // Derivatives wrt all the active params are computed and cached locally.
  void Update(const math::Vector<double>& params);

  // Derivative of effector rotation wrt all parameters.
  // Update has to be called for this to be valid.
  const math::Matrix<double, 3, Eigen::Dynamic>& rotation_D_params() const {
    return rotation_D_params_;
  }

  // Derivative of effector translation wrt all parameters.
  // Update has to be called for this to be valid.
  const math::Matrix<double, 3, Eigen::Dynamic>& translation_D_params() const {
    return translation_D_params_;
  }

  // Access the position of the effector.
  math::Vector<double, 3> translation() const { return chain_buffer_.i_t_end.leftCols<1>(); }

  // Access the rotation of the effector.
  const math::Quaternion<double>& rotation() const { return chain_buffer_.i_R_end.front(); }

  // Access cached poses for all the links.
  const std::vector<Pose>& poses() const;

 private:
  // Poses cached from last computation.
  std::vector<Pose> pose_buffer_;

  // Cached derivative of effector rotation in SO(3) wrt parameters
  math::Matrix<double, 3, Eigen::Dynamic> rotation_D_params_;

  // Cached derivative of effector translation in SO(3) wrt parameters.
  math::Matrix<double, 3, Eigen::Dynamic> translation_D_params_;

  // Cached products while doing computations.
  ChainComputationBuffer chain_buffer_;

  // Cached parameters
  math::Vector<double> params_cached_;

  // Return true if the parameters have changed since the last time this was called.
  // Allows us to re-use intermediate values.
  bool ShouldUpdate(const math::Vector<double>& params) const;
};

}  // namespace mini_opt
