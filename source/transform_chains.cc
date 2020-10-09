// Copyright 2020 Gareth Cross
#include "mini_opt/transform_chains.hpp"

#include <numeric>

#include "mini_opt/assertions.hpp"

namespace mini_opt {
using namespace Eigen;

Pose ChainComputationBuffer::start_T_end() const {
  ASSERT(!i_R_end.empty() && i_t_end.cols() > 0);
  return Pose{i_R_end.front(), i_t_end.leftCols<1>()};
}

void ComputeChain(const std::vector<Pose>& links, ChainComputationBuffer* const c) {
  if (links.empty()) {
    // no iteration to do
    c->orientation_D_tangent.resize(3, 0);
    c->translation_D_tangent.resize(3, 0);
    c->i_R_end.clear();
    c->i_t_end.resize(3, 0);
    return;
  }
  const int N = static_cast<int>(links.size());

  // Compute backwards rotations (right to left)
  // Bucket `i` stores [i]_R_end.
  c->i_R_end.resize(N + 1);
  c->i_R_end[N].setIdentity();
  for (int i = N - 1; i >= 0; --i) {
    // rotation = previous_R_current
    c->i_R_end[i] = links[i].rotation * c->i_R_end[i + 1];
  }

  // Now compute translations.
  // We are multiplying the transforms, going right to left. The last element (i==0) is
  // root_t_effector.
  c->i_t_end.resize(3, N + 1);
  c->i_t_end.col(N).setZero();
  for (int i = N - 1; i >= 0; --i) {
    // rotation = previous_R_current
    c->i_t_end.col(i).noalias() = links[i].rotation * c->i_t_end.col(i + 1) + links[i].translation;
  }

  // Compute derivative of translation at the end wrt angle i.
  // d(0_t_N) / d(theta_[i]) = start_R_i * d(i_R_[i+1] * [i+1]_t_N) / d(theta_[i])
  //  = start_D_[i+1] * [-[i+1]_t_N]_x
  c->translation_D_tangent.resize(3, N * 3);
  math::Quaternion<double> start_R_i_plus_1 =
      math::Quaternion<double>::Identity();  //  forward rotation
  for (int i = 0; i < N - 1; ++i) {
    start_R_i_plus_1 *= links[i].rotation;
    c->translation_D_tangent.middleCols(i * 3, 3).noalias() =
        start_R_i_plus_1.matrix() * math::Skew3(-c->i_t_end.col(i + 1));
  }
  c->translation_D_tangent.rightCols<3>().setZero();  //  last angle does not affect translation

  // Compute derivative of rotation at the end wrt angle i.
  c->orientation_D_tangent.resize(3, N * 3);
  for (int i = 0; i < N - 1; ++i) {
    // d(root_R_eff)/d(theta_i) = n_R_[i+1]
    c->orientation_D_tangent.middleCols(i * 3, 3).noalias() =
        c->i_R_end[i + 1].conjugate().matrix();
  }
  c->orientation_D_tangent.rightCols<3>().setIdentity();
}

std::vector<Pose> ComputeAllPoses(const ChainComputationBuffer& buffer) {
  const Pose start_T_end{buffer.i_R_end.front(), buffer.i_t_end.leftCols<1>()};
  std::vector<Pose> start_T_i;
  start_T_i.reserve(buffer.i_R_end.size());
  for (std::size_t i = 0; i < buffer.i_R_end.size(); ++i) {
    start_T_i.push_back(start_T_end * Pose(buffer.i_R_end[i], buffer.i_t_end.col(i)).Inverse());
  }
  return start_T_i;
}

int ActuatorLink::ActiveCount() const {
  return static_cast<int>(active[0] > 0) + static_cast<int>(active[1] > 0) +
         static_cast<int>(active[2] > 0);
}

ActuatorLink::ActuatorLink(const Pose& pose, const std::array<uint8_t, 3>& mask)
    // Invert here, since this function assumes the order ZYX
    : rotation_xyz(-math::EulerAnglesFromSO3(pose.rotation.conjugate())),
      translation(pose.translation),
      active(mask) {}

// Return pose representing this transform, given the euler angles.
Pose ActuatorLink::Compute(const math::Vector<double>& angles, const int position,
                           math::Matrix<double, 3, Eigen::Dynamic>* const J_out) const {
  // Pull out just the angles we care about.
  math::Vector<double, 3> xyz_copy = rotation_xyz;
  for (int i = 0, angle_pos = position; i < 3; ++i) {
    if (active[i]) {
      xyz_copy[i] = angles[angle_pos++];
    }
  }
  // compute rotation and derivatives
  const math::SO3FromEulerAngles_<double> rot =
      math::SO3FromEulerAngles(xyz_copy, math::CompositionOrder::XYZ);
  if (J_out) {
    // copy out derivative blocks we'll need later
    for (int i = 0, angle_pos = position; i < 3; ++i) {
      if (active[i]) {
        J_out->col(angle_pos++) = rot.rotation_D_angles.col(i);
      }
    }
  }
  // Return a pose w/ our fixed translation.
  return {rot.q, translation};
}

void ActuatorChain::Update(const math::Vector<double>& angles) {
  if (!ShouldUpdate(angles)) {
    return;
  }
  angles_cached_ = angles;

  // Recompute.
  if (rotation_D_angles_.size() == 0) {
    // compute total active
    const int total_active = TotalActive();
    // allocate space
    rotation_D_angles_.resize(3, total_active);
  }
  ASSERT(angles.rows() == rotation_D_angles_.cols(),
         "Mismatch between # angles. angles.rows() = %i, expected = %i", angles.rows(),
         rotation_D_angles_.cols());

  // compute poses and rotational derivatives
  pose_buffer_.resize(links.size());
  for (std::size_t i = 0, position = 0; i < links.size(); ++i) {
    const ActuatorLink& link = links[i];
    const int num_active = link.ActiveCount();
    pose_buffer_[i] = link.Compute(angles, position, &rotation_D_angles_);
    position += num_active;
  }

  // linearize
  ComputeChain(pose_buffer_, &chain_buffer_);
}

// Return true if the angles have changed since the last time this was called.
// Allows us to re-use intermediate values.
bool ActuatorChain::ShouldUpdate(const math::Vector<double>& angles) const {
  if (angles_cached_.rows() != angles.rows()) {
    return true;
  }
  for (int i = 0; i < angles.rows(); ++i) {
    if (std::abs(angles_cached_[i] - angles[i]) > 1.0e-9) {
      return true;
    }
  }
  return false;
}

// True if only the Z component is active.
// In this case, the chain ruling can be avoided altogether.
static bool IsOnlyZ(const ActuatorLink& l) { return !l.active[0] && !l.active[1] && l.active[2]; }

// Compute rotation and translation of the effector.
math::Vector<double, 3> ActuatorChain::ComputeEffectorPosition(
    const math::Vector<double>& angles, math::Matrix<double, 3, Eigen::Dynamic>* const J) {
  Update(angles);

  // chain rule
  if (J) {
    ASSERT(J->cols() == TotalActive());
    for (int i = 0, position = 0; i < static_cast<int>(links.size()); ++i) {
      const ActuatorLink& link = links[i];
      const int active_count = link.ActiveCount();
      if (active_count == 0) {
        continue;
      }
      // Chain rule w/ the angle representation.
      if (IsOnlyZ(link)) {
        J->middleCols<1>(position) = chain_buffer_.translation_D_tangent.middleCols<1>(i * 3 + 2);
      } else {
        J->middleCols(position, active_count).noalias() =
            chain_buffer_.translation_D_tangent.middleCols<3>(i * 3) *
            rotation_D_angles_.middleCols(position, active_count);
      }
      position += active_count;
    }
  }
  return chain_buffer_.i_t_end.leftCols<1>();
}

int ActuatorChain::TotalActive() const {
  return std::accumulate(links.begin(), links.end(), 0,
                         [](const int t, const ActuatorLink& l) { return t + l.ActiveCount(); });
}

}  // namespace mini_opt
