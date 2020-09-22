// Copyright 2020 Gareth Cross
#include "transform_chains.hpp"

#include <numeric>

#include "assertions.hpp"

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
  // TODO(gareth): Dumb that we have to copy the fixed translation always...
  return Pose{rot.q, translation};
}

void ActuatorLink::FillJacobian(
    const Eigen::Block<const Eigen::Matrix<double, 3, Eigen::Dynamic>, 3, 3, true>&
        output_D_tangent,
    const Eigen::Block<const Eigen::Matrix<double, 3, Eigen::Dynamic>, 3, Eigen::Dynamic, true>&
        tangent_D_angles,
    Eigen::Block<Eigen::Matrix<double, 3, Eigen::Dynamic>, 3, Eigen::Dynamic, true> J_out) const {
  // Output buffer should be correct size already.
  ASSERT(J_out.cols() == ActiveCount());
  if (!active[0] && !active[1] && active[2]) {
    // Fast path for common case, we know dz = [0, 0, 1]
    J_out = output_D_tangent.rightCols<1>();
  } else {
    J_out.noalias() = output_D_tangent * tangent_D_angles;
  }
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
    ASSERT(angles.rows() == total_active);
    // allocate space
    rotation_D_angles_.resize(3, total_active);
  }

  // compute poses and rotational derivatives
  pose_buffer_.resize(links.size());
  for (int i = 0, position = 0; i < links.size(); ++i) {
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
    if (std::abs(angles_cached_[i] - angles[i]) > 1.0e-6) {
      return true;
    }
  }
  return false;
}

// Compute rotation and translation of the effector.
math::Vector<double, 3> ActuatorChain::ComputeEffectorPosition(
    const math::Vector<double>& angles, math::Matrix<double, 3, Eigen::Dynamic>* const J) {
  Update(angles);

  // chain rule
  if (J) {
    ASSERT(J->cols() == TotalActive());
    for (int i = 0, position = 0; i < links.size(); ++i) {
      const ActuatorLink& link = links[i];
      const int active_count = link.ActiveCount();
      if (active_count == 0) {
        continue;
      }

      // Need const-references so we can get const-blocks.
      const ChainComputationBuffer& const_buffer = chain_buffer_;
      const math::Matrix<double, 3, Dynamic>& const_rotation_D_angles = rotation_D_angles_;

      // Chain rule w/ the angle representation.
      link.FillJacobian(const_buffer.translation_D_tangent.middleCols<3>(i * 3),
                        const_rotation_D_angles.middleCols(position, active_count),
                        J->middleCols(position, active_count));

      position += link.ActiveCount();
    }
  }
  return chain_buffer_.start_T_end().translation;
}

int ActuatorChain::TotalActive() const {
  return std::accumulate(links.begin(), links.end(), 0,
                         [](const int t, const ActuatorLink& l) { return t + l.ActiveCount(); });
}

}  // namespace mini_opt