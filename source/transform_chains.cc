// Copyright 2020 Gareth Cross
#include "mini_opt/transform_chains.hpp"

#include <numeric>

#include "mini_opt/assertions.hpp"

namespace mini_opt {
using namespace Eigen;

math::Matrix<double, 4, 4> Pose::ToMatrix() const {
  return (math::Matrix<double, 4, 4>() << rotation.matrix(), translation, 0, 0, 0, 1).finished();
}

Pose ChainComputationBuffer::start_T_end() const {
  ASSERT(!i_R_end.empty() && i_t_end.cols() > 0);
  return Pose{i_R_end.front(), i_t_end.leftCols<1>()};
}

void ComputeChain(const std::vector<Pose>& links, ChainComputationBuffer* const c) {
  if (links.empty()) {
    // no iteration to do
    c->rotation_D_rotation.resize(3, 0);
    c->translation_D_rotation.resize(3, 0);
    c->translation_D_translation.resize(3, 0);
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

  // Compute derivative of translation at the end wrt intermediate translations.
  // d(0_t_N) / d(i_t_[i+1]) = 0_R_[i]
  c->translation_D_translation.resize(3, N * 3);
  math::Quaternion<double> start_R_i = math::Quaternion<double>::Identity();  // forward rotation
  for (int i = 0; i < N; ++i) {
    c->translation_D_translation.middleCols<3>(i * 3) = start_R_i.matrix();
    start_R_i *= links[i].rotation;
  }

  // Compute derivative of translation at the end wrt angle i.
  // d(0_t_N) / d(theta_[i]) = start_R_i * d(i_R_[i+1] * [i+1]_t_N) / d(theta_[i])
  //  = start_D_[i+1] * [-[i+1]_t_N]_x
  c->translation_D_rotation.resize(3, N * 3);
  for (int i = 0; i < N - 1; ++i) {
    // We already computed this as part of translation_D_translation.
    const auto start_R_i_plus_1 = c->translation_D_translation.middleCols<3>((i + 1) * 3);
    c->translation_D_rotation.middleCols<3>(i * 3).noalias() =
        start_R_i_plus_1 * math::Skew3(-c->i_t_end.col(i + 1));
  }
  c->translation_D_rotation.rightCols<3>().setZero();  //  last angle does not affect translation

  // Compute derivative of rotation at the end wrt angle i.
  c->rotation_D_rotation.resize(3, N * 3);
  for (int i = 0; i < N - 1; ++i) {
    // d(root_R_eff)/d(theta_i) = n_R_[i+1]
    c->rotation_D_rotation.middleCols<3>(i * 3).noalias() = c->i_R_end[i + 1].conjugate().matrix();
  }
  c->rotation_D_rotation.rightCols<3>().setIdentity();
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
  return static_cast<int>(
      std::count_if(active.begin(), active.end(), [](uint8_t x) { return x > 0; }));
}

int ActuatorLink::ActiveRotationCount() const {
  return static_cast<int>(active[0] > 0) + static_cast<int>(active[1] > 0) +
         static_cast<int>(active[2] > 0);
}

ActuatorLink::ActuatorLink(const Pose& pose, const std::array<uint8_t, 6>& mask)
    : parent_T_child(pose), active(mask) {
  // Check that decomposition worked to catch issues early.
  if (ActiveRotationCount() > 0) {
    // TODO(gareth): Do something that generalizes better than just plain euler angles.
    // Invert here, since this function assumes the order ZYX
    rotation_xyz = -math::EulerAnglesFromSO3(pose.rotation.conjugate());

    const math::Matrix<double, 3, 3> R_delta =
        (math::SO3FromEulerAngles(rotation_xyz, math::CompositionOrder::XYZ).q.matrix() -
         pose.rotation.matrix())
            .cwiseAbs();
    ASSERT((R_delta.array() < 1.0e-5).all(), "Euler angle decomposition failed");
  }
}

// Return pose representing this transform, given the euler angles.
// TODO(gareth): Not a fan of this logic. It might be cleaner to just create a lambda in the
// constructor that does the correct logic, and specialize it depending on which set of flags
// are active? Alternatively, could just use polymorphism on each active link and make them
// all specialized like that.
Pose ActuatorLink::Compute(const math::Vector<double>& params, const int position,
                           DerivativeBlock J_out) const {
  ASSERT(J_out.cols() == ActiveRotationCount(), "Wrong number of columns in output jacobian");
  if (ActiveRotationCount() == 0) {
    math::Vector<double, 3> translation_xyz = parent_T_child.translation;
    for (int i = 0, param_index = position; i < 3; ++i) {
      if (active[static_cast<std::size_t>(i) + 3]) {
        ASSERT(param_index < params.rows());
        translation_xyz[i] = params[param_index++];
      }
    }
    return Pose{parent_T_child.rotation, translation_xyz};
  }
  // Pull out just the angles and translations we care about.
  math::Vector<double, 6> params_updated;
  params_updated.head<3>() = rotation_xyz;
  params_updated.tail<3>() = parent_T_child.translation;
  for (int i = 0, param_index = position; i < 6; ++i) {
    if (active[i]) {
      params_updated[i] = params[param_index++];
    }
  }
  // compute rotation and derivatives
  const math::SO3FromEulerAngles_<double> rot =
      math::SO3FromEulerAngles(params_updated.head<3>(), math::CompositionOrder::XYZ);
  // copy out derivative blocks we'll need later
  for (int axis = 0, output_index = 0; axis < 3; ++axis) {
    if (active[axis]) {
      J_out.col(output_index++) = rot.rotation_D_angles.col(axis);
    }
  }
  // Return a pose.
  return Pose(rot.q, params_updated.tail<3>());
}

// We use rotation_D_params_ twice here. We fill it once w/ the derivative of
// each link's SO(3) tangent space wrt the underlying angles. Then we write to
// it by multiplying on the left with effector_rot_R_link_rot.
// TODO(gareth): Not the biggest fan of all the indexing involved in having the `active` set
// of params. Would like something cleaner for that.
void ActuatorChain::Update(const math::Vector<double>& params) {
  if (!ShouldUpdate(params)) {
    return;
  }
  params_cached_ = params;

  if (translation_D_params_.size() == 0) {
    // compute total active
    const int total_active = TotalActive();
    ASSERT(params.rows() == total_active,
           "Wrong number of params passed. Expected = {}, actual = {}", total_active,
           params.rows());
    rotation_D_params_.resize(3, total_active);
    rotation_D_params_.setZero();
    translation_D_params_.resize(3, total_active);
    translation_D_params_.setZero();
  }

  // Dimensions cannot change after first call.
  ASSERT(params.rows() == rotation_D_params_.cols(),
         "Mismatch between # params. params.rows() = %i, expected = %i", params.rows(),
         rotation_D_params_.cols());

  // Compute poses and rotational derivatives.
  pose_buffer_.resize(links.size());
  for (std::size_t i = 0, position = 0; i < links.size(); ++i) {
    const ActuatorLink& link = links[i];
    const int num_active = link.ActiveCount();
    const int num_rotation_active = link.ActiveRotationCount();
    pose_buffer_[i] = link.Compute(params, static_cast<int>(position),
                                   rotation_D_params_.middleCols(position, num_rotation_active));
    position += num_active;
  }

  // Evaluate the full chain, include derivatives wrt intermediate poses.
  ComputeChain(pose_buffer_, &chain_buffer_);

  // Chain rule to get params wrt just the active set of parameters.
  for (int i = 0, position = 0; i < static_cast<int>(links.size()); ++i) {
    const ActuatorLink& link = links[i];
    const int active_count = link.ActiveCount();
    const int num_rotation_active = link.ActiveRotationCount();
    if (active_count == 0) {
      // Pose is fixed, skip it.
      continue;
    }

    // Chain rule w/ the angle representation.
    // The right half of rot_D_rot will be zero, but this is likely ok on the grounds that
    // translation parameters should be less common than rotational ones, so the wasted space
    // and work is not that bad.
    ASSERT(position + active_count <= rotation_D_params_.cols());

    auto rot_D_angles = rotation_D_params_.middleCols(
        position, num_rotation_active);  //  wrt just the rotation part
    auto trans_D_params = translation_D_params_.middleCols(position, active_count);

    // first use the rotation part on the right, to get the derivative of translation wrt rotation
    trans_D_params.leftCols(num_rotation_active).noalias() =
        chain_buffer_.translation_D_rotation.middleCols<3>(i * 3) * rot_D_angles;

    // then multiply on the left with the derivative of rotation wrt the
    // TODO(gareth): Optimize for the case where active = [0, 0, 1]
    const auto effector_rot_D_rot = chain_buffer_.rotation_D_rotation.middleCols<3>(i * 3);
    for (int axis = 0; axis < num_rotation_active; ++axis) {
      rot_D_angles.col(axis) = (effector_rot_D_rot * rot_D_angles.col(axis)).eval();
    }

    const int num_translation_active = active_count - num_rotation_active;
    if (num_translation_active > 0) {
      // Here we need to copy the active cols, no chain ruling required.
      for (int axis = 0, output_index = 0; axis < 3; ++axis) {
        if (link.active[axis + 3]) {
          trans_D_params.col(num_rotation_active + output_index) =
              chain_buffer_.translation_D_translation.col(i * 3 + axis);
          ++output_index;
        }
      }
    }
    position += active_count;
  }
}

// Return true if the parameters have changed since the last time this was called.
// Allows us to re-use intermediate values.
bool ActuatorChain::ShouldUpdate(const math::Vector<double>& params) const {
  if (params_cached_.rows() != params.rows()) {
    return true;
  }
  for (int i = 0; i < params.rows(); ++i) {
    if (std::abs(params_cached_[i] - params[i]) > 1.0e-9) {
      return true;
    }
  }
  return false;
}

int ActuatorChain::TotalActive() const {
  return std::accumulate(links.begin(), links.end(), 0,
                         [](const int t, const ActuatorLink& l) { return t + l.ActiveCount(); });
}

// TODO(gareth): Assert that this has been filled out?
const std::vector<Pose>& ActuatorChain::poses() const { return pose_buffer_; }

}  // namespace mini_opt
