// Copyright 2020 Gareth Cross
#include "transform_chains.hpp"

#include "assertions.hpp"

namespace mini_opt {

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

}  // namespace mini_opt
