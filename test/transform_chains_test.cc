// Copyright 2020 Gareth Cross
#include "mini_opt/transform_chains.hpp"

#include "geometry_utils/numerical_derivative.hpp"
#include "test_utils.hpp"

namespace mini_opt {
using namespace Eigen;

TEST(ChainComputationBufferTest, TestComputeChain) {
  // create some links
  // clang-format off
  const std::vector<Pose> links = {
    {math::QuaternionExp(Vector3d{-0.5, 0.5, 0.3}), {1.0, 0.5, 2.0}}, 
    {math::QuaternionExp(Vector3d{0.8, 0.5, 1.2}), {0.5, 0.75, -0.5}},
    {math::QuaternionExp(Vector3d{1.5, -0.2, 0.0}), {1.2, -0.5, 0.1}},
    {math::QuaternionExp(Vector3d{0.2, -0.1, 0.3}), {0.1, -0.1, 0.2}}
  };
  // clang-format on

  const auto translation_lambda = [&](const VectorXd& angles) -> Vector3d {
    std::vector<Pose> links_copied = links;
    for (int i = 0; i < angles.rows() / 3; ++i) {
      links_copied[i].rotation *= math::QuaternionExp(angles.segment(i * 3, 3));
    }
    ChainComputationBuffer c{};
    ComputeChain(links_copied, &c);
    return c.i_t_end.leftCols<1>();
  };

  const auto rotation_lambda = [&](const VectorXd& angles) -> Quaterniond {
    std::vector<Pose> links_copied = links;
    for (int i = 0; i < angles.rows() / 3; ++i) {
      links_copied[i].rotation *= math::QuaternionExp(angles.segment(i * 3, 3));
    }
    ChainComputationBuffer c{};
    ComputeChain(links_copied, &c);
    return c.i_R_end.front();
  };

  // compute numerically
  const Matrix<double, 3, Dynamic> J_trans_numerical =
      math::NumericalJacobian(VectorXd::Zero(links.size() * 3), translation_lambda);
  const Matrix<double, 3, Dynamic> J_trans_rotational =
      math::NumericalJacobian(VectorXd::Zero(links.size() * 3), rotation_lambda);

  // check against anlytical
  ChainComputationBuffer c{};
  ComputeChain(links, &c);

  ASSERT_EIGEN_NEAR(J_trans_numerical, c.translation_D_tangent, tol::kNano)
      << "Numerical:\n"
      << J_trans_numerical.format(test_utils::kNumPyMatrixFmt) << "\n"
      << "Analytical:\n"
      << c.translation_D_tangent.format(test_utils::kNumPyMatrixFmt);

  ASSERT_EIGEN_NEAR(J_trans_rotational, c.orientation_D_tangent, tol::kNano)
      << "Numerical:\n"
      << J_trans_rotational.format(test_utils::kNumPyMatrixFmt) << "\n"
      << "Analytical:\n"
      << c.orientation_D_tangent.format(test_utils::kNumPyMatrixFmt);

  // pull out poses
  ASSERT_EQ(links.size() + 1, c.i_R_end.size());
  ASSERT_EQ(c.i_R_end.size(), static_cast<size_t>(c.i_t_end.cols()));
  const Pose start_T_end{c.i_R_end.front(), c.i_t_end.leftCols<1>()};

  // note we are iterating over inverted poses `i_T_end`, so i = 0 is in fact the full transform
  Pose start_T_current{};
  const std::vector<Pose> start_T_i = ComputeAllPoses(c);
  for (int i = 0; i < c.i_R_end.size(); ++i) {
    // compare poses
    ASSERT_EIGEN_NEAR(start_T_current.translation, start_T_i[i].translation, tol::kNano)
        << "i = " << i;
    ASSERT_EIGEN_NEAR(start_T_current.rotation.matrix(), start_T_i[i].rotation.matrix(), tol::kNano)
        << "i = " << i;
    // advance
    start_T_current = start_T_current * links[i];
  }
}

TEST(ActuatorLinkTest, TestComputePose) {
  const Pose pose{math::QuaternionExp(Vector3d{-0.3, 0.5, 0.4}), Vector3d(0.4, -0.2, 1.2)};

  const std::array<uint8_t, 3> mask = {{true, false, true}};
  ActuatorLink link{pose, mask};

  // At least for these angles, this is true.
  ASSERT_EIGEN_NEAR(
      math::SO3FromEulerAngles(link.rotation_xyz, math::CompositionOrder::XYZ).q.matrix(),
      pose.rotation.matrix(), tol::kPico);

  // compute analytically, place it somewhere in this matrix
  math::Matrix<double, 3, Eigen::Dynamic> J_out;
  J_out.resize(3, 10);
  J_out.setZero();

  const Array3d mask_float =
      Eigen::Map<const Eigen::Matrix<uint8_t, 3, 1>>(mask.data()).cast<double>();

  // some input angles
  const Vector3d input_angles{0.2, 0.1, 0.35};

  // what the result should amount to
  const Vector3d combined_angles =
      mask_float * input_angles.array() + (1 - mask_float) * link.rotation_xyz.array();

  // put the input angles in the right space in the buffer
  VectorXd input_angles_dynamic(10);
  input_angles_dynamic.setZero();
  input_angles_dynamic[5] = input_angles[0];  //  skip disabled angle
  input_angles_dynamic[6] = input_angles[2];

  const Pose computed_pose = link.Compute(input_angles_dynamic, 5, &J_out);
  ASSERT_EIGEN_NEAR(computed_pose.translation, pose.translation, tol::kPico);
  ASSERT_EIGEN_NEAR(
      computed_pose.rotation.matrix(),
      math::SO3FromEulerAngles(combined_angles, math::CompositionOrder::XYZ).q.matrix(),
      tol::kPico);

  // derivative should also respect the active flag
  const auto lambda = [&](const VectorXd& angles) {
    return link.Compute(angles, 5, nullptr).rotation;
  };
  const Matrix<double, 3, Dynamic> J_numerical =
      math::NumericalJacobian(input_angles_dynamic, lambda);
  ASSERT_EIGEN_NEAR(J_numerical, J_out, tol::kPico);
}

TEST(ActuatorChainTest, TestComputeEffectorPosition) {
  // create some links
  // clang-format off
  const std::vector<Pose> links = {
    {math::QuaternionExp(Vector3d{-0.5, 0.5, 0.3}), {1.0, 0.5, 2.0}}, 
    {math::QuaternionExp(Vector3d{0.8, 0.5, 1.2}), {0.5, 0.75, -0.5}},
    {math::QuaternionExp(Vector3d{1.5, -0.2, 0.0}), {1.2, -0.5, 0.1}},
    {math::QuaternionExp(Vector3d{0.2, -0.1, 0.3}), {0.1, -0.1, 0.2}}
  };
  // clang-format on

  int mask_index = 0;
  ActuatorChain chain{};
  for (const Pose& pose : links) {
    std::array<uint8_t, 3> mask;
    mask.fill(0);
    mask[mask_index] = true;
    mask_index = (mask_index + 1) % 3;
    chain.links.emplace_back(pose, mask);
  }

  const int total_active = chain.TotalActive();
  ASSERT_EQ(4, total_active);

  const VectorXd angles = Vector4d(-0.3, 0.2, 0.5, 0.1);
  Matrix<double, 3, Dynamic> J_analytical;
  J_analytical.resize(3, angles.size());
  chain.ComputeEffectorPosition(angles, &J_analytical);

  const auto J_numerical = math::NumericalJacobian(
      angles, [&](const VectorXd& angles) { return chain.ComputeEffectorPosition(angles); });
  ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kPico);
}

}  // namespace mini_opt
