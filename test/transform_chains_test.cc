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

  const auto translation_wrt_rot = [&](const VectorXd& angles) -> Vector3d {
    std::vector<Pose> links_copied = links;
    for (int i = 0; i < angles.rows() / 3; ++i) {
      links_copied[i].rotation *= math::QuaternionExp(angles.segment(i * 3, 3));
    }
    ChainComputationBuffer c{};
    ComputeChain(links_copied, &c);
    return c.i_t_end.leftCols<1>();
  };

  const auto rotation_wrt_rot = [&](const VectorXd& angles) -> Quaterniond {
    std::vector<Pose> links_copied = links;
    for (int i = 0; i < angles.rows() / 3; ++i) {
      links_copied[i].rotation *= math::QuaternionExp(angles.segment(i * 3, 3));
    }
    ChainComputationBuffer c{};
    ComputeChain(links_copied, &c);
    return c.i_R_end.front();
  };

  // also perturb effector translation wrt intermediate translations
  const auto translation_wrt_trans = [&](const VectorXd& translations) -> Vector3d {
    std::vector<Pose> links_copied = links;
    for (int i = 0; i < translations.rows() / 3; ++i) {
      links_copied[i].translation += translations.segment<3>(i * 3);
    }
    ChainComputationBuffer c{};
    ComputeChain(links_copied, &c);
    return c.i_t_end.leftCols<1>();
  };

  // compute numerically
  const Matrix<double, 3, Dynamic> trans_D_rot_numerical =
      math::NumericalJacobian(VectorXd::Zero(links.size() * 3), translation_wrt_rot);
  const Matrix<double, 3, Dynamic> rot_D_rot_numerical =
      math::NumericalJacobian(VectorXd::Zero(links.size() * 3), rotation_wrt_rot);
  const Matrix<double, 3, Dynamic> trans_D_trans_numerical =
      math::NumericalJacobian(VectorXd::Zero(links.size() * 3), translation_wrt_trans);

  // check against analytical
  ChainComputationBuffer c{};
  ComputeChain(links, &c);

  ASSERT_EIGEN_NEAR(trans_D_rot_numerical, c.translation_D_rotation, tol::kNano)
      << "Numerical:\n"
      << trans_D_rot_numerical.format(test_utils::kNumPyMatrixFmt) << "\n"
      << "Analytical:\n"
      << c.translation_D_rotation.format(test_utils::kNumPyMatrixFmt);

  ASSERT_EIGEN_NEAR(rot_D_rot_numerical, c.rotation_D_rotation, tol::kNano)
      << "Numerical:\n"
      << rot_D_rot_numerical.format(test_utils::kNumPyMatrixFmt) << "\n"
      << "Analytical:\n"
      << c.rotation_D_rotation.format(test_utils::kNumPyMatrixFmt);

  ASSERT_EIGEN_NEAR(trans_D_trans_numerical, c.translation_D_translation, tol::kNano)
      << "Numerical:\n"
      << trans_D_trans_numerical.format(test_utils::kNumPyMatrixFmt) << "\n"
      << "Analytical:\n"
      << c.translation_D_translation.format(test_utils::kNumPyMatrixFmt);

  // pull out poses
  ASSERT_EQ(links.size() + 1, c.i_R_end.size());
  ASSERT_EQ(c.i_R_end.size(), static_cast<size_t>(c.i_t_end.cols()));
  const Pose start_T_end{c.i_R_end.front(), c.i_t_end.leftCols<1>()};

  // note we are iterating over inverted poses `i_T_end`, so i = 0 is in fact the full transform
  Pose start_T_current{};
  const std::vector<Pose> start_T_i = ComputeAllPoses(c);
  for (std::size_t i = 0; i < c.i_R_end.size(); ++i) {
    // compare poses
    ASSERT_EIGEN_NEAR(start_T_current.translation, start_T_i[i].translation, tol::kNano)
        << "i = " << i;
    ASSERT_EIGEN_NEAR(start_T_current.rotation.matrix(), start_T_i[i].rotation.matrix(), tol::kNano)
        << "i = " << i;
    // advance
    start_T_current = start_T_current * links[i];
  }
}

template <std::size_t N, typename Handler>
void GenerateAllMasks(std::array<uint8_t, N> mask, const int i, Handler handler) {
  if (i == N) {
    handler(mask);
    return;
  }
  mask[i] = 0;
  GenerateAllMasks(mask, i + 1, handler);
  mask[i] = 1;
  GenerateAllMasks(mask, i + 1, handler);
}

TEST(ActuatorLinkTest, TestComputePose) {
  const Pose pose{math::QuaternionExp(Vector3d{-0.3, 0.5, 0.4}), Vector3d(0.4, -0.2, 1.2)};

  // try all the possible combinations of params (64 of them)
  std::vector<std::array<uint8_t, 6>> possible_masks;
  GenerateAllMasks(std::array<uint8_t, 6>(), 0,
                   [&](const auto& m) { possible_masks.push_back(m); });
  ASSERT_EQ(64lu, possible_masks.size());

  // some input values we will substitute
  const math::Vector<double, 6> input_params =
      (math::Vector<double, 6>() << 0.2, 0.1, 0.35, -0.2, 0.5, 0.6).finished();

  // storage for the output derivative of rotations
  math::Matrix<double, 3, Eigen::Dynamic> J_out;
  J_out.resize(3, 10);

  // storage for parameters
  VectorXd input_params_dynamic(10);
  constexpr int kParamOffset = 3;

  for (const auto& mask : possible_masks) {
    ActuatorLink link{pose, mask};
    const int num_active = link.ActiveCount();

    const Array<double, 6, 1> mask_float =
        Eigen::Map<const Eigen::Matrix<uint8_t, 6, 1>>(mask.data()).cast<double>();

    // what the result should amount to
    const math::Vector<double, 6> combined_params =
        mask_float * input_params.array() +
        (1 - mask_float) * (math::Vector<double, 6>() << -math::EulerAnglesFromSO3(
                                link.parent_T_child.rotation.conjugate()),
                            link.parent_T_child.translation)
                               .finished()
                               .array();

    // put the input angles in the right space in the buffer
    input_params_dynamic.setConstant(std::numeric_limits<double>::quiet_NaN());
    for (int i = 0, output_pos = kParamOffset; i < 6; ++i) {
      if (mask[i]) {
        input_params_dynamic[output_pos++] = combined_params[i];
      }
    }

    J_out.setZero();
    const Pose computed_pose =
        link.Compute(input_params_dynamic, 3, J_out.middleCols(3, link.ActiveRotationCount()));
    ASSERT_EIGEN_NEAR(computed_pose.translation, combined_params.tail<3>(), tol::kPico);
    ASSERT_EIGEN_NEAR(
        computed_pose.rotation.matrix(),
        math::SO3FromEulerAngles(combined_params.head<3>(), math::CompositionOrder::XYZ).q.matrix(),
        tol::kPico);

    // derivative should also respect the active flag
    const auto lambda = [&](const VectorXd& params) {
      Eigen::Matrix<double, 3, Dynamic> unused(3, 10);
      return link.Compute(params, kParamOffset, unused.leftCols(link.ActiveRotationCount()))
          .rotation;
    };
    const Matrix<double, 3, Dynamic> J_numerical =
        math::NumericalJacobian(input_params_dynamic, lambda);
    ASSERT_EIGEN_NEAR(J_numerical, J_out, tol::kPico);
  }
}

TEST(ActuatorChainTest, TestComputeEffector) {
  // create some links
  // clang-format off
  const std::vector<Pose> links = {
    {math::QuaternionExp(Vector3d{-0.5, 0.5, 0.3}), {1.0, 0.5, 2.0}},
    {math::QuaternionExp(Vector3d{0.8, 0.5, 1.2}), {0.5, 0.75, -0.5}},
    {math::QuaternionExp(Vector3d{1.5, -0.2, 0.0}), {1.2, -0.5, 0.1}},
    {math::QuaternionExp(Vector3d{0.2, -0.1, 0.3}), {0.1, -0.1, 0.2}}
  };
  // clang-format on

  // generate mask combinations
  std::vector<std::array<uint8_t, 6>> possible_masks;
  GenerateAllMasks(std::array<uint8_t, 6>(), 0,
                   [&](const auto& m) { possible_masks.push_back(m); });

  // apply different masks to different links to make sure this works
  for (std::size_t mask_index = 0; mask_index <= possible_masks.size() - links.size();) {
    ActuatorChain chain{};
    for (const Pose& pose : links) {
      chain.links.emplace_back(pose, possible_masks[mask_index++]);
    }
    const int total_active = chain.TotalActive();
    ASSERT_GT(total_active, 0);
    ASSERT_LT(total_active, 6 * 4);

    // create a bunch of values to use for the optimized parameters
    // the exact values don't matter here
    Eigen::VectorXd params(total_active);
    for (int i = 0; i < total_active; ++i) {
      params[i] = (i % 2) ? (i * 0.112) : (-i * 0.0421);
    }

    // update the chain
    chain.Update(params);
    const math::Matrix<double, 3, Dynamic> translation_D_params = chain.translation_D_params();
    const math::Matrix<double, 3, Dynamic> rotation_D_params = chain.rotation_D_params();

    const auto translation_D_params_numerical =
        math::NumericalJacobian(params, [&](const VectorXd& params) {
          chain.Update(params);
          return chain.translation();
        });
    ASSERT_EIGEN_NEAR(translation_D_params_numerical, translation_D_params, tol::kPico);

    const auto rotation_D_params_numerical =
        math::NumericalJacobian(params, [&](const VectorXd& params) {
          chain.Update(params);
          return chain.rotation();
        });
    ASSERT_EIGEN_NEAR(rotation_D_params_numerical, rotation_D_params, tol::kPico);
  }
}
}  // namespace mini_opt
