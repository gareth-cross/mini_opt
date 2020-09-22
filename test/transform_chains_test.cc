// Copyright 2020 Gareth Cross
#include "transform_chains.hpp"

#include "geometry_utils/numerical_derivative.hpp"
#include "test_utils.hpp"

namespace mini_opt {
using namespace Eigen;

TEST(TransformChainsTest, TestChainComputationBuffer) {
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

  ASSERT_EIGEN_NEAR(J_trans_numerical, c.translation_D_tangent, tol::kNano);
  PRINT_MATRIX(J_trans_numerical);
  PRINT_MATRIX(c.translation_D_tangent);

  ASSERT_EIGEN_NEAR(J_trans_rotational, c.orientation_D_tangent, tol::kNano);
  PRINT_MATRIX(J_trans_rotational);
  PRINT_MATRIX(c.orientation_D_tangent);

  // pull out poses
  ASSERT_EQ(links.size() + 1, c.i_R_end.size());
  ASSERT_EQ(c.i_R_end.size(), static_cast<size_t>(c.i_t_end.cols()));
  const Pose start_T_end{c.i_R_end.front(), c.i_t_end.leftCols<1>()};

  // note we are iterating over inverted poses `i_T_end`, so i = 0 is in fact the full transform
  Pose start_T_current{};
  for (int i = 0; i < c.i_R_end.size(); ++i) {
    const Pose start_T_i = start_T_end * Pose(c.i_R_end[i], c.i_t_end.col(i)).Inverse();
    // compare poses
    ASSERT_EIGEN_NEAR(start_T_current.translation, start_T_i.translation, tol::kNano)
        << "i = " << i;
    ASSERT_EIGEN_NEAR(start_T_current.rotation.matrix(), start_T_i.rotation.matrix(), tol::kNano)
        << "i = " << i;
    // advance
    start_T_current = start_T_current * links[i];
  }
}

}  // namespace mini_opt
