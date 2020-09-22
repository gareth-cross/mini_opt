// Copyright 2020 Gareth Cross
#include "mini_opt/residual.hpp"

#include "test_utils.hpp"

namespace mini_opt {
using namespace Eigen;

// our test function, two polynomials
//    f(x, y, z) = [
//      x*2 + x*y - z^2*y
//      x*y^2 - z*y^2 + z^2*x
//    ]
static Eigen::Vector2d DummyFunction(const Matrix<double, 3, 1>& p, Matrix<double, 2, 3>* const J) {
  const double x = p.x();
  const double y = p.y();
  const double z = p.z();
  const Vector2d f{
      x * x + x * y - z * z * y,
      x * y * y - z * y * y + z * z * x,
  };
  if (J) {
    J->topRows<1>() = RowVector3d{x + y, x - z * z, y};
    J->bottomRows<1>() = RowVector3d{y * y + z * z, x - z, -y * y + x};
  }
  return f;
}

// Create remap matrix for a given index set.
// This is the matrix `M` such that (M * H * M.T) will extract the relevant values
// from the hessian.
template <std::size_t N>
static Eigen::MatrixXd CreateRemapMatrix(const std::array<int, N>& index, const int full_size) {
  Eigen::MatrixXd small_D_large(N, full_size);
  small_D_large.setZero();
  for (int row = 0; row < static_cast<int>(N); ++row) {
    ASSERT(index[row] < full_size);
    small_D_large(row, index[row]) = 1;
  }
  return small_D_large;
}

// Test the statically-sized residual struct.
TEST(MiniOptTest, TestStaticResidualSimple) {
  Residual<2, 3> res;
  res.function = &DummyFunction;
  res.index = {{0, 1, 2}};

  // pick some params for xyz
  const Vector3d params_xyz = {-0.5, 1.2, 0.3};

  Matrix<double, 2, 3> J;
  const Vector2d expected_error = DummyFunction(params_xyz, &J);

  // allocate H and b
  MatrixXd H = Matrix3d::Zero();
  VectorXd b = Vector3d::Zero();

  // check error
  const VectorXd global_params = params_xyz;
  ASSERT_EQ(expected_error.squaredNorm(), 2 * res.Error(global_params));

  // update
  res.UpdateHessian(global_params, &H, &b);

  const auto H_full = H.selfadjointView<Eigen::Lower>().toDenseMatrix();
  ASSERT_EIGEN_NEAR(J.transpose() * J, H_full, tol::kPico);
  ASSERT_EIGEN_NEAR(J.transpose() * expected_error, b, tol::kPico);
}

// Test re-ordering the params.
TEST(MiniOptTest, TestStaticResidualOutOfOrder) {
  Residual<2, 3> res;
  res.function = &DummyFunction;
  res.index = {{2, 0, 1}};
  const auto local_D_global = CreateRemapMatrix(res.index, 3);

  // pick some params for xyz
  const Vector3d params_xyz = {0.23, -0.9, 1.11};

  Matrix<double, 2, 3> J;
  const Vector2d expected_error = DummyFunction(params_xyz, &J);

  // allocate H and b
  MatrixXd H = Matrix3d::Zero();
  VectorXd b = Vector3d::Zero();

  // check error
  const VectorXd global_params = local_D_global.transpose() * params_xyz;
  ASSERT_EQ(expected_error.squaredNorm(), 2 * res.Error(global_params));

  // update
  res.UpdateHessian(global_params, &H, &b);

  // check that it agrees after remapping
  const auto H_full = H.selfadjointView<Eigen::Lower>().toDenseMatrix();
  ASSERT_EIGEN_NEAR(J.transpose() * J, local_D_global * H_full * local_D_global.transpose(),
                    tol::kPico);
  ASSERT_EIGEN_NEAR(J.transpose() * expected_error, local_D_global * b, tol::kPico);
}

// Test indexing into a larger matrix.
TEST(MiniOptTest, TestStaticResidualSparseIndex) {
  Residual<2, 3> res;
  res.function = &DummyFunction;
  res.index = {{5, 1, 3}};
  const auto local_D_global = CreateRemapMatrix(res.index, 7);

  // pick some params for xyz
  const Vector3d params_xyz = {0.99, -0.23, 2.2};

  Matrix<double, 2, 3> J;
  const Vector2d expected_error = DummyFunction(params_xyz, &J);

  // allocate H and b
  MatrixXd H = MatrixXd::Zero(7, 7);
  VectorXd b = VectorXd::Zero(7);

  // check error
  const VectorXd global_params = local_D_global.transpose() * params_xyz;
  ASSERT_EQ(expected_error.squaredNorm(), 2 * res.Error(global_params));

  // update
  res.UpdateHessian(global_params, &H, &b);

  // check that top half is zero
  for (int col = 0; col < H.cols(); ++col) {
    for (int row = 0; row < col; ++row) {
      ASSERT_EQ(0.0, H(row, col));
    }
  }

  // check that it agrees after remapping
  const auto H_full = H.selfadjointView<Eigen::Lower>().toDenseMatrix();
  ASSERT_EIGEN_NEAR(J.transpose() * J, local_D_global * H_full * local_D_global.transpose(),
                    tol::kPico);
  ASSERT_EIGEN_NEAR(J.transpose() * expected_error, local_D_global * b, tol::kPico);

  // check that other cells were left at zero by UpdateHessian
  const auto H_empty = (H.selfadjointView<Eigen::Lower>().toDenseMatrix() -
                        (J * local_D_global).transpose() * (J * local_D_global))
                           .eval();
  ASSERT_EIGEN_NEAR(MatrixXd::Zero(7, 7), H_empty, 0.0);
}

// Test with dynamic # of params.
TEST(MiniOptTest, TestDynamicParameterVector) {
  Residual<2, Dynamic> res;
  res.index = {0, 1, 2};
  res.function = [&](const VectorXd& p, Matrix<double, 2, Dynamic>* const J) -> Vector2d {
    Matrix<double, 2, 3> J_static;
    const auto r = DummyFunction(p, J ? &J_static : nullptr);
    if (J) {
      ASSERT(2 == J->rows());
      ASSERT(3 == J->cols());
      J->noalias() = J_static;
    }
    return r;
  };

  // pick some params for xyz
  const Vector3d params_xyz = {.099, -0.5, 0.76};

  Matrix<double, 2, 3> J;
  const Vector2d expected_error = DummyFunction(params_xyz, &J);

  // allocate H and b
  MatrixXd H = Matrix3d::Zero();
  VectorXd b = Vector3d::Zero();

  // check error
  const VectorXd global_params = params_xyz;
  ASSERT_EQ(expected_error.squaredNorm(), 2 * res.Error(global_params));

  // update
  res.UpdateHessian(global_params, &H, &b);

  const auto H_full = H.selfadjointView<Eigen::Lower>().toDenseMatrix();
  ASSERT_EIGEN_NEAR(J.transpose() * J, H_full, tol::kPico);
  ASSERT_EIGEN_NEAR(J.transpose() * expected_error, b, tol::kPico);
}

}  // namespace mini_opt
