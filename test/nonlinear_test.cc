// Copyright 2020 Gareth Cross
#include "mini_opt/nonlinear.hpp"

#include "geometry_utils/numerical_derivative.hpp"
#include "test_utils.hpp"
#include "mini_opt/transform_chains.hpp"

namespace mini_opt {
using namespace Eigen;

template <int ResidualDim, int NumParams>
void TestResidualFunctionDerivative(
    const std::function<Eigen::Matrix<double, ResidualDim, 1>(
        const Eigen::Matrix<double, NumParams, 1>&,
        Eigen::Matrix<double, ResidualDim, NumParams>* const)>& function,
    const Eigen::Matrix<double, NumParams, 1>& params, const double tol = tol::kNano) {
  static_assert(ResidualDim != Eigen::Dynamic, "ResidualDim cannot be dynamic");

  // Compute analytically.
  Eigen::Matrix<double, ResidualDim, NumParams> J;
  J.resize(ResidualDim, params.rows());
  function(params, &J);

  // evaluate numerically
  const Eigen::Matrix<double, ResidualDim, NumParams> J_numerical = math::NumericalJacobian(
      params, [&](const Eigen::Matrix<double, NumParams, 1>& x) { return function(x, nullptr); });

  ASSERT_EIGEN_NEAR(J_numerical, J, tol);
}

// Test constrained non-linear least squares.
class ConstrainedNLSTest : public ::testing::Test {
 public:
  // Test a simple non-linear least squares problem.
  void TestActuatorChain() {
    // We have a chain of three rotational actuators, at the end of which we have an effector.
    // Two actuators can effectuate translation, whereas the last one can only rotate the effector.
    std::unique_ptr<ActuatorChain> chain = std::make_unique<ActuatorChain>();
    const std::array<uint8_t, 3> mask = {{0, 0, 1}};
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.0, 0.0, 0.0}), mask);
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.4, 0.0, 0.0}), mask);
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.4, 0.0, 0.0}),
                              std::array<uint8_t, 3>{{0, 0, 0}} /* turn off for now */);

    // make a cost that we want to achieve a specific point vertically
    Residual<1, Dynamic> y_residual;
    y_residual.index = {0, 1};
    y_residual.function = [&](const VectorXd& params,
                              Matrix<double, 1, Dynamic>* const J_out) -> Matrix<double, 1, 1> {
      // todo: don't evaluate full xyz jacobian here
      Matrix<double, 3, Dynamic> J_full(3, chain->TotalActive());
      const Vector3d effector_xyz =
          chain->ComputeEffectorPosition(params, J_out ? &J_full : nullptr);
      if (J_out) {
        *J_out = J_full.middleRows<1>(1);
      }
      return Matrix<double, 1, 1>{effector_xyz.y() - 0.6};
    };

    // make a cost that minimizes the sum of squares of the angles
    /*Residual<2, Dynamic> ss_residual;
    ss_residual.index = {{0, 1}};
    ss_residual.function = [&](const VectorXd& params,
                               Matrix<double, 2, Dynamic>* const J_out) -> Matrix<double, 2, 1> {
      if (J_out) {
        J_out->setZero();
        J_out->diagonal().setConstant(0.1);
      }
      return 0.1 * params;
    };*/

    // make an equality constraint on x
    Residual<1, Dynamic> x_eq_constraint;
    x_eq_constraint.index = {0, 1};
    x_eq_constraint.function =
        [&](const VectorXd& params,
            Matrix<double, 1, Dynamic>* const J_out) -> Matrix<double, 1, 1> {
      Matrix<double, 3, Dynamic> J_full(3, chain->TotalActive());
      const Vector3d effector_xyz =
          chain->ComputeEffectorPosition(params, J_out ? &J_full : nullptr);
      if (J_out) {
        *J_out = J_full.topRows<1>();
      }
      return Matrix<double, 1, 1>{effector_xyz.x() - 0.45};
    };

    Residual<2, Dynamic> combined_soft;
    combined_soft.index = {0, 1};
    combined_soft.function = [&](const VectorXd& params,
                                 Matrix<double, 2, Dynamic>* const J_out) -> Matrix<double, 2, 1> {
      // todo: don't evaluate full xyz jacobian here
      Matrix<double, 3, Dynamic> J_full(3, chain->TotalActive());
      const Vector3d effector_xyz =
          chain->ComputeEffectorPosition(params, J_out ? &J_full : nullptr);
      if (J_out) {
        ASSERT(J_out->cols() == 2);
        *J_out = J_full.topRows<2>();
      }
      return effector_xyz.head<2>() - Vector2d{0.45, 0.6};
    };

    std::vector<double> angles;
    for (double angle = -M_PI * 0.999; angle <= M_PI * 0.999; angle += (0.5 * M_PI / 180.0)) {
      angles.push_back(angle);
    }

    // Eigen::MatrixXd costs(angles.size(), angles.size());
    ///* Eigen::MatrixXd gradients(angles.size(), angles.size() * 2);
    // Eigen::MatrixXd hessian_det(angles.size(), angles.size());*/
    // for (int row = 0; row < angles.size(); ++row) {
    //  for (int col = 0; col < angles.size(); ++col) {
    //    const Vector2d angles_pt(angles[row], angles[col]);

    //    /* const Eigen::Vector2d gradient =
    //         math::NumericalJacobian(angles_pt, [&](const Vector2d& angles_pt) -> double {
    //           return combined_soft.function(angles_pt, nullptr).squaredNorm();
    //         }).transpose();

    //     const Eigen::Matrix2d hessian =
    //         math::NumericalJacobian(angles_pt, [&](const Vector2d& angles_pt) -> Vector2d {
    //           return math::NumericalJacobian(
    //                      angles_pt,
    //                      [&](const Vector2d& angles_pt) -> double {
    //                        return combined_soft.function(angles_pt, nullptr).squaredNorm();
    //                      })
    //               .transpose();
    //         });
    //     hessian_det(row, col) = hessian.determinant();

    //     gradients(row, col) = gradient[0];
    //     gradients(row, angles.size() + col) = gradient[1];*/

    //    costs(row, col) = 0.5 * combined_soft.function(angles_pt, nullptr).squaredNorm();
    //  }
    //}

    // Eigen::Index min_row, min_col;
    // const double min_coeff = costs.minCoeff(&min_row, &min_col);

    /*PRINT(min_coeff);
    PRINT(angles[min_row]);
    PRINT(angles[min_col]);*/

    /*const IOFormat csv_format(FullPrecision, 0, ", ", "\n", "", "", "", "");
    std::ofstream out("C:/Users/garet/Documents/test.csv");
    out << costs.format(csv_format);
    out.flush();*/
    /*
    std::ofstream out_gradients("C:/Users/garet/Documents/gradients.csv");
    out_gradients << gradients.format(csv_format);
    out_gradients.flush();

    std::ofstream out_hessian("C:/Users/garet/Documents/hessian.csv");
    out_hessian << hessian_det.format(csv_format);
    out_hessian.flush();*/

    TestResidualFunctionDerivative<2, Dynamic>(combined_soft.function,
                                               VectorXd{Vector2d(-0.5, 0.4)});
    TestResidualFunctionDerivative<1, Dynamic>(y_residual.function, VectorXd{Vector2d(-0.5, 0.4)});
    TestResidualFunctionDerivative<1, Dynamic>(x_eq_constraint.function,
                                               VectorXd{Vector2d(0.3, -0.6)});

    Problem problem{};
    problem.costs.emplace_back(new Residual<1, Dynamic>(y_residual));
    // problem.costs.emplace_back(new Residual<2, Dynamic>(combined_soft));
    // problem.costs.emplace_back(new Residual<2, Dynamic>(ss_residual));
    problem.equality_constraints.emplace_back(new Residual<1, Dynamic>(x_eq_constraint));
    // problem.inequality_constraints.push_back(Var(0) <= 3 * M_PI / 4);
    // problem.inequality_constraints.push_back(Var(0) >= -3 * M_PI / 4);
    // problem.inequality_constraints.push_back(Var(1) <= 3 * M_PI / 4);
    problem.inequality_constraints.push_back(Var(1) >= 0);
    problem.dimension = 2;

    ConstrainedNonlinearLeastSquares nls(&problem);
    // nls.SetQPLoggingCallback(&QPSolverTest::ProgressPrinter);

    const Vector2d initial_values{M_PI / 4, -M_PI / 6};
    nls.SetVariables(initial_values);
    // try {
    nls.LinearizeAndSolve(10.0);
    nls.LinearizeAndSolve(0.001);
    nls.LinearizeAndSolve(0.001);
    nls.LinearizeAndSolve(0.001);
    nls.LinearizeAndSolve(0.001);
    nls.LinearizeAndSolve(0.001);
    //} catch (const FailedFactorization&) {
    //}

    /*nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(0.1);
    nls.LinearizeAndSolve(0.01);
    nls.LinearizeAndSolve(1.0);
    nls.LinearizeAndSolve(0.1);
    nls.LinearizeAndSolve(0.1);
    nls.LinearizeAndSolve(0.1);
    nls.LinearizeAndSolve(0.1);
    nls.LinearizeAndSolve(0.1);*/

    const VectorXd angles_out = nls.variables();
    PRINT_MATRIX(chain->ComputeEffectorPosition(angles_out).transpose());

    /*   const Eigen::Matrix2d hessian =
           math::NumericalJacobian(Vector2d(angles_out), [&](const Vector2d& angles_pt) -> Vector2d
       { return math::NumericalJacobian( angles_pt,
                        [&](const Vector2d& angles_pt) -> double {
                          return combined_soft.function(angles_pt, nullptr).squaredNorm();
                        })
                 .transpose();
           });

       PRINT_MATRIX(hessian);*/

    // PRINT(y_residual.Error(angles_out));
    // PRINT(x_eq_constraint.Error(angles_out));
    PRINT(combined_soft.Error(angles_out));

    combined_soft.Error(angles_out);
    const Pose start_T_end = chain->chain_buffer_.start_T_end();

    for (int i = 0; i < chain->chain_buffer_.i_t_end.cols(); ++i) {
      const Quaterniond i_R_end = chain->chain_buffer_.i_R_end[i];
      const Vector3d& i_t_end = chain->chain_buffer_.i_t_end.col(i);
      const Pose start_T_i = start_T_end * Pose(i_R_end, i_t_end).Inverse();

      PRINT(i);
      PRINT_MATRIX(start_T_i.translation);
    }
  }
};

TEST_FIXTURE(ConstrainedNLSTest, TestActuatorChain)

}  // namespace mini_opt
