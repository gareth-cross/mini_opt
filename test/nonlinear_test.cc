// Copyright 2020 Gareth Cross
#include "mini_opt/nonlinear.hpp"

#include "geometry_utils/numerical_derivative.hpp"
#include "mini_opt/logging.hpp"
#include "mini_opt/transform_chains.hpp"
#include "test_utils.hpp"

namespace mini_opt {
using namespace Eigen;
using namespace std::placeholders;

template <int ResidualDim, int NumParams>
static void TestResidualFunctionDerivative(
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
  // Test function that computes the gradient of the cost function against
  // a numerical version.
  void TestComputeQPCostDerivative() {
    using Vector5 = Matrix<double, 5, 1>;
    Residual<3, 5> cost;
    cost.index = {{0, 2, 1, 4, 3}};
    cost.function = [](const Vector5& params, Matrix<double, 3, 5>* J_out) -> Matrix<double, 3, 1> {
      // some nonsense
      const Vector3d result{params[0] * params[1] - params[3],
                            std::sin(params[1]) * std::cos(params[2]),
                            params[3] * params[0] - params[2]};
      if (J_out) {
        J_out->setZero();
        J_out->operator()(0, 0) = params[1];
        J_out->operator()(0, 1) = params[0];
        J_out->operator()(0, 3) = -1;
        J_out->operator()(1, 1) = std::cos(params[1]) * std::cos(params[2]);
        J_out->operator()(1, 2) = -std::sin(params[2]) * std::sin(params[1]);
        J_out->operator()(2, 0) = params[3];
        J_out->operator()(2, 2) = -1;
        J_out->operator()(2, 3) = params[0];
      }
      return result;
    };

    // Make up an equality constraint
    Residual<2, 3> eq;
    eq.index = {{3, 4, 0}};
    eq.function = [](const Vector3d& params, Matrix<double, 2, 3>* J_out) -> Vector2d {
      const Vector2d result{params[0] - params[1], params[2] * params[2]};
      if (J_out) {
        J_out->setZero();
        J_out->operator()(0, 0) = 1;
        J_out->operator()(0, 1) = -1;
        J_out->operator()(1, 2) = 2 * params[2];
      }
      return result;
    };

    // check that the derivatives are correct
    const Vector5 params = (Vector5() << -0.5, 0.2, 0.3, -0.8, 1.2).finished();
    TestResidualFunctionDerivative<3, 5>(cost.function, params);
    TestResidualFunctionDerivative<2, 3>(eq.function, params.head<3>());

    // pick a direction for the directional derivative
    const Vector5 dx = (Vector5() << 0.1, 0.25, -0.87, 1.1, -0.02).finished();

    // compute the derivative of the sum cost function
    const Matrix<double, 1, 1> J_numerical = math::NumericalJacobian(0.0, [&](const double alpha) {
      return cost.Error(params + dx * alpha) + eq.Error(params + dx * alpha);
    });

    // set up a problem
    Problem problem{};
    problem.dimension = 5;
    problem.costs.emplace_back(new Residual<3, 5>(cost));
    problem.equality_constraints.emplace_back(new Residual<2, 3>(eq));

    // compute analytically as well
    QP qp{static_cast<Index>(problem.dimension)};
    qp.A_eq.resize(2, 5);
    qp.b_eq.resize(2);
    ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(params, 0.0, problem, &qp);

    const double J_analytical = ConstrainedNonlinearLeastSquares::ComputeQPCostDerivative(qp, dx);
    ASSERT_NEAR(J_numerical[0], J_analytical, tol::kPico);
  }

  void TestQuadraticApproxMinimum() {
    // make up some values
    const double alpha_0 = 0.8;
    const double phi_0 = 2.0;
    const double phi_prime_0 = -1.2;
    const double phi_alpha_0 = 2.2;

    // compute via form 1
    const double solution = ConstrainedNonlinearLeastSquares::QuadraticApproxMinimum(
        phi_0, phi_prime_0, alpha_0, phi_alpha_0);

    const double a = (phi_alpha_0 - phi_0 - alpha_0 * phi_prime_0) / std::pow(alpha_0, 2);
    const double b = phi_prime_0;
    ASSERT_NEAR(-b / (2 * a), solution, tol::kPico);
  }

  // Check that close form cubic approximation is correct.
  void TestCubicApproxCoeffs() {
    const double alpha_0 = 0.8;
    const double alpha_1 = 0.4;
    const double phi_0 = 1.44;
    const double phi_prime_0 = -1.23;

    const double phi_alpha_0 = 2.2;  //  cost did not decrease
    const double phi_alpha_1 = 1.6;  //  still did not decrease

    // fit the cubic
    const Vector2d ab = ConstrainedNonlinearLeastSquares::CubicApproxCoeffs(
        phi_0, phi_prime_0, alpha_0, phi_alpha_0, alpha_1, phi_alpha_1);

    // check that the polynomial is correct
    ASSERT_NEAR(
        ab[0] * std::pow(alpha_0, 3) + ab[1] * std::pow(alpha_0, 2) + alpha_0 * phi_prime_0 + phi_0,
        phi_alpha_0, tol::kPico);
    ASSERT_NEAR(
        ab[0] * std::pow(alpha_1, 3) + ab[1] * std::pow(alpha_1, 2) + alpha_1 * phi_prime_0 + phi_0,
        phi_alpha_1, tol::kPico);

    // find the minimum
    const double min_alpha = ConstrainedNonlinearLeastSquares::CubicApproxMinimum(phi_prime_0, ab);

    // check it actually is the minimum
    ASSERT_NEAR(0.0, 3 * ab[0] * std::pow(min_alpha, 2) + 2 * ab[1] * min_alpha + phi_prime_0,
                tol::kPico);
    ASSERT_GT(6 * ab[0] * min_alpha + 2 * ab[1], 0);
  }

  static Vector2d Rosenbrock(const Vector2d& params, Matrix2d* const J_out = nullptr) {
    constexpr double a = 1.0;
    constexpr double b = 100.0;
    if (J_out) {
      J_out->setZero();
      J_out->operator()(0, 0) = -1.0;
      J_out->operator()(1, 0) = -2 * params.x() * std::sqrt(b);
      J_out->operator()(1, 1) = std::sqrt(b);
    }
    // we select h(x,y) st. h(x,y)^T * h(x,y) = (a - x)^2 + b*(y - x^2)^2 where b=100, a=1
    return {a - params.x(), std::sqrt(b) * (params.y() - params.x() * params.x())};
  }

  // Test optimization of rosenbrock function w/ no constraints.
  void TestRosenbrock() {
    Residual<2, 2> rosenbrock;
    rosenbrock.index = {{0, 1}};
    rosenbrock.function = &ConstrainedNLSTest::Rosenbrock;

    // check that it behaves correctly
    TestResidualFunctionDerivative<2, 2>(rosenbrock.function, Vector2d{5, -3});
    TestResidualFunctionDerivative<2, 2>(rosenbrock.function, Vector2d{1, 1});
    ASSERT_EIGEN_NEAR(Vector2d::Zero(), Rosenbrock({1, 1}), tol::kPico);

    // simple problem with only one cost
    Problem problem{};
    problem.costs.emplace_back(new Residual<2, 2>(rosenbrock));
    problem.dimension = 2;

    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 5;
    p.max_qp_iterations =
        1;  //  since this is quadratic w/ no constrains, should only need one of these

    // Solve it from a few different initial guesses
    const AlignedVector<Vector2d> initial_guesses = {{-5, -3}, {10, 8},   {-20, 3}, {0, -5},
                                                     {4, 0},   {100, 50}, {-35, 40}};
    for (const Vector2d& guess : initial_guesses) {
      Logger logger{};
#if 0
      nls.SetQPLoggingCallback(
          std::bind(&Logger::QPSolverCallbackVerbose, &logger, _1, _2, _3, _4));
#endif
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1));

      // solve it
      const NLSTerminationState term_state = nls.Solve(p, guess);
      ASSERT_EQ(term_state, NLSTerminationState::SATISFIED_ABSOLUTE_TOL);

      std::cout << logger.GetString() << std::endl;

      // check solution
      ASSERT_EIGEN_NEAR(Vector2d::Ones(), nls.variables(), tol::kNano) << "Summary:\n"
                                                                       << logger.GetString();
    }
  }

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
    // nls.SetVariables(initial_values);
    //// try {
    // nls.LinearizeAndSolve(10.0);
    // nls.LinearizeAndSolve(0.001);
    // nls.LinearizeAndSolve(0.001);
    // nls.LinearizeAndSolve(0.001);
    // nls.LinearizeAndSolve(0.001);
    // nls.LinearizeAndSolve(0.001);
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

TEST_FIXTURE(ConstrainedNLSTest, TestComputeQPCostDerivative)
TEST_FIXTURE(ConstrainedNLSTest, TestQuadraticApproxMinimum)
TEST_FIXTURE(ConstrainedNLSTest, TestCubicApproxCoeffs)
TEST_FIXTURE(ConstrainedNLSTest, TestRosenbrock)
// TEST_FIXTURE(ConstrainedNLSTest, TestActuatorChain)

}  // namespace mini_opt
