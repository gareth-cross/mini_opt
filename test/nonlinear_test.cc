// Copyright 2020 Gareth Cross
#include "mini_opt/nonlinear.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <fstream>
#include <numeric>
#include <random>

#include "geometry_utils/numerical_derivative.hpp"
#include "mini_opt/logging.hpp"
#include "mini_opt/math_utils.hpp"
#include "mini_opt/transform_chains.hpp"
#include "test_utils.hpp"

// TODO(gareth): Split up this file a bit.
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
      Vector3d result{params[0] * params[1] - params[3], std::sin(params[1]) * std::cos(params[2]),
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
      Vector2d result{params[0] - params[1], -params[2] * params[2]};
      if (J_out) {
        J_out->setZero();
        J_out->operator()(0, 0) = 1;
        J_out->operator()(0, 1) = -1;
        J_out->operator()(1, 2) = -2 * params[2];
      }
      return result;
    };

    // check that the derivatives are correct
    const Vector5 params = (Vector5() << -0.5, 0.2, 0.3, -0.8, 1.2).finished();
    TestResidualFunctionDerivative<3, 5>(cost.function, params);
    TestResidualFunctionDerivative<2, 3>(eq.function, params.head<3>());

    // pick a direction for the directional derivative
    const VectorXd dx = (Vector5() << 0.1, 0.25, -0.87, 1.1, -0.02).finished();

    // The penalty on the equality constraint
    const double penalty = 0.334;

    // set up a problem
    Problem problem{};
    problem.dimension = 5;
    problem.costs.emplace_back(new Residual<3, 5>(cost));
    problem.equality_constraints.emplace_back(new Residual<2, 3>(eq));

    // test with L1
    {
      // compute the derivative of the sum cost function
      const Matrix<double, 1, 1> J_numerical =
          math::NumericalJacobian(0.0, [&](const double alpha) {
            Eigen::VectorXd cost_out(3);
            Eigen::VectorXd equality_out(2);
            cost.ErrorVector(params + dx * alpha, cost_out.head(3));
            eq.ErrorVector(params + dx * alpha, equality_out.head(2));
            return 0.5 * cost_out.squaredNorm() + penalty * equality_out.lpNorm<1>();
          });

      // compute analytically as well
      QP qp{static_cast<Index>(problem.dimension)};
      qp.A_eq.resize(2, 5);
      qp.b_eq.resize(2);
      ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(params, 0.0, Norm::L1, problem, &qp);

      const DirectionalDerivatives J_analytical =
          ConstrainedNonlinearLeastSquares::ComputeQPCostDerivative(qp, dx.head(5), Norm::L1);
      ASSERT_NEAR(J_numerical[0], J_analytical.Total(penalty), tol::kPico);
    }

    // test with Quadratic
    {
      const Matrix<double, 1, 1> J_numerical =
          math::NumericalJacobian(0.0, [&](const double alpha) {
            Eigen::VectorXd cost_out(3);
            Eigen::VectorXd equality_out(2);
            cost.ErrorVector(params + dx * alpha, cost_out.head(3));
            eq.ErrorVector(params + dx * alpha, equality_out.head(2));
            return 0.5 * cost_out.squaredNorm() + penalty * 0.5 * equality_out.squaredNorm();
          });

      // compute analytically as well
      QP qp{static_cast<Index>(problem.dimension)};
      qp.A_eq.resize(2, 5);
      qp.b_eq.resize(2);
      ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(params, 0.0, Norm::L1, problem, &qp);

      const DirectionalDerivatives J_analytical =
          ConstrainedNonlinearLeastSquares::ComputeQPCostDerivative(qp, dx.head(5),
                                                                    Norm::QUADRATIC);
      ASSERT_NEAR(J_numerical[0], J_analytical.Total(penalty), tol::kPico);
    }
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

  void TestComputeSecondOrderCorrection() {
    // create a non-linear equality constraint
    Residual<2, 3> eq;
    eq.index = {{0, 1, 2}};
    eq.function = [](const Vector3d& x, Matrix<double, 2, 3>* J) -> Vector2d {
      if (J) {
        J->setZero();
        J->operator()(0, 0) = 2 * x[0];
        J->operator()(1, 0) = x[1];
        J->operator()(1, 1) = x[0];
        J->operator()(1, 2) = -1;
      }
      return {
          x[0] * x[0],
          x[1] * x[0] - x[2],
      };
    };

    const Vector3d x_lin{-0.5, 1.2, -0.5};
    TestResidualFunctionDerivative<2, 3>(eq.function, x_lin);

    // make space for linearized problem
    QP qp{};
    qp.A_eq.resize(2, 3);
    qp.b_eq.resize(2);
    qp.G.resize(3, 3);  //  unused but needs to be allocated
    qp.c.resize(3, 1);  //  unused but needs to be allocated

    // linearize it
    Problem problem{};
    problem.dimension = 3;
    problem.equality_constraints.emplace_back(new Residual<2, 3>(eq));
    ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(x_lin, 0.0, Norm::L1, problem, &qp);

    // find a solution that satisfies the linearized problem
    CompleteOrthogonalDecomposition<MatrixXd> decomposition(qp.A_eq);
    const VectorXd dx_original = -decomposition.solve(qp.b_eq);
    PRINT_MATRIX(dx_original.transpose());
    ASSERT_EIGEN_NEAR(qp.A_eq * dx_original + qp.b_eq, Vector2d::Zero(), tol::kPico);

    // stuff in the null space of A should also satisfy this
    FullPivLU<MatrixXd> full_piv_lu(qp.A_eq);
    const auto null_space = full_piv_lu.kernel();
    ASSERT_EQ(3, null_space.rows());
    ASSERT_EQ(1, null_space.cols());
    PRINT_MATRIX(null_space.transpose());
    ASSERT_EIGEN_NEAR(qp.A_eq * (dx_original + null_space * 4.231) + qp.b_eq, Vector2d::Zero(),
                      tol::kPico);

    // should be the min norm solution (ie. step in any other valid direction increases |x|^2)
    const auto norm_func = [&](double alpha) { return (dx_original + null_space * alpha).norm(); };
    ASSERT_NEAR(0.0, math::NumericalDerivative(0.0, 0.01, norm_func), tol::kPico);
    const auto der_func = [&](double alpha) {
      return math::NumericalDerivative(alpha, 0.01, norm_func);
    };
    ASSERT_GT(math::NumericalDerivative(0.0, 0.01, der_func), 0.0);

    // shift by that step and re-evaluate the nonlinear constraint
    Eigen::VectorXd dx_in(3);
    dx_in = dx_original + null_space * 5.334;  // a hypothetical update step we might have computed

    Eigen::VectorXd dx_out(3);
    dx_out.setZero();
    ConstrainedNonlinearLeastSquares::ComputeSecondOrderCorrection(
        x_lin + dx_in, problem.equality_constraints, &qp, &decomposition, &dx_out);
    PRINT_MATRIX(dx_out.transpose());

    // should satisfy: A * dx_out + c(x + dx_in)
    ASSERT_EIGEN_NEAR(qp.A_eq * dx_out + eq.function(x_lin + dx_in, nullptr), Vector2d::Zero(),
                      tol::kPico);

    // error should be reduced if we re-evaluate
    PRINT_MATRIX(eq.function(x_lin + dx_in, nullptr).transpose());
    PRINT_MATRIX(eq.function(x_lin + dx_in + dx_out, nullptr).transpose());
    ASSERT_GT(eq.function(x_lin + dx_in, nullptr).norm(),
              eq.function(x_lin + dx_in + dx_out, nullptr).norm());
  }

  static void SummarizeCounts(const std::string& name, const std::vector<StatCounters>& counters) {
    ASSERT_GT(counters.size(), 0u);
    // get all the stats and dump them
    fmt::print("Stats from {} trials.\n", counters.size());
    for (const StatCounters::Stats& v :
         {StatCounters::NUM_NLS_ITERATIONS, StatCounters::NUM_QP_ITERATIONS,
          StatCounters::NUM_FAILED_LINE_SEARCHES, StatCounters::NUM_LINE_SEARCH_STEPS}) {
      std::vector<int> sorted;
      std::transform(
          counters.begin(), counters.end(), std::back_inserter(sorted),
          [&](const StatCounters& c) { return (c.counts.count(v) > 0) ? c.counts.at(v) : 0; });
      std::sort(sorted.begin(), sorted.end());
      const auto total = std::accumulate(sorted.begin(), sorted.end(), 0);
      const std::size_t num = sorted.size();
      fmt::print(
          "Iteration counts for [{}], {}:\n"
          "  Mean: {}\n"
          "  Median: {}\n"
          "  Max: {}\n"
          "  Min: {}\n"
          "  95 percentile: {}\n",
          name, v, total / static_cast<double>(num), sorted[num / 2], sorted.back(), sorted.front(),
          sorted[(num * 95) / 100]);
    }
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
  // A very simple function that is easy to minimize.
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

    // Solve it from a few different initial guesses.
    // These guesses aren't that principled, I kind of picked at random.
    const AlignedVector<Vector2d> initial_guesses = {{-5, -3},  {10, 8},     {-20, 3},
                                                     {0, -5},   {4, 0},      {100, 50},
                                                     {-35, 40}, {1000, -50}, {0.8, -0.3}};
    for (const Vector2d& guess : initial_guesses) {
      Logger logger{};
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1, _2));

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      ASSERT_EQ(outputs.termination_state, NLSTerminationState::SATISFIED_ABSOLUTE_TOL);
      ASSERT_EQ(outputs.num_qp_iterations, outputs.num_iterations);

      // check solution
      ASSERT_EIGEN_NEAR(Vector2d::Ones(), nls.variables(), tol::kMicro)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());
    }
  }

  // Solve un-constrained rosenbrock, but don't use line search. Instead, depend on LM.
  void TestRosenbrockLM() {
    Residual<2, 2> rosenbrock;
    rosenbrock.index = {{0, 1}};
    rosenbrock.function = &ConstrainedNLSTest::Rosenbrock;

    // simple problem with only one cost
    Problem problem{};
    problem.costs.emplace_back(new Residual<2, 2>(rosenbrock));
    problem.dimension = 2;

    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 10;
    p.max_qp_iterations = 1;

    // set very tight tolerance on first derivative
    p.absolute_first_derivative_tol = tol::kPico;

    // don't allow line search, instead we depend on LM
    p.max_line_search_iterations = 0;

    // Solve it from a few different initial guesses
    const AlignedVector<Vector2d> initial_guesses = {{-5, -3},  {10, 8},     {-20, 3},
                                                     {0, -5},   {4, 0},      {100, 50},
                                                     {-35, 40}, {1000, -50}, {0.8, -0.3}};
    for (const Vector2d& guess : initial_guesses) {
      Logger logger{false, true};
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1, _2));

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      ASSERT_EQ(outputs.termination_state, NLSTerminationState::SATISFIED_ABSOLUTE_TOL);
      ASSERT_EQ(outputs.num_qp_iterations, outputs.num_iterations);

      // check solution
      ASSERT_EIGEN_NEAR(Vector2d::Ones(), nls.variables(), tol::kMicro)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());
    }
  }

  // Test rosenbrock w/ inequality constraints about the optimum.
  void TestInequalityConstrainedRosenbrock() {
    Residual<2, 2> rosenbrock;
    rosenbrock.index = {{0, 1}};
    rosenbrock.function = &ConstrainedNLSTest::Rosenbrock;

    // simple problem with only one cost
    Problem problem{};
    problem.costs.emplace_back(new Residual<2, 2>(rosenbrock));
    problem.dimension = 2;
    problem.inequality_constraints.push_back(Var(0) >= 1.2);
    problem.inequality_constraints.push_back(Var(1) <= 0.5);

    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 10;
    p.max_qp_iterations = 10;

    // Solve it from a few different initial guesses
    // The last three are actually infeasible to begin with.
    const AlignedVector<Vector2d> initial_guesses = {
        {12, -5}, {100.0, -20.0}, {1423.0, -400.0}, {-20.0, 10.0}, {-120.0, 35.0}, {-50.0, 0.5}};
    std::vector<StatCounters> counters;
    for (const Vector2d& guess : initial_guesses) {
      Logger logger{false, true};
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1, _2));
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.push_back(logger.counters());

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE((outputs.termination_state != NLSTerminationState::MAX_ITERATIONS) &&
                  (outputs.termination_state != NLSTerminationState::MAX_LAMBDA))
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());

      // check solution, it should be at the constraint
      ASSERT_EIGEN_NEAR(Vector2d(1.2, 0.5), nls.variables(), tol::kMicro)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());
    }
    SummarizeCounts("Rosenbrock 2D", counters);
  }

  static Matrix<double, 10, 1> Rosenbrock6D(const Matrix<double, 6, 1>& params,
                                            Matrix<double, 10, 6>* const J_out = nullptr) {
    constexpr double a = 1.0;
    constexpr double b = 100.0;
    if (J_out) {
      J_out->setZero();
    }
    Matrix<double, 10, 1> output;
    output.setConstant(std::numeric_limits<double>::quiet_NaN());  //  make sure we fill
    for (int i = 0; i < params.rows() - 1; i++) {
      output[i * 2] = a - params[i];
      output[i * 2 + 1] = std::sqrt(b) * (params[i + 1] - params[i] * params[i]);
      if (J_out) {
        J_out->operator()(i * 2, i) = -1.0;
        J_out->operator()(i * 2 + 1, i) = -2 * params[i] * std::sqrt(b);
        J_out->operator()(i * 2 + 1, i + 1) = std::sqrt(b);
      }
    }
    return output;
  }

  // Test rosenbrock w/ inequality constraints about the optimum, in 6 dimensions.
  void TestInequalityConstrainedRosenbrock6D() {
    Residual<10, 6> rosenbrock;
    std::iota(rosenbrock.index.begin(), rosenbrock.index.end(), 0);
    rosenbrock.function = &ConstrainedNLSTest::Rosenbrock6D;

    using Vector6 = Matrix<double, 6, 1>;
    TestResidualFunctionDerivative<10, 6>(rosenbrock.function, Vector6::Zero());
    TestResidualFunctionDerivative<10, 6>(rosenbrock.function, Vector6::Ones());

    // simple problem with only one cost
    Problem problem{};
    problem.costs.emplace_back(new Residual<10, 6>(rosenbrock));
    problem.dimension = 6;
    problem.inequality_constraints.push_back(Var(0) >= 2.3);
    problem.inequality_constraints.push_back(Var(1) <= -1.2);
    problem.inequality_constraints.push_back(Var(2) >= 3.0);
    problem.inequality_constraints.push_back(Var(3) <= -2.5);

    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    //  my second guess here takes a lot of iterations - worth studying more perhaps
    p.max_iterations = 30;
    p.max_qp_iterations = 30;
    p.relative_exit_tol = tol::kPico;
    p.absolute_first_derivative_tol = tol::kPico;
    p.termination_kkt_tolerance = tol::kMicro;

    // Solve it from a couple of different initial guesses
    const Vector6 guess0 = (Vector6() << 10.5, -8.0, 50., -14.0, 4.0, -0.6).finished();
    const Vector6 guess1 = (Vector6() << 100.0, -50.0, 30.0, -100.0, 150.0, -400.0).finished();

    // TODO(gareth): Check this value more thoroughly. :S
    const Vector6 solution =
        (Vector6() << 2.3, -1.2, 3.0, -2.5, 6.19802, std::pow(6.19802, 2)).finished();

    std::vector<StatCounters> counters;
    for (const Vector6& guess : {guess0, guess1}) {
      Logger logger{false, true};
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1, _2));
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.push_back(logger.counters());

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE((outputs.termination_state != NLSTerminationState::MAX_ITERATIONS) &&
                  (outputs.termination_state != NLSTerminationState::MAX_LAMBDA) &&
                  (outputs.termination_state != NLSTerminationState::NONE))
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());

      // check solution, it should be at the constraint
      ASSERT_EIGEN_NEAR(solution, nls.variables(), 1.0e-4)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());
    }
    SummarizeCounts("Rosenbrock 6D", counters);
  }

  // For testing, break the Himmelblau function up into two parts.
  static Matrix<double, 1, 1> Himmelblau1(const Vector2d& xy, Matrix<double, 1, 2>* J_out) {
    if (J_out) {
      J_out->operator()(0, 0) = 2 * xy[0];
      J_out->operator()(0, 1) = 1.0;
    }
    return Matrix<double, 1, 1>{std::pow(xy[0], 2) + xy[1] - 11};
  }

  static Matrix<double, 1, 1> Himmelblau2(const Vector2d& xy, Matrix<double, 1, 2>* J_out) {
    if (J_out) {
      J_out->operator()(0, 0) = 1.0;
      J_out->operator()(0, 1) = 2 * xy[1];
    }
    return Matrix<double, 1, 1>{xy[0] + std::pow(xy[1], 2) - 7};
  }

  // Himmelblau w/ box constraints to clamp it into the typical region.
  void TestHimmelblau() {
    TestResidualFunctionDerivative<1, 2>(&ConstrainedNLSTest::Himmelblau1, Vector2d{0, 0});
    TestResidualFunctionDerivative<1, 2>(&ConstrainedNLSTest::Himmelblau1, Vector2d{4, -3});
    TestResidualFunctionDerivative<1, 2>(&ConstrainedNLSTest::Himmelblau2, Vector2d{-1, 3});
    TestResidualFunctionDerivative<1, 2>(&ConstrainedNLSTest::Himmelblau2, Vector2d{0.5, -1.5});

    // break problem into 2 costs for testing
    Problem problem{};
    problem.costs.emplace_back(new Residual<1, 2>({{0, 1}}, &ConstrainedNLSTest::Himmelblau1));
    problem.costs.emplace_back(new Residual<1, 2>({{0, 1}}, &ConstrainedNLSTest::Himmelblau2));
    problem.dimension = 2;

    // first test version bounded to [-5, 5]...
    problem.inequality_constraints.push_back(Var(0) >= -5.0);
    problem.inequality_constraints.push_back(Var(0) <= 5.0);
    problem.inequality_constraints.push_back(Var(1) >= -5.0);
    problem.inequality_constraints.push_back(Var(1) <= 5.0);

    ConstrainedNonlinearLeastSquares nls(&problem);

    // From wikipedia, should get more accurate values for these
    AlignedVector<Vector2d> valid_solutions;
    valid_solutions.emplace_back(3.0, 2.0);
    valid_solutions.emplace_back(-2.805118, 3.131312);
    valid_solutions.emplace_back(-3.779310, -3.283186);
    valid_solutions.emplace_back(3.584428, -1.848126);
    for (const Vector2d& sol : valid_solutions) {
      ASSERT_NEAR(0.0, nls.EvaluateNonlinearErrors(sol, Norm::L1).Total(1.), tol::kMicro);
    }

    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 20;
    p.max_qp_iterations = 10;
    p.relative_exit_tol = tol::kPico;
    p.absolute_first_derivative_tol = tol::kNano * 10;
    p.termination_kkt_tolerance = tol::kMicro;

    // generate a bunch of initial guesses
    AlignedVector<Vector2d> initial_guesses;
    for (double x = -4.5; x <= 4.5; x += 0.3) {    // NOLINT(cert-flp30-c)
      for (double y = -4.5; y <= 4.5; y += 0.3) {  // NOLINT(cert-flp30-c)
        initial_guesses.emplace_back(x, y);
      }
    }

    std::vector<StatCounters> counters;
    for (const Vector2d& guess : initial_guesses) {
      Logger logger{false, true};
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1, _2));
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.push_back(logger.counters());

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE((outputs.termination_state != NLSTerminationState::MAX_ITERATIONS) &&
                  (outputs.termination_state != NLSTerminationState::MAX_LAMBDA) &&
                  (outputs.termination_state != NLSTerminationState::NONE))
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());

      // one of the solutions should match
      const Vector2d best_sol = *std::min_element(
          valid_solutions.begin(), valid_solutions.end(), [&](const auto& v1, const auto& v2) {
            return (v1 - nls.variables()).norm() < (v2 - nls.variables()).norm();
          });

      ASSERT_EIGEN_NEAR(best_sol, nls.variables(), 5.0e-5)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());
    }
    SummarizeCounts("Himmelblau", counters);
  }

  // Himmelblau but constrained to one global optimum.
  void TestHimmelblauQuadrantConstrained() {
    // break problem into 2 costs
    Problem problem{};
    problem.costs.emplace_back(new Residual<1, 2>({{0, 1}}, &ConstrainedNLSTest::Himmelblau1));
    problem.costs.emplace_back(new Residual<1, 2>({{0, 1}}, &ConstrainedNLSTest::Himmelblau2));
    problem.dimension = 2;

    // Constrain to the top right quadrant.
    // We are cheating a bit here, we set the barrier >= 0.1 so that we don't
    // get trapped at local maxima that lies along x = 0, ~y=2.9
    problem.inequality_constraints.push_back(Var(0) >= 0.1);
    problem.inequality_constraints.push_back(Var(0) <= 5.0);
    problem.inequality_constraints.push_back(Var(1) >= 0.1);
    problem.inequality_constraints.push_back(Var(1) <= 5.0);

    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 20;
    p.max_qp_iterations = 10;
    p.relative_exit_tol = tol::kPico;
    p.absolute_first_derivative_tol = tol::kNano * 10;
    p.termination_kkt_tolerance = tol::kMicro;

    // generate a bunch of initial guesses
    AlignedVector<Vector2d> initial_guesses;
    for (double x = 0.2; x <= 4.8; x += 0.2) {    // NOLINT(cert-flp30-c)
      for (double y = 0.2; y <= 4.8; y += 0.2) {  // NOLINT(cert-flp30-c)
        initial_guesses.emplace_back(x, y);
      }
    }

    std::vector<StatCounters> counters;
    for (const Vector2d& guess : initial_guesses) {
      Logger logger{false, true};
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1, _2));
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.push_back(logger.counters());

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE((outputs.termination_state != NLSTerminationState::MAX_ITERATIONS) &&
                  (outputs.termination_state != NLSTerminationState::MAX_LAMBDA) &&
                  (outputs.termination_state != NLSTerminationState::NONE))
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());

      // should match this solution well
      ASSERT_EIGEN_NEAR(Vector2d(3.0, 2.0), nls.variables(), 5.0e-5)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());
    }
    SummarizeCounts("Himmelblau Quadrant Inequality Constrained", counters);
  }

  template <int N>
  static Matrix<double, N, 1> SphereFunction(const Matrix<double, N, 1>& x,
                                             Matrix<double, N, N>* J) {
    if (J) {
      J->setIdentity();
    }
    return x;
  }

  // x_0 * x_1 = v
  // This constraint makes the problem tricky, because as one variable tends towards 0, the
  // other goes to infinity and the cost x^2 becomes very large. It also requires that both
  // variables have the same sign, which means if one starts negative and the other positive,
  // the optimizer has to cross over the gap at 0.
  static Matrix<double, 1, 1> ProductExpression(const Vector2d& x, double v,
                                                Matrix<double, 1, 2>* J) {
    if (J) {
      J->operator[](0) = x[1];
      J->operator[](1) = x[0];
    }
    return Matrix<double, 1, 1>{x[0] * x[1] - v};
  }

  // Test a problem with a non-linear equality constraint.
  // This is a quadratic problem subject to a quadratic equality constraint.
  void TestSphereWithNonlinearEqualityConstraints() {
    constexpr int N = 6;
    std::array<int, N> index = {{0}};
    std::iota(index.begin(), index.end(), 0);

    Problem problem{};
    problem.costs.emplace_back(new Residual<N, N>(index, &ConstrainedNLSTest::SphereFunction<N>));
    problem.dimension = N;

    problem.equality_constraints.emplace_back(new Residual<1, 2>(
        {{0, 1}}, std::bind(&ConstrainedNLSTest::ProductExpression, _1, 4.0, _2)));
    problem.equality_constraints.emplace_back(new Residual<1, 2>(
        {{2, 3}}, std::bind(&ConstrainedNLSTest::ProductExpression, _1, 9.0, _2)));

    // add a nonlinear equality constraint
    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 100;
    p.max_qp_iterations = 1;  //  no inequalities, should be solved in a single step
    p.relative_exit_tol = tol::kPico;
    p.absolute_first_derivative_tol = tol::kNano;
    p.termination_kkt_tolerance = tol::kMicro;

    // We can speed up convergence by adding some non-zero value to the diagonal of the hessian.
    p.lambda_initial = 0.001;

    // generate a bunch of random initial guesses
    AlignedVector<Matrix<double, N, 1>> guesses;
    std::default_random_engine engine{7};  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution<double> dist{-30.0, 30.0};
    for (int i = 0; i < 100; ++i) {
      Matrix<double, N, 1> guess;
      for (int j = 0; j < N; ++j) {
        guess[j] = dist(engine);
      }
      guesses.push_back(guess);
    }

    // viable solutions
    AlignedVector<Matrix<double, N, 1>> solutions;
    for (double x0 : {-2.0, 2.0}) {
      for (double x2 : {-3.0, 3.0}) {
        Matrix<double, N, 1> sol;
        sol.setZero();
        sol[0] = x0;
        sol[1] = x0;
        sol[2] = x2;
        sol[3] = x2;
        solutions.push_back(sol);
      }
    }

    std::vector<StatCounters> counters;
    for (const auto& guess : guesses) {
      Logger logger{true, true};
      nls.SetLoggingCallback(std::bind(&Logger::NonlinearSolverCallback, &logger, _1, _2));
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.push_back(logger.counters());

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE((outputs.termination_state != NLSTerminationState::MAX_ITERATIONS) &&
                  (outputs.termination_state != NLSTerminationState::MAX_LAMBDA) &&
                  (outputs.termination_state != NLSTerminationState::NONE))
          << fmt::format("Termination: {}\nInitial guess: {}\nSummary:\n{}\n",
                         outputs.termination_state,
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());

      // find whichever of the 4 best solutions we found
      const auto min_it =
          std::min_element(solutions.begin(), solutions.end(), [&](const auto& a, const auto& b) {
            return (a - nls.variables()).squaredNorm() < (b - nls.variables()).squaredNorm();
          });

      ASSERT_EIGEN_NEAR(*min_it, nls.variables(), 5.0e-5)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());

      ASSERT_EQ(logger.GetCount(StatCounters::NUM_FAILED_LINE_SEARCHES), 0) << logger.GetString();
    }
    SummarizeCounts("Sphere With Nonlinear Equalities", counters);
  }

  // Test a simple non-linear least squares problem.
  void TestTwoAngleActuatorChain() {
    // We have a chain of three rotational actuators, at the end of which we have an effector.
    // Two actuators can effectuate translation, whereas the last one can only rotate the
    // effector.
    std::unique_ptr<ActuatorChain> chain = std::make_unique<ActuatorChain>();
    const std::array<uint8_t, 6> mask = {{0, 0, 1, 0, 0, 0}};
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.0, 0.0, 0.0}), mask);
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.4, 0.0, 0.0}), mask);
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.4, 0.0, 0.0}),
                              std::array<uint8_t, 6>{{0, 0, 0, 0, 0, 0}} /* turn off for now */);

    // make a cost that we want to achieve a specific point vertically
    Residual<1, Dynamic> y_residual;
    y_residual.index = {0, 1};
    y_residual.function = [&](const VectorXd& params,
                              Matrix<double, 1, Dynamic>* const J_out) -> Matrix<double, 1, 1> {
      chain->Update(params);
      const Vector3d effector_xyz = chain->translation();
      if (J_out) {
        *J_out = chain->translation_D_params().middleRows<1>(1);
      }
      return Matrix<double, 1, 1>{effector_xyz.y() - 0.6};
    };

    // make an equality constraint on x
    Residual<1, Dynamic> x_eq_constraint;
    x_eq_constraint.index = {0, 1};
    x_eq_constraint.function =
        [&](const VectorXd& params,
            Matrix<double, 1, Dynamic>* const J_out) -> Matrix<double, 1, 1> {
      chain->Update(params);
      const Vector3d effector_xyz = chain->translation();
      if (J_out) {
        *J_out = chain->translation_D_params().topRows<1>();
      }
      return Matrix<double, 1, 1>{effector_xyz.x() - 0.45};
    };

    TestResidualFunctionDerivative<1, Dynamic>(y_residual.function, VectorXd{Vector2d(-0.5, 0.4)});
    TestResidualFunctionDerivative<1, Dynamic>(x_eq_constraint.function,
                                               VectorXd{Vector2d(0.3, -0.6)});

    Problem problem{};
    problem.costs.emplace_back(new Residual<1, Dynamic>(y_residual));
    problem.equality_constraints.emplace_back(new Residual<1, Dynamic>(x_eq_constraint));
    problem.dimension = 2;

    ConstrainedNonlinearLeastSquares nls(
        &problem, [](Eigen::VectorXd* const x, const ConstVectorBlock& dx, const double alpha) {
          for (int i = 0; i < x->rows(); ++i) {
            // These are angles, so clamp them in range of [-pi, pi]
            x->operator[](i) = ModPi(x->operator[](i) + dx[i] * alpha);
          }
        });

    // These tolerances are pretty tight - we're likely prompting more iterations than are really
    // useful in practice, but it helps for testing.
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 50;
    p.max_qp_iterations = 1;
    p.relative_exit_tol = tol::kPico;
    p.absolute_first_derivative_tol = 1.0e-10;
    p.absolute_exit_tol = tol::kNano;
    p.termination_kkt_tolerance = tol::kMicro;
    p.max_line_search_iterations = 10;

    // The polynomial approximation does very poorly on this problem near the minimum. Perhaps
    // the quadratic approximation is just really unsuitable?
    p.line_search_strategy = LineSearchStrategy::ARMIJO_BACKTRACK;
    p.equality_constraint_norm = Norm::L1;
    p.lambda_failure_init = 0.001;
    p.armijo_search_tau = 0.5;  //  backtrack more aggressively

    // We add some non-zero lambda because this problem technically does not have
    // a positive semi-definite hessian (since there is only one nonlinear cost
    // on the effector position).
    p.lambda_initial = 0.001;
    p.min_lambda = 1.0e-9;

    // generate a bunch of initial guesses
    AlignedVector<Vector2d> initial_guesses;
    for (double theta0 = tol::kDeci; theta0 <= M_PI / 2; theta0 += 0.1) {   // NOLINT(cert-flp30-c)
      for (double theta1 = -M_PI / 3; theta1 <= M_PI / 3; theta1 += 0.1) {  // NOLINT(cert-flp30-c)
        initial_guesses.emplace_back(theta0, theta1);
      }
    }

    std::vector<StatCounters> counters;
    for (const auto& guess : initial_guesses) {
      Logger logger{true, true};
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));
      nls.SetLoggingCallback(
          [&](const ConstrainedNonlinearLeastSquares& solver, const NLSLogInfo& info) {
            logger.NonlinearSolverCallback(solver, info);
            chain->Update(solver.variables());
            logger.stream() << fmt::format(
                "  Effector: {}\n",
                chain->translation().head(2).transpose().format(test_utils::kNumPyMatrixFmt));
            return true;
          });

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.push_back(logger.counters());

      // check that we reached the desired position
      const VectorXd& angles_out = nls.variables();
      chain->Update(angles_out);
      ASSERT_EIGEN_NEAR(Vector2d(0.45, 0.6), chain->translation().head(2), 5.0e-5) << fmt::format(
          "Termination: {}\nInitial guess: {}\nSummary:\n{}\n", outputs.termination_state,
          guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());
    }
    SummarizeCounts("Only Equality Constrained (NLS)", counters);

    // Now add an inequality constraint and solve it again.
    // force angle 1 to be positive.
    problem.inequality_constraints.push_back(Var(1) >= 0);
    problem.inequality_constraints.push_back(Var(1) <= M_PI);
    initial_guesses.clear();
    for (double theta0 = tol::kDeci; theta0 <= M_PI / 2; theta0 += 0.1) {  // NOLINT(cert-flp30-c)
      for (double theta1 = tol::kMilli; theta1 <= M_PI / 2 - tol::kMilli;
           theta1 += 0.1) {  // NOLINT(cert-flp30-c)
        initial_guesses.emplace_back(theta0, theta1);
      }
    }

    // Need multiple iterations on the QP now.
    p.max_qp_iterations = 10;

    nls.SetLoggingCallback(nullptr);
    nls.SetQPLoggingCallback(nullptr);

    counters.clear();
    for (const auto& guess : initial_guesses) {
      Logger logger{true, true};
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));
      nls.SetLoggingCallback(
          [&](const ConstrainedNonlinearLeastSquares& solver, const NLSLogInfo& info) {
            logger.NonlinearSolverCallback(solver, info);
            logger.stream() << fmt::format("    dx = {}\n",
                                           info.dx.transpose().format(test_utils::kNumPyMatrixFmt));
            return true;
          });

      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.push_back(logger.counters());

      // check that we reached the desired position
      const VectorXd& angles_out = nls.variables();
      chain->Update(angles_out);
      ASSERT_EIGEN_NEAR(Vector2d(0.45, 0.6), chain->translation().head(2), tol::kMilli)
          << fmt::format("Termination: {}\nInitial guess: {}\nSummary:\n{}\n",
                         outputs.termination_state,
                         guess.transpose().format(test_utils::kNumPyMatrixFmt), logger.GetString());

      ASSERT_LT(logger.GetCount(StatCounters::NUM_LINE_SEARCH_STEPS), 100) << logger.GetString();
    }
    SummarizeCounts("Inequality constrained (NLS)", counters);
  }

  // Simple two-legged robot. We apply equality constraints that the feet must contact
  // the floor. We apply a soft cost that the moments must sum to zero (ie. the robot
  // is statically stable).
  // TODO(gareth): This test is a bit gnarly, could do with some cleanup.
  void TestDualActuatorBalancing() {
    const std::array<uint8_t, 6> mask = {{0, 0, 1, 0, 0, 0}};
    const std::array<uint8_t, 6> mask_off = {{0, 0, 0, 0, 0, 0}};

    // front leg
    const Vector3d robot_origin{0, 0.4, 0};
    std::unique_ptr<ActuatorChain> chain_front = std::make_unique<ActuatorChain>();
    chain_front->links.emplace_back(Pose(Quaterniond::Identity(), robot_origin), mask);
    chain_front->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.25, 0.0, 0.0}), mask);
    chain_front->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.3, 0.0, 0.0}), mask);
    chain_front->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.3, 0.0, 0.0}),
                                    mask_off);

    // rear leg
    std::unique_ptr<ActuatorChain> chain_rear = std::make_unique<ActuatorChain>();
    chain_rear->links.emplace_back(Pose(Quaterniond::Identity(), robot_origin), mask);
    chain_rear->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.0, 0.0, 0.0}), mask);
    chain_rear->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.3, 0.0, 0.0}), mask);
    chain_rear->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.3, 0.0, 0.0}),
                                   mask_off);

    Problem problem{};
    problem.dimension = 5;

    // Try to keep the body orientation close to level.
    const auto level_cost = [](const Matrix<double, 1, 1>& body_angle,
                               Matrix<double, 1, 1>* const J_out) -> Matrix<double, 1, 1> {
      if (J_out) {
        J_out->setConstant(0.1);
      }
      return 0.1 * body_angle;
    };
    problem.costs.emplace_back(new Residual<1, 1>({{0}}, level_cost));
    TestResidualFunctionDerivative<1, 1>(level_cost, Matrix<double, 1, 1>{0.4});

    // We want feet to contact the floor (y=0) and achieve a position of the body (which is
    // located on top of the first joint of rear leg) of y=0.4
    const double rear_foot_y = 0.0;
    const double front_foot_y = 0.05;
    const auto rear_foot_expr = [&](const Matrix<double, 3, 1>& angles_rear,
                                    Matrix<double, 1, 3>* const J_out) -> Matrix<double, 1, 1> {
      chain_rear->Update(angles_rear);
      const Vector3d anchor_t_foot = chain_rear->translation();
      if (J_out) {
        *J_out = chain_rear->translation_D_params().middleRows<1>(1);
      }
      return Matrix<double, 1, 1>{anchor_t_foot.y() - rear_foot_y};
    };
    problem.equality_constraints.emplace_back(new Residual<1, 3>({{0, 1, 2}}, rear_foot_expr));
    TestResidualFunctionDerivative<1, 3>(rear_foot_expr, Vector3d{-0.4, 0.2, 0.5});

    // front foot has to end at y=0 as well
    const auto front_foot_expr = [&](const Matrix<double, 3, 1>& angles_front,
                                     Matrix<double, 1, 3>* const J_out) -> Matrix<double, 1, 1> {
      chain_front->Update(angles_front);
      if (J_out) {
        *J_out = chain_front->translation_D_params().middleRows<1>(1);
      }
      return Matrix<double, 1, 1>{chain_front->translation().y() - front_foot_y};
    };
    problem.equality_constraints.emplace_back(new Residual<1, 3>({{0, 3, 4}}, front_foot_expr));
    TestResidualFunctionDerivative<1, 3>(front_foot_expr, Vector3d{0.4, 0.2221, -.8});

    // We want the moments to cancel out.
    // We set mg = 1 (gravity force) and assume two different frictions, mu_rear and mu_front.
    // For simplicity, we assume friction on the rear foot acts to the left (negative x).
    const double mu1 = 1.;
    const double mu2 = 2.;
    const Vector2d com_wrt_anchor{0.15, 0.0};
    const auto moment_expression = [&](const Matrix<double, 5, 1>& all_angles,
                                       Matrix<double, 1, 5>* J_out) -> Matrix<double, 1, 1> {
      chain_rear->Update(all_angles.head<3>());
      chain_front->Update(Vector3d{all_angles[0], all_angles[3], all_angles[4]});

      const Vector3d anchor_t_foot_rear = chain_rear->translation();
      const Vector3d anchor_t_foot_front = chain_front->translation();

      // sum of moments must equal zero
      const double moments = mu1 * (anchor_t_foot_rear.y() - anchor_t_foot_front.y()) +
                             (anchor_t_foot_rear.x() - com_wrt_anchor.x()) +
                             (anchor_t_foot_front.x() - com_wrt_anchor.x()) * mu1 / mu2;
      if (J_out) {
        J_out->setZero();
        // rear
        J_out->leftCols<3>() = mu1 * chain_rear->translation_D_params().middleRows<1>(1);
        J_out->leftCols<3>() += chain_rear->translation_D_params().topRows<1>();
        // front
        // TODO(gareth): Gross, add utilities for this.
        J_out->leftCols<1>() -=
            mu1 * chain_front->translation_D_params().middleRows<1>(1).leftCols<1>();
        J_out->leftCols<1>() +=
            (mu1 / mu2) * chain_front->translation_D_params().topRows<1>().leftCols<1>();
        J_out->rightCols<2>() -=
            mu1 * chain_front->translation_D_params().middleRows<1>(1).rightCols<2>();
        J_out->rightCols<2>() +=
            (mu1 / mu2) * chain_front->translation_D_params().topRows<1>().rightCols<2>();
      }
      return Matrix<double, 1, 1>{moments};
    };
    problem.costs.emplace_back(new Residual<1, 5>({{0, 1, 2, 3, 4}}, moment_expression));
    TestResidualFunctionDerivative<1, 5>(
        moment_expression, (Matrix<double, 5, 1>() << 0.22, -0.3, 0.45, 0.6, -0.1).finished());

    // inequality constraint on the knee of the rear leg
    problem.inequality_constraints.push_back(Var(2) >= 0.0);
    problem.inequality_constraints.push_back(Var(2) <= M_PI);

    // everything is an angle, so retract in the range [-pi, pi]
    ConstrainedNonlinearLeastSquares nls(
        &problem, [](Eigen::VectorXd* const x, const ConstVectorBlock& dx, const double alpha) {
          for (int i = 0; i < x->rows(); ++i) {
            // These are angles, so clamp them in range of [-pi, pi]
            x->operator[](i) = ModPi(x->operator[](i) + dx[i] * alpha);
          }
        });

    // set up optimizer params
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 100;
    p.max_qp_iterations = 5;
    p.relative_exit_tol = tol::kPico;
    p.absolute_first_derivative_tol = 1.0e-10;
    p.absolute_exit_tol = 1.0e-8;
    p.termination_kkt_tolerance = tol::kMicro;
    p.max_line_search_iterations = 5;
    p.line_search_strategy = LineSearchStrategy::ARMIJO_BACKTRACK;
    p.equality_constraint_norm = Norm::L1;
    p.lambda_failure_init = 0.01;
    p.armijo_search_tau = 0.5;  //  backtrack more aggressively

    // We add some non-zero lambda because this problem technically does not have
    // a positive semi-definite hessian (since there is only one nonlinear cost
    // on the effector position).
    p.lambda_initial = 0.001;
    p.min_lambda = 1.0e-9;

    // create a guess
    AlignedVector<Matrix<double, 5, 1>> guesses;
    guesses.push_back(
        (Matrix<double, 5, 1>() << M_PI / 6, -M_PI / 2, M_PI / 6, -M_PI / 2, M_PI / 4).finished());
    guesses.push_back(
        (Matrix<double, 5, 1>() << -M_PI / 4, -M_PI / 4, M_PI / 6, -M_PI / 3, -M_PI / 4)
            .finished());
    guesses.push_back(
        (Matrix<double, 5, 1>() << -M_PI / 3, -M_PI / 2, 0.001, -M_PI / 2, 0.0).finished());

    // solve it
    std::vector<StatCounters> counters{};
    for (const auto& guess : guesses) {
      Logger logger{true, true};
      nls.SetQPLoggingCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));
      nls.SetLoggingCallback(
          [&](const ConstrainedNonlinearLeastSquares& solver, const NLSLogInfo& info) {
            logger.NonlinearSolverCallback(solver, info);
            const auto& vars = solver.variables();
            chain_rear->Update(vars.head<3>());
            chain_front->Update(Vector3d{vars[0], vars[3], vars[4]});
            logger.stream() << fmt::format(
                "  Rear: {}\n",
                chain_rear->translation().head(2).transpose().format(test_utils::kNumPyMatrixFmt));
            logger.stream() << fmt::format(
                "  Front: {}\n",
                chain_front->translation().head(2).transpose().format(test_utils::kNumPyMatrixFmt));
            return true;
          });

      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      ASSERT_EQ(outputs.termination_state, NLSTerminationState::SATISFIED_ABSOLUTE_TOL)
          << logger.GetString();
      counters.push_back(logger.counters());

      // check costs
      for (const ResidualBase::unique_ptr& eq : problem.equality_constraints) {
        ASSERT_NEAR(0.0, eq->QuadraticError(nls.variables()), 1.0e-8);
      }
      for (const ResidualBase::unique_ptr& eq : problem.costs) {
        ASSERT_NEAR(0.0, eq->QuadraticError(nls.variables()), 1.0e-8);
      }

      // No strong reason for 30, just place a max to track performance here.
      ASSERT_LT(logger.GetCount(StatCounters::NUM_LINE_SEARCH_STEPS), 30) << logger.GetString();
    }
    SummarizeCounts("Dual Actuator Balancing", counters);
  }
};

TEST_FIXTURE(ConstrainedNLSTest, TestComputeQPCostDerivative)
TEST_FIXTURE(ConstrainedNLSTest, TestQuadraticApproxMinimum)
TEST_FIXTURE(ConstrainedNLSTest, TestCubicApproxCoeffs)
TEST_FIXTURE(ConstrainedNLSTest, TestComputeSecondOrderCorrection)
TEST_FIXTURE(ConstrainedNLSTest, TestRosenbrock)
TEST_FIXTURE(ConstrainedNLSTest, TestRosenbrockLM)
TEST_FIXTURE(ConstrainedNLSTest, TestInequalityConstrainedRosenbrock)
TEST_FIXTURE(ConstrainedNLSTest, TestInequalityConstrainedRosenbrock6D)
TEST_FIXTURE(ConstrainedNLSTest, TestHimmelblau)
TEST_FIXTURE(ConstrainedNLSTest, TestHimmelblauQuadrantConstrained)
TEST_FIXTURE(ConstrainedNLSTest, TestSphereWithNonlinearEqualityConstraints)
TEST_FIXTURE(ConstrainedNLSTest, TestTwoAngleActuatorChain)
TEST_FIXTURE(ConstrainedNLSTest, TestDualActuatorBalancing)

}  // namespace mini_opt
