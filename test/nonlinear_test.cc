// Copyright 2021 Gareth Cross
#include <numeric>
#include <random>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <Eigen/Dense>

#include "mini_opt/nonlinear.hpp"
#include "mini_opt/numerical_jacobian.hpp"

#include "test_utils.hpp"

// TODO(gareth): Split up this file a bit.
namespace mini_opt {
using namespace std::placeholders;
using namespace Eigen;

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
  const Eigen::Matrix<double, ResidualDim, NumParams> J_numerical = numerical_jacobian(
      params, [&](const Eigen::Matrix<double, NumParams, 1>& x) { return function(x, nullptr); });

  ASSERT_EIGEN_NEAR(J_numerical, J, tol);
}

enum class StatCounterName : int32_t {
  NUM_NLS_ITERATIONS = 0,
  NUM_QP_ITERATIONS,
  NUM_FAILED_LINE_SEARCHES,
  NUM_LINE_SEARCH_STEPS,
  MAX_VALUE,
};

std::ostream& operator<<(std::ostream& s, const StatCounterName val) {
  switch (val) {
    case StatCounterName::NUM_FAILED_LINE_SEARCHES:
      s << "NUM_FAILED_LINE_SEARCHES";
      break;
    case StatCounterName::NUM_NLS_ITERATIONS:
      s << "NUM_NLS_ITERATIONS";
      break;
    case StatCounterName::NUM_QP_ITERATIONS:
      s << "NUM_QP_ITERATIONS";
      break;
    case StatCounterName::NUM_LINE_SEARCH_STEPS:
      s << "NUM_LINE_SEARCH_STEPS";
      break;
    case StatCounterName::MAX_VALUE:
      s << "<INVALID VALUE>";
      break;
  }
  return s;
}

// Accumulate counts for testing.
struct StatCounters {
  // All the counts.
  std::array<std::size_t, static_cast<std::size_t>(StatCounterName::MAX_VALUE)> counts{};

  StatCounters() noexcept { counts.fill(0); }

  std::size_t at(const StatCounterName name) const {
    return counts[static_cast<std::size_t>(name)];
  }

  std::size_t& at(const StatCounterName name) { return counts[static_cast<std::size_t>(name)]; }

  StatCounters operator+(const StatCounters& c) noexcept {
    StatCounters out = *this;
    out += c;
    return out;
  }

  constexpr StatCounters& operator+=(const StatCounters& c) noexcept {
    for (std::size_t i = 0; i < counts.size(); ++i) {
      counts[i] += c.counts[i];
    }
    return *this;
  }

  explicit StatCounters(const NLSSolverOutputs& outputs) noexcept : StatCounters() {
    at(StatCounterName::NUM_NLS_ITERATIONS) = outputs.iterations.size();
    at(StatCounterName::NUM_QP_ITERATIONS) = outputs.NumQPIterations();
    at(StatCounterName::NUM_FAILED_LINE_SEARCHES) = outputs.NumFailedLineSearches();
    at(StatCounterName::NUM_LINE_SEARCH_STEPS) = outputs.NumLineSearchSteps();
  }
};

// Test constrained non-linear least squares.
class ConstrainedNLSTest : public ::testing::Test {
 public:
  // Test function that computes the gradient of the cost function against
  // a numerical version.
  void TestComputeQPCostDerivative() {
    using Vector5 = Matrix<double, 5, 1>;

    auto cost_func = [](const Vector5& params,
                        Matrix<double, 3, 5>* J_out) -> Matrix<double, 3, 1> {
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

    Residual cost = MakeResidual<3, 5>({0, 2, 1, 4, 3}, cost_func);

    // Make up an equality constraint
    auto eq_func = [](const Vector3d& params, Matrix<double, 2, 3>* J_out) -> Vector2d {
      Vector2d result{params[0] - params[1], -params[2] * params[2]};
      if (J_out) {
        J_out->setZero();
        J_out->operator()(0, 0) = 1;
        J_out->operator()(0, 1) = -1;
        J_out->operator()(1, 2) = -2 * params[2];
      }
      return result;
    };

    Residual eq = MakeResidual<2, 3>({3, 4, 0}, eq_func);

    // check that the derivatives are correct
    const Vector5 params = (Vector5() << -0.5, 0.2, 0.3, -0.8, 1.2).finished();
    TestResidualFunctionDerivative<3, 5>(cost_func, params);
    TestResidualFunctionDerivative<2, 3>(eq_func, params.head<3>());

    // pick a direction for the directional derivative
    const VectorXd dx = (Vector5() << 0.1, 0.25, -0.87, 1.1, -0.02).finished();

    // The penalty on the equality constraint
    constexpr double penalty = 0.334;

    // test with L1
    // compute the derivative of the sum cost function
    const Matrix<double, 1, 1> J_numerical = numerical_jacobian(0.0, [&](const double alpha) {
      Eigen::VectorXd cost_out(3);
      Eigen::VectorXd equality_out(2);
      cost.ErrorVector(params + dx * alpha, cost_out.head(3));
      eq.ErrorVector(params + dx * alpha, equality_out.head(2));
      return 0.5 * cost_out.squaredNorm() + penalty * equality_out.lpNorm<1>();
    });

    // set up a problem
    Problem problem{};
    problem.dimension = 5;
    problem.costs.emplace_back(std::move(cost));
    problem.equality_constraints.emplace_back(std::move(eq));

    // compute analytically as well
    QP qp{static_cast<Index>(problem.dimension)};
    qp.A_eq.resize(2, 5);
    qp.b_eq.resize(2);
    ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(params, 0.0, problem, &qp);

    const DirectionalDerivatives J_analytical =
        ConstrainedNonlinearLeastSquares::ComputeQPCostDerivative(qp, dx.head(5));
    ASSERT_NEAR(J_numerical[0], J_analytical.Total(penalty), tol::kPico);
  }

  void TestQuadraticApproxMinimum() {
    // make up some values
    constexpr double alpha_0 = 0.8;
    constexpr double phi_0 = 2.0;
    constexpr double phi_prime_0 = -1.2;
    constexpr double phi_alpha_0 = 2.2;

    // compute via form 1
    const std::optional<double> solution = ConstrainedNonlinearLeastSquares::QuadraticApproxMinimum(
        phi_0, phi_prime_0, alpha_0, phi_alpha_0);
    ASSERT_TRUE(solution);

    const double a = (phi_alpha_0 - phi_0 - alpha_0 * phi_prime_0) / std::pow(alpha_0, 2);
    const double b = phi_prime_0;
    ASSERT_NEAR(-b / (2 * a), solution.value(), tol::kPico);

    // Try invalid solutions:
    // phi_prime_0 > 0
    ASSERT_FALSE(
        ConstrainedNonlinearLeastSquares::QuadraticApproxMinimum(phi_0, 1.3, alpha_0, phi_alpha_0));
    // phi_alpha_0 > phi_prime_0 * alpha_0 - phi_0
    ASSERT_FALSE(ConstrainedNonlinearLeastSquares::QuadraticApproxMinimum(
        phi_0, 1.3, alpha_0, (phi_prime_0 * alpha_0 - phi_0) * 1.01));
  }

  // Check that close form cubic approximation is correct.
  void TestCubicApproxCoeffs() {
    constexpr double alpha_0 = 0.8;
    constexpr double alpha_1 = 0.4;
    constexpr double phi_0 = 1.44;
    constexpr double phi_prime_0 = -1.23;

    constexpr double phi_alpha_0 = 2.2;  //  cost did not decrease
    constexpr double phi_alpha_1 = 1.6;  //  still did not decrease

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
    const std::optional<double> min_alpha =
        ConstrainedNonlinearLeastSquares::CubicApproxMinimum(phi_prime_0, ab);
    ASSERT_TRUE(min_alpha);

    // check it actually is the minimum
    ASSERT_NEAR(
        0.0,
        3 * ab[0] * std::pow(min_alpha.value(), 2) + 2 * ab[1] * min_alpha.value() + phi_prime_0,
        tol::kPico);
    ASSERT_GT(6 * ab[0] * min_alpha.value() + 2 * ab[1], 0);

    // create an invalid scenario:
    ASSERT_FALSE(ConstrainedNonlinearLeastSquares::CubicApproxMinimum(phi_prime_0, {0.0, ab[1]}));

    //  b * b - 3 * a * phi_prime_0 --> a > b*b/(3 * phi_prime_0)
    ASSERT_FALSE(ConstrainedNonlinearLeastSquares::CubicApproxMinimum(
        phi_prime_0, {(ab[1] * ab[1] / (3 * phi_prime_0)) * 1.01, ab[1]}));
  }

  void ComputeSecondOrderCorrection(const Eigen::VectorXd& updated_x,
                                    const std::vector<Residual>& equality_constraints, QP* qp,
                                    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>* solver,
                                    Eigen::VectorXd* dx_out) {
    F_ASSERT(qp);
    F_ASSERT(solver);
    F_ASSERT(dx_out);

    // we use the QP `b` vector as storage for this operation
    int row = 0;
    for (const Residual& eq : equality_constraints) {
      const int dim = eq.Dimension();
      F_ASSERT_LE(row + dim, qp->b_eq.rows(), "Insufficient rows in vector b");
      eq.ErrorVector(updated_x, qp->b_eq.segment(row, dim));
      row += dim;
    }

    // compute the pseudo-inverse
    solver->compute(qp->A_eq);
    dx_out->noalias() -= solver->solve(qp->b_eq);
  }

  void TestComputeSecondOrderCorrection() {
    // create a non-linear equality constraint
    auto func = [](const Vector3d& x, Matrix<double, 2, 3>* J) -> Vector2d {
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
    TestResidualFunctionDerivative<2, 3>(func, x_lin);

    // make space for linearized problem
    QP qp{};
    qp.A_eq.resize(2, 3);
    qp.b_eq.resize(2);
    qp.G.resize(3, 3);  //  unused but needs to be allocated
    qp.c.resize(3, 1);  //  unused but needs to be allocated

    // linearize it
    Problem problem{};
    problem.dimension = 3;
    problem.equality_constraints.emplace_back(MakeResidual<2, 3>({0, 1, 2}, func));
    ConstrainedNonlinearLeastSquares::LinearizeAndFillQP(x_lin, 0.0, problem, &qp);

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
    ASSERT_NEAR(0.0, numerical_derivative(0.0, 0.01, norm_func), tol::kPico);

    const auto der_func = [&](double alpha) {
      return numerical_derivative(alpha, 0.01, norm_func);
    };
    ASSERT_GT(numerical_derivative(0.0, 0.01, der_func), 0.0);

    // shift by that step and re-evaluate the nonlinear constraint
    Eigen::VectorXd dx_in(3);
    dx_in = dx_original + null_space * 5.334;  // a hypothetical update step we might have computed

    Eigen::VectorXd dx_out(3);
    dx_out.setZero();
    ComputeSecondOrderCorrection(x_lin + dx_in, problem.equality_constraints, &qp, &decomposition,
                                 &dx_out);
    PRINT_MATRIX(dx_out.transpose());

    // should satisfy: A * dx_out + c(x + dx_in)
    ASSERT_EIGEN_NEAR(qp.A_eq * dx_out + func(x_lin + dx_in, nullptr), Vector2d::Zero(),
                      tol::kPico);

    // error should be reduced if we re-evaluate
    PRINT_MATRIX(func(x_lin + dx_in, nullptr).transpose());
    PRINT_MATRIX(func(x_lin + dx_in + dx_out, nullptr).transpose());
    ASSERT_GT(func(x_lin + dx_in, nullptr).norm(), func(x_lin + dx_in + dx_out, nullptr).norm());
  }

  static void SummarizeCounts(std::string_view name, const std::vector<StatCounters>& counters) {
    ASSERT_GT(counters.size(), 0u);
    // get all the stats and dump them
    fmt::print("Stats  from {} trials.\n", counters.size());
    for (const auto stat_name :
         {StatCounterName::NUM_NLS_ITERATIONS, StatCounterName::NUM_QP_ITERATIONS,
          StatCounterName::NUM_FAILED_LINE_SEARCHES, StatCounterName::NUM_LINE_SEARCH_STEPS}) {
      std::vector<std::size_t> sorted;
      std::transform(counters.begin(), counters.end(), std::back_inserter(sorted),
                     [&stat_name](const StatCounters& c) { return c.at(stat_name); });
      std::sort(sorted.begin(), sorted.end());
      const auto total = std::accumulate(sorted.begin(), sorted.end(), static_cast<std::size_t>(0));
      const std::size_t num = sorted.size();
      fmt::print(
          "Iteration counts for [{}], {}:\n"
          "  Mean: {}\n"
          "  Median: {}\n"
          "  Max: {}\n"
          "  Min: {}\n"
          "  95 percentile: {}\n",
          name, fmt::streamed(stat_name), total / static_cast<double>(num), sorted[num / 2],
          sorted.back(), sorted.front(), sorted[(num * 95) / 100]);
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
    // check that it behaves correctly
    TestResidualFunctionDerivative<2, 2>(&ConstrainedNLSTest::Rosenbrock, Vector2d{5, -3});
    TestResidualFunctionDerivative<2, 2>(&ConstrainedNLSTest::Rosenbrock, Vector2d{1, 1});
    ASSERT_EIGEN_NEAR(Vector2d::Zero(), Rosenbrock({1, 1}), tol::kPico);

    // simple problem with only one cost
    Problem problem{};
    problem.costs.push_back(MakeResidual<2, 2>({0, 1}, &ConstrainedNLSTest::Rosenbrock));
    problem.dimension = 2;

    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 5;
    p.max_qp_iterations =
        1;  //  since this is quadratic w/ no constrains, should only need one of these

    // Solve it from a few different initial guesses.
    // These guesses aren't that principled, I kind of picked at random.
    const std::vector<Vector2d> initial_guesses = {{-5, -3},  {10, 8},     {-20, 3},
                                                   {0, -5},   {4, 0},      {100, 50},
                                                   {-35, 40}, {1000, -50}, {0.8, -0.3}};
    for (const Vector2d& guess : initial_guesses) {
      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      ASSERT_EQ(outputs.termination_state, NLSTerminationState::SATISFIED_ABSOLUTE_TOL);
      ASSERT_EQ(outputs.NumQPIterations(), outputs.iterations.size());

      // check solution
      ASSERT_EIGEN_NEAR(Vector2d::Ones(), nls.variables(), tol::kMicro)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));
    }
  }

  // Solve un-constrained rosenbrock, but don't use line search. Instead, depend on LM.
  void TestRosenbrockLM() {
    // simple problem with only one cost
    Problem problem{};
    problem.costs.emplace_back(MakeResidual<2, 2>({0, 1}, &ConstrainedNLSTest::Rosenbrock));
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
    const std::vector<Vector2d> initial_guesses = {{-5, -3},  {10, 8},     {-20, 3},
                                                   {0, -5},   {4, 0},      {100, 50},
                                                   {-35, 40}, {1000, -50}, {0.8, -0.3}};
    for (const Vector2d& guess : initial_guesses) {
      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      ASSERT_EQ(outputs.termination_state, NLSTerminationState::SATISFIED_ABSOLUTE_TOL);
      ASSERT_EQ(outputs.NumQPIterations(), outputs.iterations.size());

      // check solution
      ASSERT_EIGEN_NEAR(Vector2d::Ones(), nls.variables(), tol::kMicro)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));
    }
  }

  // Test rosenbrock w/ inequality constraints about the optimum.
  void TestInequalityConstrainedRosenbrock() {
    // simple problem with only one cost
    Problem problem{};
    problem.costs.emplace_back(MakeResidual<2, 2>({0, 1}, &ConstrainedNLSTest::Rosenbrock));
    problem.dimension = 2;
    problem.inequality_constraints.push_back(Var(0) >= 1.2);
    problem.inequality_constraints.push_back(Var(1) <= 0.5);

    ConstrainedNonlinearLeastSquares nls(&problem);
    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 10;
    p.max_qp_iterations = 10;

    // Solve it from a few different initial guesses
    // The last three are actually infeasible to begin with.
    const std::vector<Vector2d> initial_guesses = {{12, -5},      {100.0, -20.0}, {1423.0, -400.0},
                                                   {-20.0, 10.0}, {-120.0, 35.0}, {-50.0, 0.5}};
    std::vector<StatCounters> counters;
    for (const Vector2d& guess : initial_guesses) {
      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.emplace_back(outputs);

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE((outputs.termination_state != NLSTerminationState::MAX_ITERATIONS) &&
                  (outputs.termination_state != NLSTerminationState::MAX_LAMBDA))
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));

      // check solution, it should be at the constraint
      ASSERT_EIGEN_NEAR(Vector2d(1.2, 0.5), nls.variables(), tol::kMicro)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));
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
    using Vector6 = Matrix<double, 6, 1>;
    TestResidualFunctionDerivative<10, 6>(&ConstrainedNLSTest::Rosenbrock6D, Vector6::Zero());
    TestResidualFunctionDerivative<10, 6>(&ConstrainedNLSTest::Rosenbrock6D, Vector6::Ones());

    // simple problem with only one cost
    Problem problem{};
    problem.costs.push_back(
        MakeResidual<10, 6>({0, 1, 2, 3, 4, 5}, &ConstrainedNLSTest::Rosenbrock6D));
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
    p.relative_exit_tol = tol::kMicro;
    p.absolute_first_derivative_tol = 5 * tol::kMicro;
    p.termination_kkt_tolerance = tol::kMicro;
    p.max_lambda = 10.0;

    // Solve it from a couple of different initial guesses
    const Vector6 guess0 = (Vector6() << 10.5, -8.0, 50., -14.0, 4.0, -0.6).finished();
    const Vector6 guess1 = (Vector6() << 100.0, -50.0, 30.0, -100.0, 150.0, -400.0).finished();

    // TODO(gareth): Check this value more thoroughly. :S
    const Vector6 solution =
        (Vector6() << 2.3, -1.2, 3.0, -2.5, 6.19802, std::pow(6.19802, 2)).finished();

    std::vector<StatCounters> counters;
    for (const Vector6& guess : {guess0, guess1}) {
      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.emplace_back(outputs);

      // we can terminate due to absolute tol, derivative tol, etc
      EXPECT_TRUE(TerminationStateIndicatesSatisfiedTol(outputs.termination_state))
          << fmt::format("Termination state: {}\nInitial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(outputs.termination_state),
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));

      // check solution, it should be at the constraint
      ASSERT_EIGEN_NEAR(solution, nls.variables(), 1.0e-5)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));
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
    problem.costs.push_back(MakeResidual<1, 2>({0, 1}, &ConstrainedNLSTest::Himmelblau1));
    problem.costs.push_back(MakeResidual<1, 2>({0, 1}, &ConstrainedNLSTest::Himmelblau2));
    problem.dimension = 2;

    // first test version bounded to [-5, 5]...
    problem.inequality_constraints.push_back(Var(0) >= -5.0);
    problem.inequality_constraints.push_back(Var(0) <= 5.0);
    problem.inequality_constraints.push_back(Var(1) >= -5.0);
    problem.inequality_constraints.push_back(Var(1) <= 5.0);

    ConstrainedNonlinearLeastSquares nls(&problem);

    // From wikipedia, should get more accurate values for these
    std::vector<Vector2d> valid_solutions;
    valid_solutions.emplace_back(3.0, 2.0);
    valid_solutions.emplace_back(-2.805118, 3.131312);
    valid_solutions.emplace_back(-3.779310, -3.283186);
    valid_solutions.emplace_back(3.584428, -1.848126);
    for (const Vector2d& sol : valid_solutions) {
      ASSERT_NEAR(0.0, nls.EvaluateNonlinearErrors(sol).Total(1.), tol::kMicro);
    }

    ConstrainedNonlinearLeastSquares::Params p{};
    p.max_iterations = 20;
    p.max_qp_iterations = 10;
    p.relative_exit_tol = tol::kPico;
    p.absolute_first_derivative_tol = tol::kNano * 10;
    p.termination_kkt_tolerance = tol::kMicro;

    // generate a bunch of initial guesses
    std::vector<Vector2d> initial_guesses;
    for (double x = -4.5; x <= 4.5; x += 0.3) {    // NOLINT(cert-flp30-c)
      for (double y = -4.5; y <= 4.5; y += 0.3) {  // NOLINT(cert-flp30-c)
        initial_guesses.emplace_back(x, y);
      }
    }

    std::vector<StatCounters> counters;
    for (const Vector2d& guess : initial_guesses) {
      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.emplace_back(outputs);

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE(TerminationStateIndicatesSatisfiedTol(outputs.termination_state))
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));

      // one of the solutions should match
      const Vector2d best_sol = *std::min_element(
          valid_solutions.begin(), valid_solutions.end(), [&](const auto& v1, const auto& v2) {
            return (v1 - nls.variables()).norm() < (v2 - nls.variables()).norm();
          });

      ASSERT_EIGEN_NEAR(best_sol, nls.variables(), 5.0e-5) << fmt::format(
          "Initial guess: {}\nSummary:\n{}\n",
          fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)), outputs.ToString());
    }
    SummarizeCounts("Himmelblau", counters);
  }

  // Himmelblau but constrained to one global optimum.
  void TestHimmelblauQuadrantConstrained() {
    // break problem into 2 costs
    Problem problem{};
    problem.costs.push_back(MakeResidual<1, 2>({0, 1}, &ConstrainedNLSTest::Himmelblau1));
    problem.costs.push_back(MakeResidual<1, 2>({0, 1}, &ConstrainedNLSTest::Himmelblau2));
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
    std::vector<Vector2d> initial_guesses;
    for (double x = 0.2; x <= 4.8; x += 0.2) {    // NOLINT(cert-flp30-c)
      for (double y = 0.2; y <= 4.8; y += 0.2) {  // NOLINT(cert-flp30-c)
        initial_guesses.emplace_back(x, y);
      }
    }

    std::vector<StatCounters> counters;
    for (const Vector2d& guess : initial_guesses) {
      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.emplace_back(outputs);

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE(TerminationStateIndicatesSatisfiedTol(outputs.termination_state))
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));

      // should match this solution well
      ASSERT_EIGEN_NEAR(Vector2d(3.0, 2.0), nls.variables(), 5.0e-5)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));
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
    problem.costs.push_back(MakeResidual<N, N>(index, &ConstrainedNLSTest::SphereFunction<N>));
    problem.dimension = N;

    problem.equality_constraints.push_back(
        MakeResidual<1, 2>({0, 1}, std::bind(&ConstrainedNLSTest::ProductExpression, _1, 4.0, _2)));
    problem.equality_constraints.push_back(
        MakeResidual<1, 2>({2, 3}, std::bind(&ConstrainedNLSTest::ProductExpression, _1, 9.0, _2)));

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
    std::vector<Matrix<double, N, 1>> guesses;
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
    std::vector<Matrix<double, N, 1>> solutions;
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
      // solve it
      const NLSSolverOutputs outputs = nls.Solve(p, guess);
      counters.emplace_back(outputs);

      // we can terminate due to absolute tol, derivative tol, etc
      ASSERT_TRUE(TerminationStateIndicatesSatisfiedTol(outputs.termination_state))
          << fmt::format("Termination: {}\nInitial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(outputs.termination_state),
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));

      // find whichever of the 4 best solutions we found
      const auto min_it =
          std::min_element(solutions.begin(), solutions.end(), [&](const auto& a, const auto& b) {
            return (a - nls.variables()).squaredNorm() < (b - nls.variables()).squaredNorm();
          });

      ASSERT_EIGEN_NEAR(*min_it, nls.variables(), 5.0e-5)
          << fmt::format("Initial guess: {}\nSummary:\n{}\n",
                         fmt::streamed(guess.transpose().format(test_utils::kNumPyMatrixFmt)),
                         outputs.ToString(true));

      ASSERT_EQ(0, counters.back().at(StatCounterName::NUM_FAILED_LINE_SEARCHES))
          << outputs.ToString(true);
    }
    SummarizeCounts("Sphere With Nonlinear Equalities", counters);
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

}  // namespace mini_opt
