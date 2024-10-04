// Copyright 2021 Gareth Cross
#include "mini_opt/qp.hpp"

#include <random>

#include <fmt/ostream.h>
#include <Eigen/Dense>

#include "mini_opt/logging.hpp"
#include "mini_opt/residual.hpp"
#include "test_utils.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif  // _MSC_VER

namespace mini_opt {
using namespace Eigen;
using namespace std::placeholders;

TEST(LinearInequalityConstraintTest, Test) {
  const LinearInequalityConstraint c1(3, 2.0, -4.0);
  ASSERT_TRUE(c1.IsFeasible(2.1));
  ASSERT_FALSE(c1.IsFeasible(1.9));

  const LinearInequalityConstraint shifted = c1.ShiftTo(1.0);
  ASSERT_TRUE(shifted.IsFeasible(1.1));
  ASSERT_FALSE(shifted.IsFeasible(0.9));

  ASSERT_TRUE((Var(0) >= 0.3).IsFeasible(0.5));
  ASSERT_FALSE((Var(0) <= -0.9).IsFeasible(1.2));

  ASSERT_NEAR(0.0, (Var(0) >= 0.0).ClampX(-1.3), tol::kPico);
  ASSERT_NEAR(0.5, (Var(0) >= 0.5).ClampX(-0.9), tol::kPico);
  ASSERT_NEAR(1.5, (Var(0) >= 1.3).ClampX(1.5), tol::kPico);
  ASSERT_NEAR(0.0, (Var(0) <= 0.0).ClampX(5.0), tol::kPico);
  ASSERT_NEAR(-1.3, (Var(0) <= -1.3).ClampX(1.0), tol::kPico);
  ASSERT_NEAR(6.0, (Var(0) <= 10.0).ClampX(6.0), tol::kPico);
}

// Tests for the QP interior point solver.
class QPSolverTest : public ::testing::Test {
 public:
  // Specify the root of a polynomial: (a * x - b)^2
  struct Root {
    double a;
    double b;
    Root(double a, double b) : a(a), b(b) {}
  };

  // Create a quadratic (with diagonal G matrix) for the polynomial with the given roots.
  double BuildQuadratic(const std::vector<Root>& roots, QP* const output) {
    const std::size_t N = roots.size();
    output->G.resize(N, N);
    output->G.setZero();
    output->c.resize(N);
    output->c.setZero();
    std::size_t i = 0;
    double constant = 0;
    for (const Root& root : roots) {
      output->G(i, i) = root.a * root.a;
      output->c[i] = -2 * root.a * root.b;
      constant += root.b * root.b;
      ++i;
    }
    return constant;
  }

  // Variant of the above that operates on an Eigen vector instead.
  double BuildQuadraticVector(const Eigen::VectorXd& roots, QP* const output) {
    std::vector<Root> roots_structs;
    for (int i = 0; i < roots.rows(); ++i) {
      roots_structs.emplace_back(1.0, roots[i]);
    }
    return BuildQuadratic(roots_structs, output);
  }

  static void PutDummyValuesInSlacksAndMultipliers(QPInteriorPointSolver* const solver) {
    // Give all the multipliers different positive non-zero values.
    // The exact values aren't actually important, we just want to validate indexing.
    auto s = QPInteriorPointSolver::SBlock(solver->dims_, solver->variables_);
    auto y = QPInteriorPointSolver::YBlock(solver->dims_, solver->variables_);
    auto z = QPInteriorPointSolver::ZBlock(solver->dims_, solver->variables_);
    for (Index i = 0; i < s.rows(); ++i) {
      s[i] = 2.0 / static_cast<double>(i + 1);
      z[i] = 0.5 * static_cast<double>(i + 1);
    }
    for (Index k = 0; k < y.rows(); ++k) {
      y[k] = static_cast<double>((k + 1) * (k + 1));
    }
  }

  // Check that the solution of the 'augmented system' (which leverages sparsity)
  // matches the full 'brute force' solve that uses LU decomposition.
  void CheckAugmentedSolveAgainstPartialPivot(const QP& qp, const VectorXd& x_guess) {
    // construct the solver
    QPInteriorPointSolver solver(&qp);
    QPInteriorPointSolver::XBlock(solver.dims_, solver.variables_) = x_guess;
    PutDummyValuesInSlacksAndMultipliers(&solver);

    // check the dimensions
    const Index total_dims = solver.dims_.N + solver.dims_.M * 2 + solver.dims_.K;
    ASSERT_EQ(total_dims, solver.variables_.rows());
    ASSERT_EQ(total_dims, solver.delta_.rows());
    ASSERT_EQ(total_dims, solver.r_.rows());
    ASSERT_EQ(static_cast<Index>(solver.dims_.N + solver.dims_.K),
              solver.H_.rows());  //  only x and y

    // build the full system
    MatrixXd H_full;
    VectorXd r_full;
    solver.BuildFullSystem(&H_full, &r_full);

    const PartialPivLU<MatrixXd> piv_lu(H_full);
    ASSERT_GT(std::abs(piv_lu.determinant()), tol::kMicro);

    // compute the update (w/ signs)
    const VectorXd signed_update = piv_lu.solve(-r_full);

    // flip the signs
    VectorXd update = signed_update;
    QPInteriorPointSolver::YBlock(solver.dims_, update) *= -1.0;  // flip dy
    QPInteriorPointSolver::ZBlock(solver.dims_, update) *= -1.0;  // flip dz

    // now solve w/ cholesky
    solver.EvaluateKKTConditions();
    solver.ComputeLDLT();
    solver.SolveForUpdate(0.0 /* mu = 0 */);

    // must match the full system
    ASSERT_EIGEN_NEAR(update, solver.delta_, tol::kPico);
  }

  // Check the solver that ignores inequalities against a problem w/ no inequalities.
  void CheckSolveNoInequalitiesAgainstPartialPivot(const QP& qp, const VectorXd& x_guess) {
    // construct the full solver
    QPInteriorPointSolver solver(&qp);
    QPInteriorPointSolver::XBlock(solver.dims_, solver.variables_) = x_guess;
    PutDummyValuesInSlacksAndMultipliers(&solver);

    // construct the solver with no inequalities and solve
    QP qp_reduced = qp;
    qp_reduced.constraints.clear();
    QPInteriorPointSolver solver_no_ineq(&qp_reduced);
    QPInteriorPointSolver::XBlock(solver.dims_, solver_no_ineq.variables_) = x_guess;
    PutDummyValuesInSlacksAndMultipliers(&solver_no_ineq);

    solver_no_ineq.EvaluateKKTConditions();
    solver_no_ineq.ComputeLDLT();
    solver_no_ineq.SolveForUpdate(0.0 /* mu = 0 */);

    // take the full solver, and call variations of methods that ignore inequalities
    solver.EvaluateKKTConditions(false);
    solver.ComputeLDLT(false);
    solver.SolveForUpdateNoInequalities();

    // these should match
    ASSERT_EIGEN_NEAR(solver.x_block(), solver_no_ineq.x_block(), tol::kPico);
    ASSERT_EIGEN_NEAR(solver.y_block(), solver_no_ineq.y_block(), tol::kPico);
  }

  void TestEliminationNoConstraints() {
    QP qp{};
    BuildQuadratic({Root(0.5, 2.0), Root(5.0, 25.0), Root(3.0, 9.0)}, &qp);

    const VectorXd x_guess = (Matrix<double, 3, 1>() << 0.0, -0.1, -0.3).finished();
    CheckAugmentedSolveAgainstPartialPivot(qp, x_guess);
    CheckSolveNoInequalitiesAgainstPartialPivot(qp, x_guess);
  }

  void TestEliminationEqualityConstraints() {
    QP qp{};
    BuildQuadratic({Root(1.0, -0.5), Root(2.0, -2.0), Root(-4.0, 5.0)}, &qp);

    // add one equality constraint
    qp.A_eq.resize(1, 3);
    qp.b_eq.resize(1);
    qp.A_eq.setZero();
    qp.A_eq(0, 1) = 1.0;
    qp.A_eq(0, 2) = -1.0;
    qp.b_eq(0) = -0.5;

    const VectorXd x_guess = (Matrix<double, 3, 1>() << 0.3, -0.1, -0.3).finished();
    CheckAugmentedSolveAgainstPartialPivot(qp, x_guess);
    CheckSolveNoInequalitiesAgainstPartialPivot(qp, x_guess);
  }

  void TestEliminationInequalityConstraints() {
    QP qp{};
    BuildQuadratic({Root(1.5, 3.0), Root(-1.0, 4.0)}, &qp);

    // set up inequality constraint
    qp.constraints.emplace_back(Var(1) >= 0);   // x >= 0
    qp.constraints.emplace_back(Var(0) <= 5);   // x <= 5
    qp.constraints.emplace_back(Var(0) >= -5);  // x >= -5

    const VectorXd x_guess = (Matrix<double, 2, 1>() << 0.0, 2.0).finished();
    CheckAugmentedSolveAgainstPartialPivot(qp, x_guess);
    CheckSolveNoInequalitiesAgainstPartialPivot(qp, x_guess);
  }

  void TestEliminationAllConstraints() {
    QP qp{};
    const double constant = BuildQuadratic(
        // make a quadratic in 7 variables
        {Root(0.5, 2.0), Root(5.0, 25.0), Root(3.0, 9.0), Root(4.0, 1.0), Root(1.2, 2.4),
         Root(-1.0, 2.0), Root(-0.5, 2.0)},
        &qp);

    // set up equality constraints on three variables (0, 1 and 4)
    // 2*x1 - x4 = -0.5
    // 3*x0 = 2.0
    const Index K = 2;
    qp.A_eq = MatrixXd::Zero(K, qp.G.rows());
    qp.A_eq(0, 1) = 2.0;
    qp.A_eq(0, 4) = -1.0;
    qp.A_eq(1, 0) = 3.0;
    qp.b_eq = Vector2d(0.5, -2.0);

    // set up inequality constraints on three more variables (3, 5 and 6)
    qp.constraints.emplace_back(3, 4.0, -8.0);  // 4x >= 8  -->  4x - 8 >= 0 (x >= 0.5)
    qp.constraints.emplace_back(5, 2.0, 1.0);   // 2x >= -1.0  -->  2x + 1 >= 0 (x >= -0.5)
    qp.constraints.emplace_back(6, 1.0, 0.0);   // x >= 0

    // check that the root is what we think it is
    const VectorXd x_sol =
        (Matrix<double, 7, 1>() << 4.0, 5.0, 3.0, 0.25, 2.0, -2.0, -4.0).finished();
    const double cost = x_sol.transpose() * qp.G * x_sol + x_sol.dot(qp.c) + constant;
    ASSERT_NEAR(0.0, cost, tol::kPico);

    const VectorXd x_guess =
        (Matrix<double, 7, 1>() << 0.0, 0.1, 0.2, 0.55, 0.3, 0.7, 1.0).finished();
    CheckAugmentedSolveAgainstPartialPivot(qp, x_guess);
    CheckSolveNoInequalitiesAgainstPartialPivot(qp, x_guess);
  }

  // Check alpha computation function.
  void TestComputeAlpha() {
    const Eigen::VectorXd x = Vector4d{1.0, 0.8, 1.2, 0.9};
    const Eigen::VectorXd dx = Vector4d{-2.0, 0.6, -1.3, 0.5};
    ASSERT_NEAR(0.5, QPInteriorPointSolver::ComputeAlpha(x.head(3), dx.head(3), 1.0), tol::kPico);
    ASSERT_NEAR(0.45, QPInteriorPointSolver::ComputeAlpha(x.head(3), dx.head(3), 0.9), tol::kPico);
  }

  // Scalar quadratic equation with a single inequality constraint.
  void TestWithSingleInequality() {
    using ScalarMatrix = Matrix<double, 1, 1>;

    // simple quadratic residual: f_0(x) = ||x - 5||^2, h(x) = x - 5
    Residual<1, 1> res;
    res.index = {{0}};
    res.function = [](const ScalarMatrix& x, ScalarMatrix* const J) -> ScalarMatrix {
      if (J) {
        J->setIdentity();
      }
      return x.array() - 5.0;
    };

    // linearize at x=0
    const VectorXd initial_values = VectorXd::Constant(1, 0.0);

    // Set up problem
    QP qp{1};
    res.UpdateHessian(initial_values, &qp.G, &qp.c);
    qp.constraints.emplace_back(Var(0) <= 4);

    QPInteriorPointSolver solver(&qp);
    Logger logger{};
    solver.SetLoggerCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

    QPInteriorPointSolver::Params params{};
    params.termination_kkt_tol = tol::kNano;
    params.sigma = 0.1;
    params.initial_mu = 0.1;
    const QPSolverOutputs outputs = solver.Solve(params);

    // check the solution
    ASSERT_EQ(outputs.termination_state, QPTerminationState::SATISFIED_KKT_TOL)
        << "\nSummary:\n"
        << logger.GetString();
    ASSERT_NEAR(4.0, solver.x_block()[0], tol::kMicro) << "Summary:\n" << logger.GetString();
    ASSERT_NEAR(0.0, solver.s_block()[0], tol::kMicro) << "Summary:\n" << logger.GetString();
    ASSERT_LT(1.0 - tol::kMicro, solver.z_block()[0]) << "Summary:\n" << logger.GetString();
  }

  // Quadratic in two variables w/ two inequalities keep them both from their optimal values.
  void TestWithInequalitiesActive() {
    // Quadratic in two variables. Has a PD diagonal hessian.
    Residual<2, 2> res;
    res.index = {{0, 1}};
    res.function = [](const Matrix<double, 2, 1>& x,
                      Matrix<double, 2, 2>* const J) -> Matrix<double, 2, 1> {
      if (J) {
        J->setZero();
        J->diagonal() = Matrix<double, 2, 1>(1.0, -4.0);
      }
      // solution at (2, -4)
      return Matrix<double, 2, 1>{x[0] - 2.0, -4 * x[1] - 16.0};
    };

    // linearize at x=0
    const VectorXd initial_values = VectorXd::Constant(2, .0);

    // Set up problem
    QP qp{2};
    res.UpdateHessian(initial_values, &qp.G, &qp.c);
    qp.constraints.emplace_back(Var(0) <= 1.0);
    qp.constraints.emplace_back(Var(1) >= -3.0);

    QPInteriorPointSolver solver(&qp);

    for (InitialGuessMethod method :
         {InitialGuessMethod::NAIVE, InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED}) {
      Logger logger{true};
      solver.SetLoggerCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      // solve it
      QPInteriorPointSolver::Params params{};
      params.termination_kkt_tol = tol::kPico;
      params.initial_mu = 0.1;
      params.sigma = 0.1;
      params.initial_guess_method = method;
      const auto outputs = solver.Solve(params);

      // check the solution
      ASSERT_EQ(outputs.termination_state, QPTerminationState::SATISFIED_KKT_TOL)
          << "\nSummary:\n"
          << logger.GetString();
      ASSERT_EIGEN_NEAR(Vector2d(1.0, -3.0), solver.x_block(), tol::kMicro) << "Summary:\n"
                                                                            << logger.GetString();
      ASSERT_EIGEN_NEAR(Vector2d::Zero(), solver.s_block(), tol::kMicro) << "Summary:\n"
                                                                         << logger.GetString();
    }
  }

  // Quadratic in three variables, with one active and one inactive inequality.
  void TestWithInequalitiesPartiallyActive() {
    Residual<3, 3> res;
    res.index = {{0, 1, 2}};
    res.function = [](const Matrix<double, 3, 1>& x,
                      Matrix<double, 3, 3>* const J) -> Matrix<double, 3, 1> {
      if (J) {
        J->setZero();
        J->diagonal() = Matrix<double, 3, 1>(1.0, -1.0, 0.5);
      }
      // solution at [1, -3, -10]
      return Matrix<double, 3, 1>{x[0] - 1.0, -x[1] - 3.0, 0.5 * x[2] + -5.0};
    };

    // Set up problem w/ only one relevant constraint
    QP qp{3};
    res.UpdateHessian(Vector3d::Zero(), &qp.G, &qp.c);
    qp.constraints.emplace_back(Var(1) >= -2.0);
    qp.constraints.emplace_back(Var(0) >= -3.5);  //  irrelevant

    QPInteriorPointSolver solver(&qp);

    for (InitialGuessMethod method :
         {InitialGuessMethod::NAIVE, InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED}) {
      Logger logger{true};
      solver.SetLoggerCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      // solve it
      QPInteriorPointSolver::Params params{};
      params.termination_kkt_tol = tol::kPico;
      params.barrier_strategy = BarrierStrategy::COMPLEMENTARITY;
      params.sigma = 0.1;
      params.initial_mu = 0.1;
      params.initial_guess_method = method;
      const auto term_state = solver.Solve(params).termination_state;

      // check the solution
      ASSERT_EQ(term_state, QPTerminationState::SATISFIED_KKT_TOL) << "\nSummary:\n"
                                                                   << logger.GetString();
      ASSERT_EIGEN_NEAR(Vector3d(1.0, -2.0, 10.0), solver.x_block(), tol::kMicro)
          << "\nSummary:\n"
          << logger.GetString();
      ASSERT_NEAR(0.0, solver.s_block()[0], tol::kMicro) << "Summary:\n" << logger.GetString();
      ASSERT_NEAR(0.0, solver.z_block()[1], tol::kMicro) << "Summary:\n" << logger.GetString();
    }
  }

  // Test simple problem with equality constraints.
  // Should converge in a single step.
  void TestWithEqualitiesOnly() {
    QP qp{};
    BuildQuadratic({Root(1.0, 0.5), Root(3.0, 2.0), Root(-4.0, 5.0), Root(0.25, 4)}, &qp);

    // specify x[0] - x[2]/2 == 3.0
    // specify x[1]/4 + x[3] == -2.0
    qp.A_eq.resize(2, 4);
    qp.b_eq.resize(2);
    qp.A_eq.setZero();
    qp.b_eq.setZero();
    qp.A_eq(0, 0) = 1;
    qp.A_eq(0, 2) = -0.5;
    qp.A_eq(1, 1) = 0.25;
    qp.A_eq(1, 3) = 1.0;
    qp.b_eq[0] = 3.0;
    qp.b_eq[1] = -2.0;

    QPInteriorPointSolver solver(&qp);

    Logger logger{};
    solver.SetLoggerCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

    // solve it
    QPInteriorPointSolver::Params params{};
    params.termination_kkt_tol = tol::kMicro;
    params.max_iterations = 1;  //  should only need one
    const auto term_state = solver.Solve(params).termination_state;

    // should be able to satisfy immediately
    ASSERT_EQ(term_state, QPTerminationState::SATISFIED_KKT_TOL) << "\nSummary:\n"
                                                                 << logger.GetString();
    ASSERT_EIGEN_NEAR(Vector2d::Zero(), qp.A_eq * solver.x_block() + qp.b_eq, tol::kNano)
        << "Summary:\n"
        << logger.GetString();
  }

  // Test a problem where all the variables are locked with equality constraints.
  void TestWithFullyConstrainedEqualities() {
    QP qp{};
    BuildQuadratic({Root(1.0, -0.5), Root(1.0, -0.25), Root(1.0, 1.0)}, &qp);

    // lock all the variables to a specific value (nothing to optimize)
    qp.A_eq = Matrix3d::Identity();
    qp.b_eq = -Vector3d{1., 2., 3.};

    QPInteriorPointSolver solver(&qp);
    Logger logger{};
    solver.SetLoggerCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

    // solve it in a single step
    QPInteriorPointSolver::Params params{};
    params.termination_kkt_tol = tol::kMicro;
    params.max_iterations = 1;
    const auto term_state = solver.Solve(params).termination_state;

    // should be able to satisfy immediately
    ASSERT_EQ(term_state, QPTerminationState::SATISFIED_KKT_TOL) << "\nSummary:\n"
                                                                 << logger.GetString();
    ASSERT_EIGEN_NEAR(-qp.b_eq, solver.x_block(), tol::kNano) << "Summary:\n" << logger.GetString();
    ASSERT_TRUE((solver.y_block().array() > tol::kCenti).all()) << "Summary:\n"
                                                                << logger.GetString();
  }

  // Test with both types of constraint.
  void TestWithInequalitiesAndEqualities() {
    QP qp{};
    BuildQuadratic({Root(1.0, 1.0), Root(5.0, -10.0), Root(10.0, 2.0)}, &qp);

    // lock one of the variables with an equality, x[2] == -2
    qp.A_eq = RowVector3d(0.0, 0.0, 1.0);
    qp.b_eq = Matrix<double, 1, 1>(-2.0);

    // keep both variables from their roots
    qp.constraints.push_back(Var(0) <= 0.5);
    qp.constraints.push_back(Var(1) >= -1.0);

    QPInteriorPointSolver solver(&qp);
    for (InitialGuessMethod method :
         {InitialGuessMethod::NAIVE, InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED}) {
      Logger logger{};
      solver.SetLoggerCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

      QPInteriorPointSolver::Params params{};
      params.termination_kkt_tol = tol::kPico;
      params.initial_mu = 0.1;
      params.sigma = 0.1;
      params.initial_guess_method = method;

      const auto term_state = solver.Solve(params).termination_state;

      // both inequalities should be active
      ASSERT_EQ(term_state, QPTerminationState::SATISFIED_KKT_TOL) << "\nSummary:\n"
                                                                   << logger.GetString();
      ASSERT_EIGEN_NEAR(Vector3d(0.5, -1.0, 2.0), solver.x_block(), tol::kMicro)
          << "\nSummary:\n"
          << logger.GetString();
      ASSERT_EIGEN_NEAR(Vector2d(0.0, 0.0), solver.s_block(), tol::kMicro) << "Summary:\n"
                                                                           << logger.GetString();
    }
  }

  /*
   * This is some nonsense I made up. We generate a bunch of roots of a quadratic with diagonal G,
   * and then scale it by a random positive definite matrix to jumble things up a bit. It's not
   * that principled.
   *
   * This is mostly just to 'poke' the implementation and find issues.
   *
   * Should generate these offline and save them so the random generator
   * can't muck with the results.
   */
  QP GenerateRandomQP(const int seed, const int dimension, const double p_inequality,
                      Eigen::VectorXd* const solution,
                      std::vector<uint8_t>* const constraint_mask) {
    std::default_random_engine engine{static_cast<unsigned int>(seed)};
    std::uniform_real_distribution<double> root_dist{-20.0, 20.0};
    std::uniform_real_distribution<double> ineq_dist{0.1, 0.9};
    std::bernoulli_distribution binomial{p_inequality};

    // generate a bunch of random roots
    VectorXd roots_original(dimension);
    for (int r = 0; r < dimension; ++r) {
      roots_original[r] = root_dist(engine);
    }

    QP qp{};
    BuildQuadraticVector(roots_original, &qp);

    // Generate a random PD matrix and scale the system by it.
    const MatrixXd PD = test_utils::GenerateRandomPDMatrix(dimension, /* seed = */ seed);
    const VectorXd roots_shifted = PD.inverse() * roots_original * 2;
    qp.G = (PD.transpose() * qp.G * PD).eval();
    qp.c = (PD * qp.c).eval();

    solution->noalias() = roots_shifted;
    constraint_mask->resize(dimension, false);

    // put random active inequality constraints
    for (int r = 0; r < roots_shifted.rows(); ++r) {
      if (binomial(engine)) {
        constraint_mask->at(r) = true;
        // this is pretty arbitrary:
        const double scale = ineq_dist(engine);
        if (roots_shifted[r] < 0) {
          qp.constraints.push_back(Var(r) >= roots_shifted[r] * scale);
        } else {
          qp.constraints.push_back(Var(r) <= roots_shifted[r] * scale);
        }
        solution->operator[](r) *= scale;
      }
    }
    return qp;
  }

  // Test a bunch of randomly generated problems.
  void TestGeneratedProblems() {
    const int kNumProblems = 1000;
    const int kProblemDim = 8;  //  size of `x`, for me 8-12 is a typical problem size

    QPInteriorPointSolver solver{};  //  re-use the solver
    std::map<InitialGuessMethod, int> iter_counts;
    for (int p = 0; p < kNumProblems; ++p) {
      VectorXd x_solution;
      std::vector<uint8_t> constraint_mask;
      const QP qp = GenerateRandomQP(p, kProblemDim, 0.5, &x_solution, &constraint_mask);

      // solve it, use the MPC strategy for these problems
      solver.Setup(&qp);

      QPInteriorPointSolver::Params params{};
      params.termination_kkt_tol = tol::kPico;
      params.barrier_strategy = BarrierStrategy::COMPLEMENTARITY;

      // Some of these randomly generated problems start close to the barrier, which causes
      // them to bounce around a bunch before getting close to the solution. Should implement
      // a strategy for this, but for now I'm going to crank this up.
      params.max_iterations = 30;

      for (InitialGuessMethod method :
           {InitialGuessMethod::NAIVE, InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED}) {
        Logger logger{};
        solver.SetLoggerCallback(std::bind(&Logger::QPSolverCallback, &logger, _1, _2, _3, _4));

        params.initial_guess_method = method;
        const QPSolverOutputs outputs = solver.Solve(params);
        iter_counts[method] += outputs.num_iterations;  //  total up # number of iterations

        ASSERT_EIGEN_NEAR(x_solution, solver.x_block(), 5.0e-5)
            << fmt::format("Termination: {}\nProblem p = {}\nSummary:\n{}",
                           fmt::streamed(outputs.termination_state), p, logger.GetString());

        // check the variables that are constrained
        ASSERT_EIGEN_NEAR(Eigen::VectorXd::Zero(qp.constraints.size()), solver.s_block(), 5.0e-5)
            << fmt::format("Termination: {}\nProblem p = {}\nSummary:\n{}",
                           fmt::streamed(outputs.termination_state), p, logger.GetString());
      }
    }
    PRINT(iter_counts[InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED]);
    PRINT(iter_counts[InitialGuessMethod::NAIVE]);
    // this is approximate and might fluctuate a bit, but generally the
    // naive method is ~4x worse.
    ASSERT_LT(iter_counts[InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED] * 4,
              iter_counts[InitialGuessMethod::NAIVE]);
  }
};

TEST_FIXTURE(QPSolverTest, TestEliminationNoConstraints)
TEST_FIXTURE(QPSolverTest, TestEliminationEqualityConstraints)
TEST_FIXTURE(QPSolverTest, TestEliminationInequalityConstraints)
TEST_FIXTURE(QPSolverTest, TestEliminationAllConstraints)
TEST_FIXTURE(QPSolverTest, TestComputeAlpha)
TEST_FIXTURE(QPSolverTest, TestWithSingleInequality)
TEST_FIXTURE(QPSolverTest, TestWithInequalitiesActive)
TEST_FIXTURE(QPSolverTest, TestWithInequalitiesPartiallyActive)
TEST_FIXTURE(QPSolverTest, TestWithEqualitiesOnly)
TEST_FIXTURE(QPSolverTest, TestWithFullyConstrainedEqualities)
TEST_FIXTURE(QPSolverTest, TestWithInequalitiesAndEqualities)
TEST_FIXTURE(QPSolverTest, TestGeneratedProblems)

}  // namespace mini_opt

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
