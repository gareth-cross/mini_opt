// Copyright 2020 Gareth Cross
#include <numeric>
#include <random>

#include "nonlinear.hpp"
#include "numerical_derivative.hpp"
#include "qp.hpp"
#include "test_utils.hpp"
#include "transform_chains.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif  // _MSC_VER

namespace mini_opt {
using namespace Eigen;

// Tests for the QP interior point solver.
class QPSolverTest : public ::testing::Test {
 public:
  using TerminationState = QPInteriorPointSolver::TerminationState;

  // Specify the root of a polynominal: (a * x - b)^2
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

  // Check that the solution of the 'augmented system' (which leverages sparsity)
  // matches the full 'brute force' solve that uses LU decomposition.
  void CheckAugmentedSolveAgainstPartialPivot(const QP& qp, const VectorXd& x_guess) {
    // construct the solver
    QPInteriorPointSolver solver(qp);
    QPInteriorPointSolver::XBlock(solver.dims_, solver.variables_) = x_guess;

    // Give all the multipliers different positive non-zero values.
    // The exact values aren't actually important, we just want to validate indexing.
    auto s = QPInteriorPointSolver::SBlock(solver.dims_, solver.variables_);
    auto y = QPInteriorPointSolver::YBlock(solver.dims_, solver.variables_);
    auto z = QPInteriorPointSolver::ZBlock(solver.dims_, solver.variables_);
    for (Index i = 0; i < s.rows(); ++i) {
      s[i] = 2.0 / (i + 1);
      z[i] = 0.5 * (i + 1);
    }
    for (Index k = 0; k < y.rows(); ++k) {
      y[k] = static_cast<double>((k + 1) * (k + 1));
    }

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

  void TestSolveNoConstraints() {
    QP qp{};
    BuildQuadratic({Root(0.5, 2.0), Root(5.0, 25.0), Root(3.0, 9.0)}, &qp);

    const VectorXd x_guess = (Matrix<double, 3, 1>() << 0.0, -0.1, -0.3).finished();
    CheckAugmentedSolveAgainstPartialPivot(qp, x_guess);
  }

  void TestSolveEqualityConstraints() {
    QP qp{};
    BuildQuadratic({Root(1.0, -0.5), Root(2.0, -2.0), Root(-4.0, 5.0)}, &qp);

    // add one equality constraint
    qp.A_eq.resize(1, 3);
    qp.b_eq.resize(1);
    qp.A_eq(0, 1) = 1.0;
    qp.A_eq(0, 2) = -1.0;
    qp.b_eq(0) = -0.5;

    const VectorXd x_guess = (Matrix<double, 3, 1>() << 0.3, -0.1, -0.3).finished();
    CheckAugmentedSolveAgainstPartialPivot(qp, x_guess);
  }

  void TestSolveInequalityConstraints() {
    QP qp{};
    BuildQuadratic({Root(1.5, 3.0), Root(-1.0, 4.0)}, &qp);

    // set up inequality constraint
    qp.constraints.emplace_back(1, 1.0, 0.0);  // x >= 0

    const VectorXd x_guess = (Matrix<double, 2, 1>() << 0.0, 2.0).finished();
    CheckAugmentedSolveAgainstPartialPivot(qp, x_guess);
  }

  void TestSolveAllConstraints() {
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
  }

  static void ProgressPrinterNoState(const QPInteriorPointSolver& solver,
                                     const QPInteriorPointSolver::KKTError& kkt2_prev,
                                     const QPInteriorPointSolver::KKTError& kkt2_after,
                                     const QPInteriorPointSolver::IterationOutputs& outputs) {
    (void)solver;
    std::cout << "Iteration summary: ";
    std::cout << "||kkt||^2: " << kkt2_prev.Total() << " --> " << kkt2_after.Total()
              << ", mu = " << outputs.mu << ", sigma = " << outputs.sigma
              << ", a_p = " << outputs.alpha.primal << ", a_d = " << outputs.alpha.dual << "\n";

    std::cout << "KKT errors:\n";
    std::cout << "  r_dual = " << kkt2_prev.r_dual << " --> " << kkt2_after.r_dual << "\n";
    std::cout << "  r_comp = " << kkt2_prev.r_comp << " --> " << kkt2_after.r_comp << "\n";
    std::cout << "  r_p_eq = " << kkt2_prev.r_primal_eq << " --> " << kkt2_after.r_primal_eq
              << "\n";
    std::cout << "  r_p_ineq = " << kkt2_prev.r_primal_ineq << " --> " << kkt2_after.r_primal_ineq
              << "\n";
  }

  // TODO(gareth): Would really like to use libfmt for this instead...
  static void ProgressPrinter(const QPInteriorPointSolver& solver,
                              const QPInteriorPointSolver::KKTError& kkt2_prev,
                              const QPInteriorPointSolver::KKTError& kkt2_after,
                              const QPInteriorPointSolver::IterationOutputs& outputs) {
    ProgressPrinterNoState(solver, kkt2_prev, kkt2_after, outputs);
    // dump the state with labels
    std::cout << "After update:\n";
    std::cout << "  x = " << solver.x_block().transpose().format(test_utils::kNumPyMatrixFmt)
              << "\n";
    std::cout << "  s = " << solver.s_block().transpose().format(test_utils::kNumPyMatrixFmt)
              << "\n";
    std::cout << "  y = " << solver.y_block().transpose().format(test_utils::kNumPyMatrixFmt)
              << "\n";
    std::cout << "  z = " << solver.z_block().transpose().format(test_utils::kNumPyMatrixFmt)
              << "\n";

    // summarize the inequality constraints
    std::cout << "Constraints:\n";
    std::size_t i = 0;
    for (const LinearInequalityConstraint& c : solver.problem().constraints) {
      std::cout << "  Constraint " << i << ": ax[" << c.variable
                << "] + b - s == " << c.a * solver.x_block()[c.variable] + c.b - solver.s_block()[i]
                << "  (" << c.a << " * " << solver.x_block()[c.variable] << " + " << c.b << " - "
                << solver.s_block()[i] << ")\n";
      ++i;
    }
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

    // linearize at x=0 (it's already linear, in reality)
    const VectorXd initial_values = VectorXd::Constant(1, 0.0);

    // Set up problem
    QP qp{1};
    res.UpdateHessian(initial_values, &qp.G, &qp.c);
    qp.constraints.emplace_back(Var(0) <= 4);

    QPInteriorPointSolver solver(qp);
    solver.SetLoggerCallback(&QPSolverTest::ProgressPrinter);

    QPInteriorPointSolver::Params params{};
    params.termination_kkt2_tol = tol::kPico;
    const auto term_state = solver.Solve(params);

    // check the solution
    ASSERT_TRUE(term_state == TerminationState::SATISFIED_KKT_TOL) << term_state;
    ASSERT_NEAR(0.0, solver.r_.squaredNorm(), tol::kPico);
    ASSERT_NEAR(4.0, solver.x_block()[0], tol::kMicro);
    ASSERT_NEAR(0.0, solver.s_block()[0], tol::kMicro);
    ASSERT_LT(1.0 - tol::kMicro, solver.z_block()[0]);
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

    QPInteriorPointSolver solver(qp);
    solver.SetLoggerCallback(&QPSolverTest::ProgressPrinter);

    // solve it
    QPInteriorPointSolver::Params params{};
    params.termination_kkt2_tol = tol::kPico;
    const auto term_state = solver.Solve(params);

    // check the solution
    ASSERT_TRUE(term_state == TerminationState::SATISFIED_KKT_TOL) << term_state;
    ASSERT_EIGEN_NEAR(Vector2d(1.0, -3.0), solver.x_block(), tol::kMicro);
    ASSERT_EIGEN_NEAR(Vector2d::Zero(), solver.s_block(), 1.0e-8);
    ASSERT_TRUE((solver.z_block().array() > 1).all());
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

    QPInteriorPointSolver solver(qp);
    solver.SetLoggerCallback(&QPSolverTest::ProgressPrinter);

    // solve it
    QPInteriorPointSolver::Params params{};
    params.termination_kkt2_tol = tol::kPico;
    const auto term_state = solver.Solve(params);

    // check the solution
    ASSERT_TRUE(term_state == TerminationState::SATISFIED_KKT_TOL) << term_state;
    ASSERT_EIGEN_NEAR(Vector3d(1.0, -2.0, 10.0), solver.x_block(), tol::kMicro);
    ASSERT_NEAR(0.0, solver.s_block()[0], tol::kMicro);  // first constraint is active
    ASSERT_NEAR(0.0, solver.z_block()[1], tol::kMicro);  // second constraint is inactive
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
    qp.A_eq(0, 0) = 1;
    qp.A_eq(0, 2) = -0.5;
    qp.A_eq(1, 1) = 0.25;
    qp.A_eq(1, 3) = 1.0;
    qp.b_eq[0] = 3.0;
    qp.b_eq[1] = -2.0;

    QPInteriorPointSolver solver(qp);
    solver.SetLoggerCallback(&QPSolverTest::ProgressPrinter);

    // solve it
    QPInteriorPointSolver::Params params{};
    params.termination_kkt2_tol = tol::kPico;
    params.max_iterations = 1;  //  should only need one
    const auto term_state = solver.Solve(params);
    PRINT(term_state);

    // shoudl be able to satisfy immediately
    ASSERT_TRUE(term_state == TerminationState::SATISFIED_KKT_TOL);
    ASSERT_EIGEN_NEAR(Vector2d::Zero(), qp.A_eq * solver.x_block() + qp.b_eq, tol::kNano);
  }

  // Test a problem where all the variables are locked with equality constraints.
  void TestWithFullyConstrainedEqualities() {
    QP qp{};
    BuildQuadratic({Root(1.0, -0.5), Root(1.0, -0.25), Root(1.0, 1.0)}, &qp);

    // lock all the variables to a specific value (nothing to optimized)
    qp.A_eq = Matrix3d::Identity();
    qp.b_eq = -Vector3d{1., 2., 3.};

    QPInteriorPointSolver solver(qp);
    solver.SetLoggerCallback(&QPSolverTest::ProgressPrinter);

    // solve it in a single step
    QPInteriorPointSolver::Params params{};
    params.termination_kkt2_tol = tol::kPico;
    params.max_iterations = 1;
    const auto term_state = solver.Solve(params);

    // should be able to satisfy immediately
    ASSERT_TRUE(term_state == TerminationState::SATISFIED_KKT_TOL) << term_state;
    ASSERT_EIGEN_NEAR(-qp.b_eq, solver.x_block(), tol::kNano);
    ASSERT_TRUE((solver.y_block().array() > tol::kCenti).all());
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

    QPInteriorPointSolver solver(qp);
    PRINT_MATRIX(solver.variables_.transpose());
    solver.SetLoggerCallback(&QPSolverTest::ProgressPrinter);

    solver.EvaluateKKTConditions();
    PRINT_MATRIX(solver.r_.transpose());

    // solve it in a single step
    QPInteriorPointSolver::Params params{};
    params.termination_kkt2_tol = tol::kPico;
    params.initial_sigma = 1;
    params.sigma_reduction = 0.5;

    const auto term_state = solver.Solve(params);

    // both inequalities should be active
    ASSERT_TRUE(term_state == TerminationState::SATISFIED_KKT_TOL) << term_state;
    ASSERT_EIGEN_NEAR(Vector3d(0.5, -1.0, 2.0), solver.x_block(), tol::kMicro);
    ASSERT_EIGEN_NEAR(Vector2d(0.0, 0.0), solver.s_block(), tol::kMicro);
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
    for (int p = 0; p < kNumProblems; ++p) {
      VectorXd x_solution;
      std::vector<uint8_t> constraint_mask;
      const QP qp = GenerateRandomQP(p, kProblemDim, 0.5, &x_solution, &constraint_mask);

      // solve it, use the MPC strategy for these ones
      QPInteriorPointSolver solver(qp);
      QPInteriorPointSolver::Params params{};
      params.termination_kkt2_tol = tol::kPico;
      params.barrier_strategy = BarrierStrategy::PREDICTOR_CORRECTOR;

      // Some of the randomly generated problems start close to the barrier, which causes
      // them to bounce around a bunch before getting close to the solution. Should implement
      // a strategy for this, but for now I'm gonna crank this up.
      params.max_iterations = 30;

      // can turn on for debugging...
      if (p == -1) {
        PRINT_MATRIX(x_solution);
        for (const LinearInequalityConstraint& c : qp.constraints) {
          std::cout << "x[" << c.variable << "] * " << c.a << " + " << c.b << " >= 0\n";
        }
        solver.SetLoggerCallback(&QPSolverTest::ProgressPrinter);
      }

      const auto term_state = solver.Solve(params);
      ASSERT_EIGEN_NEAR(x_solution, solver.x_block(), 1.0e-4) << "Term: " << term_state << "\n"
                                                              << "Problem p = " << p;
      // check the variables that are constrained
      ASSERT_EIGEN_NEAR(Eigen::VectorXd::Zero(qp.constraints.size()), solver.s_block(), 1.0e-4)
          << "Term: " << term_state << "\n"
          << "Problem p = " << p;
    }
  }
};

TEST_FIXTURE(QPSolverTest, TestSolveNoConstraints)
TEST_FIXTURE(QPSolverTest, TestSolveEqualityConstraints)
TEST_FIXTURE(QPSolverTest, TestSolveInequalityConstraints)
TEST_FIXTURE(QPSolverTest, TestSolveAllConstraints)
TEST_FIXTURE(QPSolverTest, TestWithSingleInequality)
TEST_FIXTURE(QPSolverTest, TestWithInequalitiesActive)
TEST_FIXTURE(QPSolverTest, TestWithInequalitiesPartiallyActive)
TEST_FIXTURE(QPSolverTest, TestWithEqualitiesOnly)
TEST_FIXTURE(QPSolverTest, TestWithFullyConstrainedEqualities)
TEST_FIXTURE(QPSolverTest, TestWithInequalitiesAndEqualities)
TEST_FIXTURE(QPSolverTest, TestGeneratedProblems)

struct ActuatorLink {
  // Euler angles from the decomposed rotation.
  // Factorized w/ order XYZ.
  math::Vector<double, 3> rotation_xyz;

  // Translational part.
  math::Vector<double, 3> translation;

  // Mask of angles that are active in the optimization.
  std::array<uint8_t, 3> active;

  // Number of active angles.
  int ActiveCount() const {
    return static_cast<int>(
        std::count_if(active.begin(), active.end(), [](auto b) { return b > 0; }));
  }

  // Construct from Pose and mask.
  ActuatorLink(const Pose& pose, const std::array<uint8_t, 3>& mask)
      : rotation_xyz(math::EulerAnglesFromSO3(pose.rotation.conjugate())),
        translation(pose.translation),
        active(mask) {}

  // Return pose representing this transform, given the euler angles.
  Pose Compute(const math::Vector<double>& angles, const int position,
               math::Matrix<double, 3, Dynamic>* const J_out) const {
    // Pull out just the angles we care about.
    math::Vector<double, 3> xyz_copy = rotation_xyz;
    for (int i = 0, angle_pos = position; i < 3; ++i) {
      if (active[i]) {
        xyz_copy[i] = angles[angle_pos++];
      }
    }
    // compute rotation and derivatives
    const math::SO3FromEulerAngles_<double> rot =
        math::SO3FromEulerAngles(xyz_copy, math::CompositionOrder::XYZ);
    if (J_out) {
      // copy out derivative blocks we'll need later
      for (int i = 0, angle_pos = position; i < 3; ++i) {
        if (active[i]) {
          J_out->col(angle_pos++) = rot.rotation_D_angles.col(i);
        }
      }
    }
    // Return a pose w/ our fixed translation.
    // TODO(gareth): Dumb that we have to copy the fixed translation always...
    return Pose{rot.q, translation};
  }

  void FillJacobian(
      const Block<const Matrix<double, 3, Dynamic>, 3, 3, true>& output_D_tangent,
      const Block<const Matrix<double, 3, Dynamic>, 3, Dynamic, true>& tangent_D_angles,
      Block<Eigen::Matrix<double, 3, Dynamic>, 3, Dynamic, true> J_out) const {
    // Output buffer should be correct size already.
    ASSERT(J_out.cols() == ActiveCount());
    if (!active[0] && !active[1] && active[2]) {
      // Fast path for common case, we know dz = [0, 0, 1]
      J_out = output_D_tangent.rightCols<1>();
    } else {
      J_out.noalias() = output_D_tangent * tangent_D_angles;
    }
  }
};

struct ActuatorChain {
  // Current poses in the chain.
  std::vector<ActuatorLink> links;

  // private:
  // Poses.
  std::vector<Pose> pose_buffer_;

  // Buffer of rotations derivatives.
  math::Matrix<double, 3, Dynamic> rotation_D_angles_;

  // Cached products while doing computations.
  ChainComputationBuffer chain_buffer_;

  // Cached angles
  math::Vector<double> angles_cached_;

  // Iterate over the chain and compute the effector pose.
  // Derivatives wrt all the input angles are computed and cached locally.
  void Update(const math::Vector<double>& angles) {
    if (!ShouldUpdate(angles)) {
      return;
    }
    angles_cached_ = angles;

    // Recompute.
    if (rotation_D_angles_.size() == 0) {
      // compute total active
      const int total_active = TotalActive();
      ASSERT(angles.rows() == total_active);
      // allocate space
      rotation_D_angles_.resize(3, total_active);
    }

    // compute poses and rotational derivatives
    pose_buffer_.resize(links.size());
    for (int i = 0, position = 0; i < links.size(); ++i) {
      const ActuatorLink& link = links[i];
      const int num_active = link.ActiveCount();
      pose_buffer_[i] = link.Compute(angles, position, &rotation_D_angles_);
      position += num_active;
    }

    // linearize
    ComputeChain(pose_buffer_, &chain_buffer_);
  }

  // Return true if the angles have changed since the last time this was called.
  // Allows us to re-use intermediate values.
  bool ShouldUpdate(const math::Vector<double>& angles) const {
    if (angles_cached_.rows() != angles.rows()) {
      return true;
    }
    for (int i = 0; i < angles.rows(); ++i) {
      if (std::abs(angles_cached_[i] - angles[i]) > 1.0e-6) {
        return true;
      }
    }
    return false;
  }

 public:
  // Compute rotation and translation of the effector.
  math::Vector<double, 3> ComputeEffectorPosition(
      const math::Vector<double>& angles,
      math::Matrix<double, 3, Eigen::Dynamic>* const J = nullptr) {
    Update(angles);

    // chain rule
    if (J) {
      ASSERT(J->cols() == TotalActive());
      for (int i = 0, position = 0; i < links.size(); ++i) {
        const ActuatorLink& link = links[i];
        const int active_count = link.ActiveCount();
        if (active_count == 0) {
          continue;
        }

        // Need const-references so we can get const-blocks.
        const ChainComputationBuffer& const_buffer = chain_buffer_;
        const math::Matrix<double, 3, Dynamic>& const_rotation_D_angles = rotation_D_angles_;

        // Chain rule w/ the angle representation.
        link.FillJacobian(const_buffer.translation_D_tangent.middleCols<3>(i * 3),
                          const_rotation_D_angles.middleCols(position, active_count),
                          J->middleCols(position, active_count));

        position += link.ActiveCount();
      }
    }
    return chain_buffer_.start_T_end().translation;
  }

  int TotalActive() const {
    return std::accumulate(links.begin(), links.end(), 0,
                           [](const int t, const ActuatorLink& l) { return t + l.ActiveCount(); });
  }
};

template <int ResidualDim, int NumParams>
void TestResidualFunctionDerivative(
    const std::function<Eigen::Matrix<double, ResidualDim, 1>(
        const Eigen::Matrix<double, NumParams, 1>&,
        Eigen::Matrix<double, ResidualDim, NumParams>* const)>& function,
    const Eigen::Matrix<double, NumParams, 1>& params, const double tol = tol::kNano) {
  static_assert(ResidualDim != Eigen::Dynamic, "ResidualDim cannot be dynamic");

  // Compute analytically.
  Eigen::Matrix<double, ResidualDim, NumParams> J;
  if (NumParams == Eigen::Dynamic) {
    J.resize(ResidualDim, params.rows());
  }
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
    nls.SetQPLoggingCallback(&QPSolverTest::ProgressPrinter);

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

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
