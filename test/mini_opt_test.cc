// Copyright 2020 Gareth Cross
#include "mini_opt.hpp"

#include <Eigen/Jacobi>
#include <chrono>
#include <numeric>
#include <random>

#include "numerical_derivative.hpp"
#include "so3.hpp"  //  from geometry_utils
#include "test_utils.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif  // _MSC_VER

namespace mini_opt {

using namespace Eigen;
static const IOFormat kMatrixFmt(FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

#define PRINT_MATRIX(x) \
  { std::cout << #x << ":\n" << (x).eval().format(kMatrixFmt) << "\n"; }

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

  // TODO(gareth): Would really like to use libfmt for this instead...
  static void ProgressPrinter(const QPInteriorPointSolver* const solver, const double kkt2_prev,
                              const double kkt2_after,
                              const QPInteriorPointSolver::IterationOutputs& outputs) {
    std::cout << "Iteration summary: ";
    std::cout << "||kkt||^2: " << kkt2_prev << " --> " << kkt2_after << ", mu = " << outputs.mu
              << ", sigma = " << outputs.sigma << ", a_p = " << outputs.alpha.primal
              << ", a_d = " << outputs.alpha.dual << "\n";

    // dump the state with labels
    std::cout << "After update:\n";
    std::cout << "  x = " << solver->x_block().transpose().format(kMatrixFmt) << "\n";
    std::cout << "  s = " << solver->s_block().transpose().format(kMatrixFmt) << "\n";
    std::cout << "  y = " << solver->y_block().transpose().format(kMatrixFmt) << "\n";
    std::cout << "  z = " << solver->z_block().transpose().format(kMatrixFmt) << "\n";
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
    solver.SetLoggerCallback(std::bind(&QPSolverTest::ProgressPrinter, &solver,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3));

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
    solver.SetLoggerCallback(std::bind(&QPSolverTest::ProgressPrinter, &solver,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3));
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
    solver.SetLoggerCallback(std::bind(&QPSolverTest::ProgressPrinter, &solver,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3));
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
    solver.SetLoggerCallback(std::bind(&QPSolverTest::ProgressPrinter, &solver,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3));
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
    solver.SetLoggerCallback(std::bind(&QPSolverTest::ProgressPrinter, &solver,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3));

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

    solver.SetLoggerCallback(std::bind(&QPSolverTest::ProgressPrinter, &solver,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3));

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
        solver.SetLoggerCallback(std::bind(&QPSolverTest::ProgressPrinter, &solver,
                                           std::placeholders::_1, std::placeholders::_2,
                                           std::placeholders::_3));
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

// TODO(gareth): Template?
struct Pose {
  // Construct w/ rotation and translation.
  Pose(const math::Quaternion<double>& q, const math::Vector<double, 3>& t)
      : rotation(q), translation(t) {}

  Pose()
      : rotation(math::Quaternion<double>::Identity()),
        translation(math::Vector<double, 3>::Zero()) {}

  // Rotation.
  math::Quaternion<double> rotation;

  // Translation.
  math::Vector<double, 3> translation;

  // Multiply together.
  Pose operator*(const Pose& other) const {
    return Pose(rotation * other.rotation, translation + rotation * other.translation);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

struct ChainComputationBuffer {
  // Derivatives of `root_R_effector` wrt joint angles.
  math::Matrix<double, 3, Eigen::Dynamic> orientation_D_tangent;

  // Derivatives of `root_t_effector` wrt joint angles.
  math::Matrix<double, 3, Eigen::Dynamic> translation_D_tangent;

  // Buffer for rotations of the joints, forward direction.
  AlignedVector<math::Quaternion<double>> start_R_i_plus_1;

  // Buffer for rotations of the joints, backwards direction.
  AlignedVector<math::Quaternion<double>> i_R_end;

  // Buffer for translations.
  math::Matrix<double, 3, Eigen::Dynamic> i_t_end;

  Pose start_T_end() const {
    ASSERT(!i_R_end.empty() && i_t_end.cols() > 0);
    return Pose{i_R_end.front(), i_t_end.leftCols<1>()};
  }
};

void ComputeChain(const std::vector<Pose>& links, ChainComputationBuffer* const c) {
  if (links.empty()) {
    // no iteration to do
    c->orientation_D_tangent.resize(3, 0);
    c->translation_D_tangent.resize(3, 0);
    c->start_R_i_plus_1.clear();
    c->i_R_end.clear();
    c->i_t_end.resize(3, 0);
    return;
  }
  const int N = static_cast<int>(links.size());

  // Compute backwards rotations (right to left)
  // Bucket `i` stores [i]_R_end. We don't store N_R_end since this is identity.
  c->i_R_end.resize(N);
  c->i_R_end[N - 1] = links[N - 1].rotation;  // previous_R_current
  for (int i = N - 2; i >= 0; --i) {
    // rotation = previous_R_current
    c->i_R_end[i] = links[i].rotation * c->i_R_end[i + 1];
  }

  // Now compute translations.
  // We are multiplying the transforms, going right to left. The last element (i==0) is
  // root_t_effector.
  c->i_t_end.resize(3, N);
  c->i_t_end.col(N - 1) = links[N - 1].translation;  //  previous_t_current
  for (int i = N - 2; i >= 0; --i) {
    // rotation = previous_R_current
    c->i_t_end.col(i).noalias() = links[i].rotation * c->i_t_end.col(i + 1) + links[i].translation;
  }

  // Compute forward rotations.
  // Bucket `i` stores start_R_[i+1] (we don't store the first one, since it would be identity)
  c->start_R_i_plus_1.resize(N);
  c->start_R_i_plus_1[0] = links[0].rotation;  //  previous_R_current
  for (int i = 1; i < N; ++i) {
    c->start_R_i_plus_1[i] = c->start_R_i_plus_1[i - 1] * links[i].rotation;  // previous_R_current
  }

  // Compute derivative of translation at the end wrt angle i.
  // d(0_t_N) / d(theta_[i]) = start_R_i * d(i_R_[i+1] * [i+1]_t_N) / d(theta_[i])
  //  = start_D_[i+1] * [-[i+1]_t_N]_x
  c->translation_D_tangent.resize(3, N * 3);
  for (int i = 0; i < N - 1; ++i) {
    c->translation_D_tangent.middleCols(i * 3, 3).noalias() =
        c->start_R_i_plus_1[i].matrix() * math::Skew3(-c->i_t_end.col(i + 1));
  }
  c->translation_D_tangent.rightCols<3>().setZero();  //  last angle does not affect translation

  //// Compute derivative of rotation at the end wrt angle i.
  c->orientation_D_tangent.resize(3, N * 3);
  for (int i = 0; i < N - 1; ++i) {
    // d(root_R_eff)/d(theta_i) = n_R_[i+1]
    c->orientation_D_tangent.middleCols(i * 3, 3).noalias() =
        c->i_R_end[i + 1].conjugate().matrix();
  }
  c->orientation_D_tangent.rightCols<3>().setIdentity();
}

TEST(LinkTest, TestRotation) {
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
    ASSERT(angles.rows() == 12);
    std::vector<Pose> links_copied = links;
    for (int i = 0; i < angles.rows() / 3; ++i) {
      links_copied[i].rotation *= math::QuaternionExp(angles.segment(i * 3, 3));
    }
    ChainComputationBuffer c{};
    ComputeChain(links_copied, &c);
    return c.i_t_end.leftCols<1>();
  };

  const auto rotation_lambda = [&](const VectorXd& angles) -> Quaterniond {
    ASSERT(angles.rows() == 12);
    std::vector<Pose> links_copied = links;
    for (int i = 0; i < angles.rows() / 3; ++i) {
      links_copied[i].rotation *= math::QuaternionExp(angles.segment(i * 3, 3));
    }
    ChainComputationBuffer c{};
    ComputeChain(links_copied, &c);
    return c.i_R_end.front();
  };

  // compute numerically
  const Matrix<double, 3, 12> J_trans_numerical =
      math::NumericalJacobian(Matrix<double, 12, 1>::Zero(), translation_lambda);
  const Matrix<double, 3, 12> J_trans_rotational =
      math::NumericalJacobian(Matrix<double, 12, 1>::Zero(), rotation_lambda);

  // check against anlytical
  ChainComputationBuffer c{};
  ComputeChain(links, &c);

  ASSERT_EIGEN_NEAR(J_trans_numerical, c.translation_D_tangent, tol::kNano);
  PRINT_MATRIX(J_trans_numerical);
  PRINT_MATRIX(c.translation_D_tangent);

  ASSERT_EIGEN_NEAR(J_trans_rotational, c.orientation_D_tangent, tol::kNano);
  PRINT_MATRIX(J_trans_rotational);
  PRINT_MATRIX(c.orientation_D_tangent);
}

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

 private:
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
        Eigen::Matrix<double, ResidualDim, NumParams, 1>* const)>& function,
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
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.25, 0.1, 0.0}), mask);
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.6, 0.05, 0.0}), mask);
    chain->links.emplace_back(Pose(Quaterniond::Identity(), Vector3d{0.3, 0.0, 0.0}),
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
        ASSERT(J_out->cols() == 2);
        *J_out = J_full.middleRows<1>(1);
      }
      return Matrix<double, 1, 1>{effector_xyz.y() - 0.1};
    };

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
        ASSERT(J_out->cols() == 2);
        *J_out = J_full.topRows<1>();
      }
      return Matrix<double, 1, 1>{effector_xyz.x() - 0.3};
    };

    TestResidualFunctionDerivative<1, Dynamic>(y_residual.function, VectorXd{Vector2d(-0.5, 0.4)});
    TestResidualFunctionDerivative<1, Dynamic>(x_eq_constraint.function,
                                               VectorXd{Vector2d(0.3, -0.6)});

    Problem problem{};
    problem.costs.emplace_back(new Residual<1, Dynamic>(y_residual));
    problem.equality_constraints.emplace_back(new Residual<1, Dynamic>(x_eq_constraint));
    // problem.inequality_constraints.push_back(Var(0) <= 3 * M_PI / 4);
    // problem.inequality_constraints.push_back(Var(0) >= 0);
    // problem.inequality_constraints.push_back(Var(1) <= 3 * M_PI / 4);
    // problem.inequality_constraints.push_back(Var(1) >= -3 * M_PI / 4);
    problem.dimension = 2;

    ConstrainedNonlinearLeastSquares nls(&problem);
    nls.SetQPLoggingCallback([](const double kkt2_prev, const double kkt2_after,
                                const QPInteriorPointSolver::IterationOutputs& outputs) {
      std::cout << "Iteration summary: ";
      std::cout << "||kkt||^2: " << kkt2_prev << " --> " << kkt2_after << ", mu = " << outputs.mu
                << ", sigma = " << outputs.sigma << ", a_p = " << outputs.alpha.primal
                << ", a_d = " << outputs.alpha.dual << "\n";
    });

    const Vector2d initial_values{M_PI / 4, -M_PI / 6};
    nls.SetVariables(initial_values);
    nls.LinearizeAndSolve();
    nls.LinearizeAndSolve();
    nls.LinearizeAndSolve();
    nls.LinearizeAndSolve();

    const VectorXd angles_out = nls.variables();
    PRINT_MATRIX(chain->ComputeEffectorPosition(angles_out).transpose());

    PRINT(y_residual.Error(angles_out));
    PRINT(x_eq_constraint.Error(angles_out));

    MatrixXd J(1, 2);
    VectorXd b(1);
    y_residual.UpdateJacobian(angles_out, J.block(0, 0, 1, 2), b.segment(0, 1));
    PRINT_MATRIX(J);
    PRINT_MATRIX(b);
  }
};

TEST_FIXTURE(ConstrainedNLSTest, TestActuatorChain)

}  // namespace mini_opt

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
