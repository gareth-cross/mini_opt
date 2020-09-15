// Copyright 2020 Gareth Cross
#include "mini_opt.hpp"

#include <chrono>

#include "test_utils.hpp"

namespace mini_opt {

using namespace Eigen;
const IOFormat kMatrixFmt(FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

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
static Eigen::MatrixXd CreateRemapMatrix(const std::array<int, N>& index,
                                         const std::size_t full_size) {
  Eigen::MatrixXd small_D_large(N, full_size);
  small_D_large.setZero();
  for (int row = 0; row < N; ++row) {
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
  ASSERT_EQ(expected_error.squaredNorm(), res.Error(global_params));

  // update
  res.UpdateSystem(global_params, &H, &b);

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
  ASSERT_EQ(expected_error.squaredNorm(), res.Error(global_params));

  // update
  res.UpdateSystem(global_params, &H, &b);

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
  ASSERT_EQ(expected_error.squaredNorm(), res.Error(global_params));

  // update
  res.UpdateSystem(global_params, &H, &b);

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

  // check that other cells were left at zero by UpdateSystem
  const auto H_empty = (H.selfadjointView<Eigen::Lower>().toDenseMatrix() -
                        (J * local_D_global).transpose() * (J * local_D_global))
                           .eval();
  ASSERT_EIGEN_NEAR(MatrixXd::Zero(7, 7), H_empty, 0.0);
}

// Tests for the QP interior point solver.
class QPSolverTest : public ::testing::Test {
 public:
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

  // Check that the solution of the 'augmented system' (which leverages sparsity)
  // matches the full 'brute force' solve that uses LU decomposition.
  void CheckAugmentedSolveAgainstPartialPivot(const QP& qp, const VectorXd& x_guess) {
    const Index N = qp.G.rows();
    const Index K = qp.A_eq.rows();
    const Index M = static_cast<Index>(qp.constraints.size());

    // construct the solver
    QPInteriorPointSolver solver(qp, x_guess);

    // Give all the multipliers different positive non-zero values.
    // The exact values aren't actually important, we just want to validate indexing.
    auto s = solver.variables_.segment(N, M);
    auto y = solver.variables_.segment(N + M, K);
    auto z = solver.variables_.tail(M);
    for (Index i = 0; i < M; ++i) {
      s[i] = 2.0 / (i + 1);
      z[i] = 0.5 * (i + 1);
    }
    for (Index k = 0; k < K; ++k) {
      y[k] = static_cast<double>((k + 1) * (k + 1));
    }

    // check the dimensions
    const Index total_dims = N + M * 2 + K;
    ASSERT_EQ(total_dims, solver.variables_.rows());
    ASSERT_EQ(total_dims, solver.delta_.rows());
    ASSERT_EQ(total_dims, solver.r_.rows());
    ASSERT_EQ(N + K, solver.H_.rows());  //  only x and y

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
    update.segment(N + M, K).array() *= -1.0;  // flip dy
    update.tail(M).array() *= -1.0;            // flip dz

    // now solve w/ cholesky
    solver.EvaluateKKTConditions();
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

  // Scalar quadratic equation with a single inequality constraint.
  void TestScalarQuadraticWithInequality() {
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
    res.UpdateSystem(initial_values, &qp.G, &qp.c);

    // inequality constraint: x <= 4 --> -x >= -4 --> -x + 4 >= 0
    qp.constraints.emplace_back(/* variable index = */ 0, -1.0, 4.0);

    QPInteriorPointSolver solver(qp, VectorXd::Zero(1));
    std::cout << solver.StateToString() << std::endl;

    // start with sigma=1
    solver.Iterate(0.1);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
  }

  // Quadratic in two variables w/ two inequalities keep them both from their optimal values.
  void TestQuadraticWithInequalities() {
    using ScalarMatrix = Matrix<double, 1, 1>;

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
    res.UpdateSystem(initial_values, &qp.G, &qp.c);
    qp.constraints.emplace_back(0, -1.0, 1.0);  // x0 <= 1.0
    qp.constraints.emplace_back(1, 1.0, 3.0);   // x1 >= -3.0

    QPInteriorPointSolver solver(qp, Vector2d::Zero());
    std::cout << solver.StateToString() << std::endl;

    // start with sigma=1
    solver.Iterate(0.1);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
    solver.Iterate(0.0000001);
    std::cout << solver.StateToString() << std::endl;
  }

 private:
};

TEST_FIXTURE(QPSolverTest, TestSolveNoConstraints)
TEST_FIXTURE(QPSolverTest, TestSolveEqualityConstraints)
TEST_FIXTURE(QPSolverTest, TestSolveInequalityConstraints)
TEST_FIXTURE(QPSolverTest, TestSolveAllConstraints)
TEST_FIXTURE(QPSolverTest, TestScalarQuadraticWithInequality)
TEST_FIXTURE(QPSolverTest, TestQuadraticWithInequalities)

}  // namespace mini_opt
