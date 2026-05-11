// Copyright 2021 Gareth Cross
#pragma once
#include <Eigen/Core>

#include "mini_opt/assertions.hpp"
#include "mini_opt/structs.hpp"

/*
 * The reference for this implementation is:
 *
 *   "Numerical Optimization, Second Edition", Jorge Nocedal and Stephen J. Wright
 *
 * Any equation numbers you see refer to this book, unless otherwise stated.
 *
 */
namespace mini_opt {

/*
 * Describes a linear (technically affine) inequality constraint.
 *
 * The constraint is specified in the form:
 *
 *    a * x[variable] + b >= 0
 *
 */
struct linear_inequality_flat_index {
  // Index of the variable this refers to.
  int variable;

  // Constraint coefficients.
  double a;
  double b;

  // Clamp a variable x to satisfy the inequality constraint.
  constexpr double ClampX(double x) const {
    MINI_OPT_ASSERT_NE(a, 0, "`a` cannot be zero");
    // a * x + b >= 0 ---> a * x >= -b
    if (a < 0) {
      // x <= b/a
      return std::min(x, b / -a);
    } else {
      // x >= -b/a
      return std::max(x, -b / a);
    }
  }

  // Construct with index and coefficients.
  constexpr linear_inequality_flat_index(int variable, double a, double b) noexcept
      : variable(variable), a(a), b(b) {}
};

/*
 * Problem specification for a QP:
 *
 *  minimize x^T * G * x + x^T * c
 *
 *  st. A_e * x + b_e == 0
 *  st. a_i * x + b_i >= 0  (A_i is diagonal)
 *
 * Note that the solver assumes G is positive definite.
 */
struct QP {
  // Default initialize empty.
  QP() = default;

  // Convenience constructor for unit tests. Initializes to zero.
  explicit QP(const Eigen::Index x_dim)
      : G(Eigen::MatrixXd::Zero(x_dim, x_dim)), c(Eigen::VectorXd::Zero(x_dim)) {}

  Eigen::MatrixXd G;
  Eigen::VectorXd c;

  // Optional equality constraints in form Ax + b = 0
  Eigen::MatrixXd A_eq;
  Eigen::VectorXd b_eq;

  // Diagonal inequality constraints.
  std::vector<linear_inequality_flat_index> constraints;

  // Compute and return the `QPEigenvalues` struct, which summarizes eigenvalues of `G`.
  QPEigenvalues ComputeEigenvalueStats() const;
};

/*
 * Minimize quadratic cost function with inequality constraints using interior point method.
 *
 * Not doing any smart initial guess selection at the moment. We assume x_guess=0, which works
 * because this is called from a non-linear solver that linearizes about the current value.
 */
struct QPInteriorPointSolver {
  // Parameters of the solver.
  struct Params {
    // Initial value of the barrier.
    double initial_mu{1.0};

    // Amount to reduce mu on each iteration.
    double sigma{0.5};

    // If max(||kkt||) < termination_kkt_tol, we terminate optimization (L2 norm).
    double termination_kkt_tol{1.0e-9};

    // Tolerance on the complementarity condition required to terminate.
    // If (s^T * z) / M <= tol, we can exit (provided KKT tol is also met).
    double termination_complementarity_tol{1.0e-6};

    // Max # of iterations.
    int max_iterations{10};

    // Strategy to apply to barrier parameter `mu`.
    BarrierStrategy barrier_strategy{BarrierStrategy::COMPLEMENTARITY};

    // If true, decrease mu only when E(x,y,z,s) < mu.
    // Ie. when the max of th KKT L2 errors is less than mu.
    // This matches the text, but leads to very slow convergence in practice.
    bool decrease_mu_only_on_small_error{false};

    // Method of generating an initial guess.
    InitialGuessMethod initial_guess_method{InitialGuessMethod::NAIVE};

    // Initialize `mu = (s^T * z) / M` instead of initial_mu.
    bool initialize_mu_with_complementarity{false};
  };

  // Construct empty.
  QPInteriorPointSolver() = default;

  // Note we don't copy the problem, it must remain in scope for the duration of the solver.
  // Calls `Setup`.
  explicit QPInteriorPointSolver(const QP* problem);

  // Setup with a problem. We allow setting a new problem so storage can be re-used.
  void Setup(const QP* problem);

  /*
   * Iterate until one of the following conditions is met:
   *
   *   - The norm of the first order KKT conditions is less than the tolerance.
   *   - The fixed max number if iterations is hit.
   *
   * Returns termination state and number of iterations executed.
   */
  [[nodiscard]] QPInteriorPointSolverOutputs Solve(const Params& params);

  // Const block accessors for state.
  ConstVectorBlock x_block() const;
  ConstVectorBlock s_block() const;
  ConstVectorBlock y_block() const;
  ConstVectorBlock z_block() const;

  // Mutable block accessors.
  VectorBlock x_block();
  VectorBlock s_block();
  VectorBlock y_block();
  VectorBlock z_block();

  // Access all variables.
  constexpr const Eigen::VectorXd& variables() const noexcept { return variables_; }

  // Set variables.
  void SetVariables(const Eigen::VectorXd& v);

  // Access the problem itself. Asserts that `p_` is not null.
  const QP& problem() const;

 private:
  // Current problem, initialized by `Setup`.
  const QP* p_{nullptr};

  struct ProblemDims {
    int N;  //  number of variables `x`
    int M;  //  number of inequality constraints, `s` and `z`
    int K;  //  number of equality constraints, `y`
  };

  // For convenience we save these here
  ProblemDims dims_{0, 0, 0};

  // Storage for the variables: (x, s, y, z)
  Eigen::VectorXd variables_;

  // Re-usable storage for the linear system and residuals
  Eigen::MatrixXd H_;
  Eigen::VectorXd r_;
  Eigen::VectorXd r_dual_aug_;
  Eigen::MatrixXd H_inv_;

  // Solution vector at each iteration
  Eigen::VectorXd delta_;
  Eigen::VectorXd delta_affine_;

  // Return true if there are any inequality constraints.
  constexpr bool HasInequalityConstraints() const noexcept { return dims_.M > 0; }

  // Take a single step.
  // Computes mu, solves for the update, and takes the longest step we can while satisfying
  // constraints.
  IPIterationOutputs Iterate(double mu_input, const BarrierStrategy& strategy);

  // Invert the augmented linear system, which is done by eliminating p_s, and p_z and then
  // solving for p_x and p_y.
  void ComputeLDLT(bool include_inequalities = true);

  // Apply the result of ComputeLDLT (the inverse H_inv) to compute the update vector
  // [dx, ds, dy, dz] for a given value of mu.
  void SolveForUpdate(double mu);

  // Apply the result of ComputeLDLT (the inverse H_inv) to compute the update vector
  // [dx, 0, dy, 0], ignoring contributions from inequality constraints.
  void SolveForUpdateNoInequalities();

  // Fill out the matrix `r_` with the KKT conditions (equations 19.2a-d).
  // Does not apply the `mu` term, which is added later. (ie. mu = 0)
  void EvaluateKKTConditions(bool include_inequalities = true);

  // Fill out the KKTError struct from `r_.
  KKTError ComputeErrors(double mu) const;

  // Compute the initial guess.
  void ComputeInitialGuess(const Params& params);

  // Compute the largest step size we can execute that satisfies constraints.
  void ComputeAlpha(AlphaValues* output, double tau) const;

  // Compute the `alpha` step size.
  // Returns alpha such that (val[i] + d_val[i]) >= val[i] * (1 - tau)
  static double ComputeAlpha(const ConstVectorBlock& val, const ConstVectorBlock& d_val,
                             double tau);

  // Compute `mu` as defined in equation 19.19
  double ComputeMu() const;

  // Compute the predictor/corrector `mu_affine`, equation (19.22)
  double ComputePredictorCorrectorMuAffine(double mu, const AlphaValues& alpha_probe) const;

  // Compute summary of the lagrange-multipliers, if we have equality constraints.
  std::optional<QPLagrangeMultipliers> ComputeLagrangeMultiplierSummary() const;

  // Helpers for accessing segments of vectors.
  static ConstVectorBlock ConstXBlock(const ProblemDims& dims, const Eigen::VectorXd& vec);
  static ConstVectorBlock ConstSBlock(const ProblemDims& dims, const Eigen::VectorXd& vec);
  static ConstVectorBlock ConstYBlock(const ProblemDims& dims, const Eigen::VectorXd& vec);
  static ConstVectorBlock ConstZBlock(const ProblemDims& dims, const Eigen::VectorXd& vec);
  static VectorBlock XBlock(const ProblemDims& dims, Eigen::VectorXd& vec);
  static VectorBlock SBlock(const ProblemDims& dims, Eigen::VectorXd& vec);
  static VectorBlock YBlock(const ProblemDims& dims, Eigen::VectorXd& vec);
  static VectorBlock ZBlock(const ProblemDims& dims, Eigen::VectorXd& vec);

  // For unit test, allow construction of the full linear system required for Newton step.
  void BuildFullSystem(Eigen::MatrixXd* H, Eigen::VectorXd* r) const;

  friend class QPSolverTest;
  friend class ConstrainedNLSTest;
};

/*
 * Solver for equality-constrained quadratic problem.
 *
 * minimize (1/2) x^T * G * x + x^T * c
 *
 *  st. A_e * x + b_e == 0
 */
class QPNullSpaceSolver {
 public:
  // Solve the QP using the null-space method.
  [[nodiscard]] QPNullSpaceTerminationState Solve(const QP& p);

  // Access all variables.
  constexpr const Eigen::VectorXd& variables() const noexcept { return x_; }

 private:
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd G_reduced_;
  Eigen::VectorXd permuted_rhs_;
  Eigen::VectorXd u_;
  Eigen::VectorXd y_;
  Eigen::VectorXd x_;
};

/*
 * Some exceptions we can throw.
 */

// Initial guess is not feasible according to constraints.
struct InfeasibleGuess : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Failed to factorize system.
struct FailedFactorization : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace mini_opt
