// Copyright 2020 Gareth Cross
#pragma once
#include <Eigen/Core>

#include "mini_opt/residual.hpp"
#include "mini_opt/structs.hpp"

/*
 * The reference for this implementation is:
 *
 *   "Numerical Optimization, Second Edition", Jorge Nocedal and Stephen J. Wright
 *
 * Any equation numbers you see refer to this book, unless otherwise stated.
 *
 * TODO(gareth): Template everything for double or float? For now double suits me.
 */
namespace mini_opt {

/*
 * Describes a linear (technically affine) inequality constraint.
 *
 * The constraint is specified in the form:
 *
 *    a * x[variable] + b >= 0
 *
 * TODO(gareth): Generalize beyond diagonal A-matrix?
 */
struct LinearInequalityConstraint {
  // Index of the variable this refers to.
  int variable;
  // Constraint coefficients.
  double a;
  double b;

  // True if x is feasible.
  bool IsFeasible(double x) const;

  // Clamp a variable x to satisfy the inequality constraint.
  double ClampX(double x) const;

  // Shift to a new linearization point.
  // a*(x + dx) + b >= 0  -->  a*dx + (ax + b) >= 0
  LinearInequalityConstraint ShiftTo(double x) const { return {variable, a, a * x + b}; }

  // Version of shift that takes vector.
  LinearInequalityConstraint ShiftTo(const Eigen::VectorXd& x) const {
    ASSERT(variable < x.rows());
    return ShiftTo(x[variable]);
  }

  // Construct with index and coefficients.
  LinearInequalityConstraint(int variable, double a, double b) : variable(variable), a(a), b(b) {}

  LinearInequalityConstraint() = default;
};

/*
 * Helper for specifying constraints in a more legible way.
 *
 * Allows you to write Var(index) >= alpha to specify the appropriate LinearInequalityConstraint.
 */
struct Var {
  explicit Var(int variable) : variable_(variable) {}

  // Specify constraint as <=
  LinearInequalityConstraint operator<=(double value) const {
    return LinearInequalityConstraint(variable_, -1.0, value);
  }

  // Specify constraint as >=
  LinearInequalityConstraint operator>=(double value) const {
    return LinearInequalityConstraint(variable_, 1.0, -value);
  }

  const int variable_;
};

/*
 * Problem specification for a QP:
 *
 *  minimize x^T * G * x + c^T * c
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
  std::vector<LinearInequalityConstraint> constraints;
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
  QPSolverOutputs Solve(const Params& params);

  // Set the logger callback to a function pointer, lambda, etc.
  template <typename T>
  void SetLoggerCallback(T cb) {
    logger_callback_ = cb;
  }

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
  const Eigen::VectorXd& variables() const;

  // Set variables.
  void SetVariables(const Eigen::VectorXd& v);

  // Type for a callback that we call after each iteration, used for logging stats, tests.
  using LoggingCallback =
      std::function<void(const QPInteriorPointSolver& solver, const KKTError& kkt_prev,
                         const KKTError& kkt_after, const IPIterationOutputs& iter_outputs)>;

  // Access the problem itself. Asserts that `p_` is not null.
  const QP& problem() const;

 private:
  // Current problem, initialized by `Setup`.
  const QP* p_{nullptr};

  struct ProblemDims {
    std::size_t N;  //  number of variables `x`
    std::size_t M;  //  number of inequality constraints, `s` and `z`
    std::size_t K;  //  number of equality constraints, `y`
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

  // Optional iteration logger.
  LoggingCallback logger_callback_;

  // Return true if there are any inequality constraints.
  bool HasInequalityConstraints() const { return dims_.M > 0; }

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
