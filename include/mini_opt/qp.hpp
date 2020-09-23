// Copyright 2020 Gareth Cross
#pragma once
#include <Eigen/Core>

#include "mini_opt/residual.hpp"

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

using ConstVectorBlock = Eigen::VectorBlock<const Eigen::VectorXd>;
using VectorBlock = Eigen::VectorBlock<Eigen::VectorXd>;

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

  // Shift to a new linearization point.
  // a*(x + dx) + b >= 0  -->  a*dx + (ax + b) >= 0
  LinearInequalityConstraint ShiftTo(double x) const {
    return LinearInequalityConstraint(variable, a, a * x + b);
  }

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

// Possible methods of picking the barrier parameter, `mu`.
enum class BarrierStrategy {
  // Set mu = sigma * (s^T * z) / M, where sigma is a scalar decreased at each iteration.
  SCALED_COMPLEMENTARITY = 0,
  // Use Predictor corrector algorithm to select `mu`.
  PREDICTOR_CORRECTOR,
};

struct AlphaValues {
  // Alpha in the primal variables (x an s), set to 1 if we have no s
  double primal{1.};
  // Alpha in the dual variables (y and z), set to 1 if we have no z
  double dual{1.};
};

// Some intermediate values we compute during the iteration of the interior point solver.
struct IPIterationOutputs {
  // Mu, the complementarity condition: s^T * z / M (Equation 16.56).
  double mu{0.};
  // The value of sigma used during the iteration.
  double sigma{1.};
  // Alpha as defined by equation (19.9).
  AlphaValues alpha{};
  // Optional alpha values computing during the MPC probing step.
  AlphaValues alpha_probe{std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()};
  // Mu as determined by taking a step with mu=0, then evaluating equation (19.21).
  // Only relevant in predictor-corrector mode.
  double mu_affine{std::numeric_limits<double>::quiet_NaN()};
};

// Squared errors in the first order KKT conditions.
// At a point that satisfies the conditions, these should all be zero.
struct KKTError {
  double r_dual{0};         // The dual objective: Gx + c - A_e^T * y - A_i * z
  double r_comp{0};         // Complementarity condition: s^T * z
  double r_primal_eq{0};    // Primal equality constraint: A_e*x + b_e
  double r_primal_ineq{0};  // Primal inequality constraint: A_i*x + b_i - s

  // Total squared error.
  double Total() const { return r_dual + r_comp + r_primal_eq + r_primal_ineq; }
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
    // Sigma is multiplied by `mu` at each iteration, to scale down the amount of 'perturbation'
    // we are applying to the KKT conditions. This is the initial value for sigma.
    double initial_sigma{0.1};

    // Amount to reduce sigma on each iteration.
    double sigma_reduction{0.5};

    // If ||kkt||^2 < termination_kkt2_tol, we terminate optimization.
    double termination_kkt2_tol{1.0e-8};

    // Max # of iterations.
    int max_iterations{10};

    // Strategy to apply to barrier parameter `mu`.
    BarrierStrategy barrier_strategy{BarrierStrategy::PREDICTOR_CORRECTOR};
  };

  // List of possible termination criteria.
  enum class TerminationState {
    SATISFIED_KKT_TOL = 0,
    MAX_ITERATIONS,
  };

  // Construct empty.
  QPInteriorPointSolver() = default;

  // Note we don't copy the problem, it must remain in scope for the duration of the solver.
  explicit QPInteriorPointSolver(const QP* const problem, const bool check_feasible = false);

  // Setup with a problem. We allow setting a new problem so storage can be re-used.
  void Setup(const QP* const problem, const bool check_feasible = false);

  /*
   * Iterate until one of the following conditions is met:
   *
   *   - The norm of the first order KKT conditions is less than the tolerance.
   *   - The fixed max number if iterations is hit.
   */
  TerminationState Solve(const Params& params);

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

  // Type for a callback that we call after each iteration, used for logging stats, tests.
  using LoggingCallback =
      std::function<void(const QPInteriorPointSolver& solver, const KKTError& kkt_2_prev,
                         const KKTError& kkt_2_after, const IPIterationOutputs& iter_outputs)>;

  // Access the problem itself. Asserts that `p_` is not null.
  const QP& problem() const;

 private:
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
  Eigen::VectorXd prev_variables_;

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
  IPIterationOutputs Iterate(const double sigma, const BarrierStrategy& strategy);

  // Invert the augmented linear system, which is done by eliminating p_s, and p_z and then
  // solving for p_x and p_y.
  void ComputeLDLT();

  // Apply the result of ComputeLDLT (the inverse H_inv) to compute the update vector
  // [dx, ds, dy, dz] for a given value of mu.
  void SolveForUpdate(const double mu);

  // Fill out the matrix `r_` with the KKT conditions (equations 19.2a-d).
  // Does not apply the `mu` term, which is added later. (ie. mu = 0)
  void EvaluateKKTConditions();

  // Fill out the KKTError struct from `r_.
  KKTError ComputeSquaredErrors() const;

  // Compute the largest step size we can execute that satisfies constraints.
  void ComputeAlpha(AlphaValues* const output, const double tau) const;

  // Compute the `alpha` step size.
  // Returns alpha such that (val[i] + d_val[i]) >= val[i] * (1 - tau)
  static double ComputeAlpha(const ConstVectorBlock& val, const ConstVectorBlock& d_val,
                             const double tau);

  // Compute the predictor/corrector `mu_affine`, equation (19.22)
  double ComputePredictorCorrectorMuAffine(const double mu, const AlphaValues& alpha_probe) const;

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
  void BuildFullSystem(Eigen::MatrixXd* const H, Eigen::VectorXd* const r) const;

  friend class QPSolverTest;
};

// ostream for termination states
std::ostream& operator<<(std::ostream& stream,
                         const QPInteriorPointSolver::TerminationState& state);
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
