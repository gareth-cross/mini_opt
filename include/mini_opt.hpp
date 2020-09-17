// Copyright 2020 Gareth Cross
#pragma once
#include <Eigen/Core>
#include <array>
#include <memory>
#include <vector>

#include "assertions.hpp"

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

// Base type for residuals in case we want more than one.
struct ResidualBase {
  using unique_ptr = std::unique_ptr<ResidualBase>;

  // We will be storing these through pointer to the base class.
  virtual ~ResidualBase();

  // Dimension of the residual vector.
  virtual std::size_t Dimension() const = 0;

  // Get the error: (1/2) * h(x)^T * h(x)
  virtual double Error(const Eigen::VectorXd& params) const = 0;

  // Update a system of equations Hx=b by writing to `H` and `b`.
  // Returns the value of `Error` as well.
  virtual double UpdateHessian(const Eigen::VectorXd& params, Eigen::MatrixXd* const H,
                               Eigen::VectorXd* const b) const = 0;

  // Output the jacobian for the linear system: J * dx + b
  virtual void UpdateJacobian(const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
                              Eigen::VectorBlock<Eigen::VectorXd> b_out) const = 0;
};

// Simple statically sized residual.
template <size_t ResidualDim, size_t NumParams>
struct Residual : public ResidualBase {
  using ParamType = Eigen::Matrix<double, NumParams, 1>;
  using ResidualType = Eigen::Matrix<double, ResidualDim, 1>;
  using JacobianType = Eigen::Matrix<double, ResidualDim, NumParams>;

  // Variables we are touching, one per column in the jacobian.
  std::array<int, NumParams> index;

  // Function that evaluates the residual given the params, and returns an error vector and
  // optionally the jacobian via the output argument.
  std::function<ResidualType(const ParamType& params, JacobianType* const J_out)> function;

  // Return constant dimension.
  std::size_t Dimension() const override { return ResidualDim; }

  // Map params from the global state vector to those required for this function, and
  // then evaluate the function.
  double Error(const Eigen::VectorXd& params) const override;

  // Map params from the global state vector to those required for this function, and
  // then evaluate the function and its derivative. Update the linear system [H|b] w/
  // the result.
  double UpdateHessian(const Eigen::VectorXd& params, Eigen::MatrixXd* const H,
                       Eigen::VectorXd* const b) const override;

  // Implementation of abstract method UpdateJacobian.
  void UpdateJacobian(const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
                      Eigen::VectorBlock<Eigen::VectorXd> b_out) const override;

 private:
  // Copy out the params that matter for this function.
  ParamType GetParamSlice(const Eigen::VectorXd& params) const;
};

/*
 * Describes a linear (technically affine) inequality constraint.
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

  // Ctor
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

/*
 * Minimize quadratic cost function with inequality constraints using interior point method.
 *
 * Not doing any smart initial guess selection at the moment. We assume x_guess=0, which must
 * be feasible (which works for my application).
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
    BarrierStrategy barrier_strategy{BarrierStrategy::SCALED_COMPLEMENTARITY};
  };

  // List of possible termination criteria.
  enum class TerminationState {
    SATISFIED_KKT_TOL = 0,
    MAX_ITERATIONS,
  };

  // Note we don't copy the problem, it must remain in scope for the duration of the solver.
  explicit QPInteriorPointSolver(const QP& problem);

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

  struct AlphaValues {
    // Alpha in the primal variables (x an s), set to 1 if we have no s
    double primal{1.};
    // Alpha in the dual variables (y and z), set to 1 if we have no z
    double dual{1.};
  };

  // Some derivative values we compute during Iterate.
  struct IterationOutputs {
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

  using LoggingCallback = std::function<void(double kkt_2_prev, double kkt_2_after,
                                             const IterationOutputs& iter_outputs)>;

 private:
  const QP& p_;

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

  bool HasInequalityConstraints() const { return dims_.M > 0; }

  // Take a single step.
  // Computes mu, solves for the update, and takes the longest step we can while satisfying
  // constraints.
  IterationOutputs Iterate(const double sigma, const BarrierStrategy& strategy);

  // Invert the augmented linear system, which is done by eliminating p_s, and p_z and then
  // solving for p_x and p_y.
  void ComputeLDLT();

  // Apply the result of ComputeLDLT (the inverse H_inv) to compute the update vector
  // [dx, ds, dy, dz] for a given value of mu.
  void SolveForUpdate(const double mu);

  // Fill out the matrix `r_` with the KKT conditions (equations 19.2a-d).
  // Does not apply the `mu` term, which is added later. (ie. mu = 0)
  void EvaluateKKTConditions();

  // Compute the largest step size we can execute that satisfies constraints.
  void ComputeAlpha(AlphaValues* const output, const double tau) const;

  // Compute the `alpha` step size.
  // Returns alpha such that (val[i] + d_val[i]) >= val[i] * (1 - tau)
  double ComputeAlpha(const ConstVectorBlock& val, const ConstVectorBlock& d_val,
                      const double tau) const;

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
 * Describes a simple [non-]linear least squares problem. The primary cost is a sum-of
 * squares.
 *
 * Supports simple linear inequality constraints on the variables.
 *
 * More formally:
 *
 *  min: f_0(x)  [where f_0(x) = (1/2) * h(x)^T * h(x)]
 *
 *  Subject to: diag(a) * x + b >= 0, and
 *  Subject to: g(x) == 0
 *
 * Note that we actually iteratively minimize the first order approximation of f(x):
 *
 *  h(x + dx) = h(x) + J * dx
 *
 * Such that: f_0(x) = (1/2) * h(x)^T * h(x) + (J * dx)^T * h(x) + (J * dx)^T * (J * dx)
 *
 * So in effect, we are solving a quadratic approximation of the nonlinear cost
 * with diagonal inequality constraints on the decision variables.
 */
struct Problem {
  using unique_ptr = std::unique_ptr<Problem>;

  // Problem dimension. (ie. max variable index + 1)
  std::size_t dimension;

  // The errors that form the sum of squares part of the cost function.
  std::vector<ResidualBase::unique_ptr> costs;

  // Linear inequality constraints.
  std::vector<LinearInequalityConstraint> inequality_constraints;

  // Nonlinear inequality constraints.
  std::vector<ResidualBase::unique_ptr> equality_constraints;
};

/*
 * Solve constrained non-linear least squares problems using SQP.
 *
 * At each step we approximate the problem as a quadratic with linear equality constraints
 * and inequality constraints about the current lineariation point. We do this iteratively
 * until satisfied with the result on the original nonlinear cost.
 */
struct ConstrainedNonlinearLeastSquares {
 public:
  // Construct w/ const pointer to a problem definition.
  explicit ConstrainedNonlinearLeastSquares(const Problem* const problem);

  // Linearize and take a step.
  void LinearizeAndSolve();

  template <typename T>
  void SetQPLoggingCallback(T cb) {
    qp_logger_callback_ = cb;
  }

 private:
  const Problem* const p_;

  // Storage for the QP representation of the problem.
  QP qp_{};

  // Parameters (the current linearization point)
  Eigen::VectorXd variables_;

  // Callback we pass to the QP solver
  QPInteriorPointSolver::LoggingCallback qp_logger_callback_;
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

//
// Template implementations.
// TODO(gareth): Could put these in a separate header.
//

template <size_t ResidualDim, size_t NumParams>
typename Residual<ResidualDim, NumParams>::ParamType
Residual<ResidualDim, NumParams>::GetParamSlice(const Eigen::VectorXd& params) const {
  ParamType sliced;
  static_assert(ParamType::RowsAtCompileTime == NumParams, "");
  for (std::size_t local = 0; local < NumParams; ++local) {
    const int i = index[local];
    ASSERT(i >= 0);
    ASSERT(i < params.rows(), "Index %i exceeds the # of provided params, which is %i", i,
           params.rows());
    sliced[local] = params[i];
  }
  return sliced;
}

template <size_t ResidualDim, size_t NumParams>
double Residual<ResidualDim, NumParams>::Error(const Eigen::VectorXd& params) const {
  const ParamType relevant_params = GetParamSlice(params);
  const ResidualType err = function(relevant_params, nullptr);
  return 0.5 * err.squaredNorm();
}

// TODO(gareth): Probably faster to associate a dimension to each variable,
// in the style of GTSAM, so that we can do block-wise updates. For now this
// suits the small problem size I am doing.
template <size_t ResidualDim, size_t NumParams>
double Residual<ResidualDim, NumParams>::UpdateHessian(const Eigen::VectorXd& params,
                                                       Eigen::MatrixXd* const H,
                                                       Eigen::VectorXd* const b) const {
  ASSERT(H != nullptr);
  ASSERT(b != nullptr);
  ASSERT(H->rows() == H->cols());
  ASSERT(b->rows() == H->rows());

  // Collect params.
  const ParamType relevant_params = GetParamSlice(params);

  // Evaluate the function and its derivative.
  JacobianType J;
  const ResidualType r = function(relevant_params, &J);

  // Add contributions to the hessian, only lower part.
  constexpr int N = static_cast<int>(NumParams);
  for (int row_local = 0; row_local < N; ++row_local) {
    // get index mapping into the full system
    const int row_global = index[row_local];
    ASSERT(row_global < H->rows(), "Index %i exceeds the bounds of the hessian (rows = %i)",
           row_global, H->rows());
    for (int col_local = 0; col_local <= row_local; ++col_local) {
      // pull column index
      // because col_local <= row_local, we already checked this global index
      const int col_global = index[col_local];

      // each param is a single column, so we can just do dot product
      const double JtT = J.col(row_local).dot(J.col(col_local));
      // swap so we only update the lower triangular part
      if (col_global <= row_global) {
        H->operator()(row_global, col_global) += JtT;
      } else {
        H->operator()(col_global, row_global) += JtT;
      }
    }
    // Also update the right hand side vector `b`.
    b->operator()(row_global) += J.col(row_local).dot(r);
  }
  return 0.5 * r.squaredNorm();
}

template <size_t ResidualDim, size_t NumParams>
void Residual<ResidualDim, NumParams>::UpdateJacobian(
    const Eigen::VectorXd& params, Eigen::Block<Eigen::MatrixXd> J_out,
    Eigen::VectorBlock<Eigen::VectorXd> b_out) const {
  ASSERT(ResidualDim == b_out.rows());
  ASSERT(ResidualDim == J_out.rows());
  // Collect params.
  const ParamType relevant_params = GetParamSlice(params);

  // Evaluate, and copy jacobian back using indices.
  JacobianType J;
  b_out.noalias() = function(relevant_params, &J);

  for (int col_local = 0; col_local < NumParams; ++col_local) {
    const int col_global = index[col_local];
    ASSERT(col_global < J_out.cols(), "Index %i exceeds the size of the Jacobian (cols = %i)",
           col_global);
    J_out.col(col_global).noalias() = J.col(col_local);
  }
}

}  // namespace mini_opt
