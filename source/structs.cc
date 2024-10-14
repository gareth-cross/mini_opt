#include "mini_opt/structs.hpp"

#include <numeric>
#include <ostream>

namespace mini_opt {

std::ostream& operator<<(std::ostream& stream, InitialGuessMethod method) {
  switch (method) {
    case InitialGuessMethod::NAIVE:
      stream << "NAIVE";
      break;
    case InitialGuessMethod::SOLVE_EQUALITY_CONSTRAINED:
      stream << "SOLVE_EQUALITY_CONSTRAINED";
      break;
    case InitialGuessMethod::USER_PROVIDED:
      stream << "USER_PROVIDED";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const QPInteriorPointTerminationState state) {
  switch (state) {
    case QPInteriorPointTerminationState::SATISFIED_KKT_TOL:
      stream << "SATISFIED_KKT_TOL";
      break;
    case QPInteriorPointTerminationState::MAX_ITERATIONS:
      stream << "MAX_ITERATIONS";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, QPNullSpaceTerminationState termination) {
  switch (termination) {
    case QPNullSpaceTerminationState::SUCCESS:
      stream << "SUCCESS";
      break;
    case QPNullSpaceTerminationState::NOT_POSITIVE_DEFINITE:
      stream << "NOT_POSITIVE_DEFINITE";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const LineSearchStrategy strategy) {
  switch (strategy) {
    case LineSearchStrategy::ARMIJO_BACKTRACK:
      stream << "ARMIJO_BACKTRACK";
      break;
    case LineSearchStrategy::POLYNOMIAL_APPROXIMATION:
      stream << "POLYNOMIAL_APPROXIMATION";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const OptimizerState v) {
  switch (v) {
    case OptimizerState::NOMINAL:
      stream << "NOMINAL";
      break;
    case OptimizerState::ATTEMPTING_RESTORE_LM:
      stream << "ATTEMPTING_RESTORE_LM";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const StepSizeSelectionResult result) {
  switch (result) {
    case StepSizeSelectionResult::SUCCESS:
      stream << "SUCCESS";
      break;
    case StepSizeSelectionResult::MAX_ITERATIONS:
      stream << "MAX_ITERATIONS";
      break;
    case StepSizeSelectionResult::POSITIVE_DERIVATIVE:
      stream << "POSITIVE_DERIVATIVE";
      break;
    case StepSizeSelectionResult::FIRST_ORDER_SATISFIED:
      stream << "FIRST_ORDER_SATISFIED";
      break;
    case StepSizeSelectionResult::FAILURE_NON_FINITE_COST:
      stream << "FAILURE_NON_FINITE_COST";
      break;
    case StepSizeSelectionResult::FAILURE_INVALID_ALPHA:
      stream << "FAILURE_INVALID_ALPHA";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const NLSTerminationState state) {
  switch (state) {
    case NLSTerminationState::MAX_ITERATIONS:
      stream << "MAX_ITERATIONS";
      break;
    case NLSTerminationState::SATISFIED_ABSOLUTE_TOL:
      stream << "SATISFIED_ABSOLUTE_TOL";
      break;
    case NLSTerminationState::SATISFIED_RELATIVE_TOL:
      stream << "SATISFIED_RELATIVE_TOL";
      break;
    case NLSTerminationState::SATISFIED_FIRST_ORDER_TOL:
      stream << "SATISFIED_FIRST_ORDER_TOL";
      break;
    case NLSTerminationState::MAX_LAMBDA:
      stream << "MAX_LAMBDA";
      break;
    case NLSTerminationState::QP_INDEFINITE:
      stream << "QP_INDEFINITE";
      break;
    case NLSTerminationState::USER_CALLBACK:
      stream << "USER_CALLBACK";
      break;
  }
  return stream;
}

std::string QPInteriorPointIteration::ToString() const {
  std::string result{};
  fmt::format_to(std::back_inserter(result),
                 "Iteration summary: "
                 "||kkt|| max: {} --> {}, mu = {}, a_p = {}, a_d = {}\n",
                 kkt_initial.Max(), kkt_final.Max(), ip_outputs.mu, ip_outputs.alpha.primal,
                 ip_outputs.alpha.dual);

  if (!std::isnan(ip_outputs.mu_affine)) {
    // print only if filled...
    fmt::format_to(
        std::back_inserter(result), " Probe alphas: a_p = {}, a_d = {}, mu_affine = {}\n",
        ip_outputs.alpha_probe.primal, ip_outputs.alpha_probe.dual, ip_outputs.mu_affine);
  }

  // dump progress of individual KKT conditions
  fmt::format_to(std::back_inserter(result),
                 " KKT errors, L2:\n"
                 " r_dual = {} --> {}\n"
                 " r_comp = {} --> {}\n"
                 " r_p_eq = {} --> {}\n"
                 " r_p_ineq = {} --> {}\n",
                 kkt_initial.r_dual, kkt_final.r_dual, kkt_initial.r_comp, kkt_final.r_comp,
                 kkt_initial.r_primal_eq, kkt_final.r_primal_eq, kkt_initial.r_primal_ineq,
                 kkt_final.r_primal_ineq);
  return result;
}

enum class ColorCode : int { GREEN = 112, RED = 160, ORANGE = 202, NONE = -1 };

struct ColorFmt {
  constexpr ColorFmt(ColorCode code, bool enabled) noexcept : code(code), enabled(enabled) {}

  ColorCode code;
  bool enabled;
};

std::string NLSIteration::ToString(const bool use_color, const bool include_qp) const {
  std::string result;
  result.reserve(100);

  fmt::format_to(std::back_inserter(result), "Iteration # {}, state = {}, lambda = {}\n", iteration,
                 optimizer_state, lambda);
  fmt::format_to(std::back_inserter(result), "  f(0): {:.16e}, c(0): {:.16e}, total: {:.16e}\n",
                 errors_initial.f, errors_initial.equality, errors_initial.Total(penalty));
  if (qp_eigenvalues) {
    fmt::format_to(std::back_inserter(result),
                   "  min, max, |min| eig = {:.16e}, {:.16e}, {:.16e}\n", qp_eigenvalues->min,
                   qp_eigenvalues->max, qp_eigenvalues->abs_min);
  }
  fmt::format_to(std::back_inserter(result), "  penalty = {:.16f}\n", penalty);
  if (const QPInteriorPointSolverOutputs* ip =
          std::get_if<QPInteriorPointSolverOutputs>(&qp_outputs);
      ip) {
    fmt::format_to(std::back_inserter(result), "  QP-IP: {}, {}\n", ip->termination_state,
                   ip->iterations.size());
  } else if (const QPNullSpaceTerminationState* ns =
                 std::get_if<QPNullSpaceTerminationState>(&qp_outputs);
             ns) {
    fmt::format_to(std::back_inserter(result), "  QP: {}\n", *ns);
  }
  fmt::format_to(std::back_inserter(result), "  df/dalpha = {}, dc/dalpha = {}\n",
                 directional_derivatives.d_f, directional_derivatives.d_equality);
  fmt::format_to(
      std::back_inserter(result), "  Search result: {}{}{}\n",
      ColorFmt(step_result == StepSizeSelectionResult::SUCCESS ? ColorCode::GREEN : ColorCode::RED,
               use_color),
      step_result, ColorFmt(ColorCode::NONE, use_color));

  std::size_t i = 0;
  for (const LineSearchStep& step : line_search_steps) {
    fmt::format_to(std::back_inserter(result),
                   "  f({}): {:.16e}, c({}): {:.16e}, total: {:.16e}, alpha = {:.10f}\n", i,
                   step.errors.f, i, step.errors.equality, step.errors.Total(penalty), step.alpha);
    ++i;
  }
  if (const QPInteriorPointSolverOutputs* ip =
          std::get_if<QPInteriorPointSolverOutputs>(&qp_outputs);
      ip && include_qp) {
    for (const auto& iter : ip->iterations) {
      result += iter.ToString();
    }
  }
  return result;
}

std::size_t NLSSolverOutputs::NumQPIterations() const noexcept {
  return std::accumulate(iterations.begin(), iterations.end(), static_cast<std::size_t>(0),
                         [](std::size_t total, const NLSIteration& iteration) {
                           if (const QPInteriorPointSolverOutputs* ip =
                                   std::get_if<QPInteriorPointSolverOutputs>(&iteration.qp_outputs);
                               ip != nullptr) {
                             return total + ip->iterations.size();
                           } else {
                             return total + 1;
                           }
                         });
}

std::size_t NLSSolverOutputs::NumLineSearchSteps() const noexcept {
  return std::accumulate(iterations.begin(), iterations.end(), static_cast<std::size_t>(0),
                         [](std::size_t total, const NLSIteration& iteration) {
                           return total + iteration.line_search_steps.size();
                         });
}

std::size_t NLSSolverOutputs::NumFailedLineSearches() const noexcept {
  return std::accumulate(
      iterations.begin(), iterations.end(), static_cast<std::size_t>(0),
      [](std::size_t total, const NLSIteration& iteration) {
        return total + static_cast<std::size_t>(
                           iteration.step_result != StepSizeSelectionResult::SUCCESS &&
                           iteration.step_result != StepSizeSelectionResult::FIRST_ORDER_SATISFIED);
      });
}

inline ColorCode ColorFromTerminationState(const NLSTerminationState state) {
  switch (state) {
    case NLSTerminationState::SATISFIED_ABSOLUTE_TOL:
    case NLSTerminationState::SATISFIED_RELATIVE_TOL:
    case NLSTerminationState::SATISFIED_FIRST_ORDER_TOL: {
      return ColorCode::GREEN;
    }
    case NLSTerminationState::MAX_LAMBDA:
    case NLSTerminationState::MAX_ITERATIONS:
    case NLSTerminationState::USER_CALLBACK: {
      return ColorCode::ORANGE;
    }
    case NLSTerminationState::QP_INDEFINITE: {
      return ColorCode::RED;
    }
    default:
      break;
  }
  return ColorCode::NONE;
}

std::string NLSSolverOutputs::ToString(bool use_color, bool include_qp) const {
  std::string result;
  for (const auto& iter : iterations) {
    result += iter.ToString(use_color, include_qp);
  }
  fmt::format_to(std::back_inserter(result), "  termination = {}{}{}\n",
                 ColorFmt(ColorFromTerminationState(termination_state), use_color),
                 termination_state, ColorFmt(ColorCode::NONE, use_color));
  return result;
}

}  // namespace mini_opt

template <>
struct fmt::formatter<mini_opt::ColorFmt> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const mini_opt::ColorFmt c, FormatContext& ctx) const -> decltype(ctx.out()) {
    if (c.enabled) {
      if (c.code != mini_opt::ColorCode::NONE) {
        return fmt::format_to(ctx.out(), "\u001b[38;5;{}m", static_cast<int>(c.code));
      } else {
        constexpr std::string_view terminator = "\u001b[0m";
        return std::copy(terminator.begin(), terminator.end(), ctx.out());
      }
    }
    return ctx.out();
  }
};
