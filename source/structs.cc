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
    case StepSizeSelectionResult::FAILURE_MAX_ITERATIONS:
      stream << "FAILURE_MAX_ITERATIONS";
      break;
    case StepSizeSelectionResult::FAILURE_POSITIVE_DERIVATIVE:
      stream << "FAILURE_POSITIVE_DERIVATIVE";
      break;
    case StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED:
      stream << "FAILURE_FIRST_ORDER_SATISFIED";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const NLSTerminationState state) {
  switch (state) {
    case NLSTerminationState::NONE:
      stream << "NONE";
      break;
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

enum class ColorCode : int { GREEN = 112, RED = 160, NONE = -1 };

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
  fmt::format_to(std::back_inserter(result), "  termination = {}{}{}\n",
                 ColorFmt(termination_state != NLSTerminationState::MAX_LAMBDA &&
                                  termination_state != NLSTerminationState::MAX_ITERATIONS
                              ? ColorCode::GREEN
                              : ColorCode::RED,
                          use_color),
                 termination_state, ColorFmt(ColorCode::NONE, use_color));
  fmt::format_to(std::back_inserter(result), "  penalty = {:.16f}\n", penalty);
  if (qp_outputs) {
    fmt::format_to(std::back_inserter(result), "  QP: {}, {}\n", qp_outputs->termination_state,
                   qp_outputs->iterations.size());
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
  if (include_qp && qp_outputs) {
    for (const auto& iter : qp_outputs->iterations) {
      result += iter.ToString();
    }
  }
  return result;
}

std::size_t NLSSolverOutputs::NumQPIterations() const noexcept {
  return std::accumulate(iterations.begin(), iterations.end(), static_cast<std::size_t>(0),
                         [](std::size_t total, const NLSIteration& iteration) {
                           return total + (iteration.qp_outputs.has_value()
                                               ? iteration.qp_outputs->iterations.size()
                                               : 1);
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
        return total +
               static_cast<std::size_t>(iteration.step_result != StepSizeSelectionResult::SUCCESS &&
                                        iteration.step_result !=
                                            StepSizeSelectionResult::FAILURE_FIRST_ORDER_SATISFIED);
      });
}

std::string NLSSolverOutputs::ToString(bool use_color, bool include_qp) const {
  std::string result;
  for (const auto& iter : iterations) {
    result += iter.ToString(use_color, include_qp);
  }
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
