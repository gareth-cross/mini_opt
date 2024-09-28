#include "mini_opt/structs.hpp"

#include <algorithm>
#include <ostream>

namespace mini_opt {

double KKTError::Max() const {
  return std::max(r_dual, std::max(r_comp, std::max(r_primal_eq, r_primal_ineq)));
}

std::ostream& operator<<(std::ostream& stream, const QPTerminationState& state) {
  switch (state) {
    case QPTerminationState::SATISFIED_KKT_TOL:
      stream << "SATISFIED_KKT_TOL";
      break;
    case QPTerminationState::MAX_ITERATIONS:
      stream << "MAX_ITERATIONS";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const LineSearchStrategy& strategy) {
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

std::ostream& operator<<(std::ostream& stream, const OptimizerState& v) {
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

std::ostream& operator<<(std::ostream& stream, const StepSizeSelectionResult& result) {
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

std::ostream& operator<<(std::ostream& stream, const NLSTerminationState& state) {
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

}  // namespace mini_opt
