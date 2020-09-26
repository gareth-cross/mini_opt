#include "mini_opt/structs.hpp"

#include <ostream>

namespace mini_opt {

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
  }
  return stream;
}

}  // namespace mini_opt