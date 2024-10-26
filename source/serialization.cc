// Copyright 2024 Gareth Cross
#include "mini_opt/serialization.hpp"

#ifdef MINI_OPT_SERIALIZATION

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mini_opt {

// Support optional types:
// https://json.nlohmann.me/features/arbitrary_types/#how-do-i-convert-third-party-types
template <typename T>
static void to_json(json& j, const std::optional<T>& x) {
  if (x) {
    j = x.value();
  } else {
    j = nullptr;
  }
}

template <typename T>
static void from_json(const json& j, std::optional<T>& x) {
  if (j.is_null()) {
    x.reset();
  } else {
    x = j.template get<T>();
  }
}

NLOHMANN_JSON_SERIALIZE_ENUM(OptimizerState,
                             {{OptimizerState::ATTEMPTING_RESTORE_LM, "ATTEMPTING_RESTORE_LM"},
                              {OptimizerState::NOMINAL, "NOMINAL"}});

NLOHMANN_JSON_SERIALIZE_ENUM(
    StepSizeSelectionResult,
    {{StepSizeSelectionResult::SUCCESS, "SUCCESS"},
     {StepSizeSelectionResult::MAX_ITERATIONS, "MAX_ITERATIONS"},
     {StepSizeSelectionResult::FIRST_ORDER_SATISFIED, "FIRST_ORDER_SATISFIED"},
     {StepSizeSelectionResult::POSITIVE_DERIVATIVE, "POSITIVE_DERIVATIVE"},
     {StepSizeSelectionResult::FAILURE_NON_FINITE_COST, "FAILURE_NON_FINITE_COST"},
     {StepSizeSelectionResult::FAILURE_INVALID_ALPHA, "FAILURE_INVALID_ALPHA"}});

NLOHMANN_JSON_SERIALIZE_ENUM(
    NLSTerminationState,
    {{NLSTerminationState::MAX_ITERATIONS, "MAX_ITERATIONS"},
     {NLSTerminationState::SATISFIED_ABSOLUTE_TOL, "SATISFIED_ABSOLUTE_TOL"},
     {NLSTerminationState::SATISFIED_RELATIVE_TOL, "SATISFIED_RELATIVE_TOL"},
     {NLSTerminationState::SATISFIED_FIRST_ORDER_TOL, "SATISFIED_FIRST_ORDER_TOL"},
     {NLSTerminationState::MAX_LAMBDA, "MAX_LAMBDA"},
     {NLSTerminationState::QP_INDEFINITE, "QP_INDEFINITE"},
     {NLSTerminationState::USER_CALLBACK, "USER_CALLBACK"}});

NLOHMANN_JSON_SERIALIZE_ENUM(QPInteriorPointTerminationState,
                             {{QPInteriorPointTerminationState::MAX_ITERATIONS, "MAX_ITERATIONS"},
                              {QPInteriorPointTerminationState::SATISFIED_KKT_TOL,
                               "SATISFIED_KKT_TOL"}});

NLOHMANN_JSON_SERIALIZE_ENUM(QPNullSpaceTerminationState,
                             {{QPNullSpaceTerminationState::SUCCESS, "SUCCESS"},
                              {QPNullSpaceTerminationState::NOT_POSITIVE_DEFINITE,
                               "NOT_POSITIVE_DEFINITE"}});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Errors, f, equality);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(QPEigenvalues, min, max, abs_min);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DirectionalDerivatives, d_f, d_equality);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LineSearchStep, alpha, errors);

// QP structs:
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AlphaValues, primal, dual);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IPIterationOutputs, mu, alpha, alpha_probe, mu_affine);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(KKTError, r_dual, r_comp, r_primal_eq, r_primal_ineq);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(QPInteriorPointIteration, kkt_initial, kkt_final, ip_outputs);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(QPLagrangeMultipliers, min, l_infinity);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(QPInteriorPointSolverOutputs, termination_state, iterations,
                                   lagrange_multipliers);

static void to_json(
    json& j, const std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs>& var) {
  std::visit([&j](const auto& x) { j = x; }, var);
}

static void from_json(
    const json& j, std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs>& var) {
  if (j.contains("termination_state") && j.contains("iterations")) {
    var = j.get<QPInteriorPointSolverOutputs>();
  } else {
    var = j.get<QPNullSpaceTerminationState>();
  }
}
}  // namespace mini_opt

namespace nlohmann {
using namespace mini_opt;

// NLSIteration is not default constructible - we define a manual serializer for it here.
template <>
struct adl_serializer<NLSIteration> {
  static NLSIteration from_json(const json& j) {
    return NLSIteration(
        j["iteration"].get<int>(), j["optimizer_state"].get<OptimizerState>(),
        j["lambda"].get<double>(), j["errors_initial"].get<Errors>(),
        j["qp_outputs"]
            .get<std::variant<QPNullSpaceTerminationState, QPInteriorPointSolverOutputs>>(),
        j["qp_eigenvalues"].get<std::optional<QPEigenvalues>>(),
        j["directional_derivatives"].get<DirectionalDerivatives>(), j["penalty"].get<double>(),
        j["step_result"].get<StepSizeSelectionResult>(),
        j["line_search_steps"].get<std::vector<LineSearchStep>>());
  }

  static void to_json(json& j, const NLSIteration& iter) {
    j["iteration"] = iter.iteration;
    j["optimizer_state"] = iter.optimizer_state;
    j["lambda"] = iter.lambda;
    j["errors_initial"] = iter.errors_initial;
    j["qp_outputs"] = iter.qp_outputs;
    j["qp_eigenvalues"] = iter.qp_eigenvalues;
    j["directional_derivatives"] = iter.directional_derivatives;
    j["penalty"] = iter.penalty;
    j["step_result"] = iter.step_result;
    j["line_search_steps"] = iter.line_search_steps;
  }
};

NLSSolverOutputs adl_serializer<NLSSolverOutputs, void>::from_json(const json& j) {
  return NLSSolverOutputs(j["termination_state"].get<NLSTerminationState>(),
                          j["iterations"].get<std::vector<NLSIteration>>());
}

void adl_serializer<NLSSolverOutputs, void>::to_json(json& j, const NLSSolverOutputs& outputs) {
  j["termination_state"] = outputs.termination_state;
  j["iterations"] = outputs.iterations;
}

}  // namespace nlohmann

#endif  // MINI_OPT_SERIALIZATION
