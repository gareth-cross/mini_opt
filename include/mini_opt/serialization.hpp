// Copyright 2024 Gareth Cross
#pragma once
#include "mini_opt/structs.hpp"

#ifdef MINI_OPT_SERIALIZATION
#include <nlohmann/json_fwd.hpp>

namespace nlohmann {
template <>
struct adl_serializer<mini_opt::NLSSolverOutputs> {
  static mini_opt::NLSSolverOutputs from_json(const json& j);
  static void to_json(json& j, const mini_opt::NLSSolverOutputs& outputs);
};
}  // namespace nlohmann

#endif  // MINI_OPT_SERIALIZATION
