#pragma once
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mini_opt {

// True if all types are void.
template <typename... Ts>
constexpr bool all_void_v = std::conjunction_v<std::is_same<void, Ts>...>;

// True if any type is void.
template <typename... Ts>
constexpr bool any_void_v = std::disjunction_v<std::is_same<void, Ts>...>;

// Compile-time for loop over an index sequence, invoking a function `F` on each index. If all
// return types are void, returns void. Otherwise, returns a tuple of the return types.
template <typename F, std::size_t... I>
auto index_sequence_for(F&& f, std::index_sequence<I...>) {
  // Either every invocation returns void, or none of them return void.
  static_assert(all_void_v<std::invoke_result_t<F, std::integral_constant<std::size_t, I>>...> ||
                    !any_void_v<std::invoke_result_t<F, std::integral_constant<std::size_t, I>>...>,
                "Cannot place void type into a tuple.");

  if constexpr (all_void_v<std::invoke_result_t<F, std::integral_constant<std::size_t, I>>...>) {
    // All return types are void.
    (std::invoke(std::forward<F>(f), std::integral_constant<std::size_t, I>{}), ...);
  } else {
    // Not all return types are void, return a tuple.
    // Use initializer list syntax to ensure order of execution is left to right.
    return std::tuple<std::invoke_result_t<F, std::integral_constant<std::size_t, I>>...>{
        std::invoke(std::forward<F>(f), std::integral_constant<std::size_t, I>{})...};
  }
}

}  // namespace mini_opt
