#pragma once
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <Eigen/Core>

// https://github.com/fmtlib/fmt/issues/4123
template <typename T>
struct fmt::is_range<T, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<T>, T>, char>>
    : std::false_type {};

// Allow formatting of Eigen matrices.
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<T>, T>, char>> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }

  template <typename Arg, typename FormatContext>
  auto format(const Arg& m, FormatContext& ctx) const -> decltype(ctx.out()) {
    const Eigen::IOFormat heavy(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");
    std::stringstream ss;
    ss << m.format(heavy);
    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};
