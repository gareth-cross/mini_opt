// Copyright (c) 2024 Gareth Cross
#pragma once
#include <exception>
#include <iterator>  // back_inserter
#include <string>
#include <string_view>

#include <fmt/core.h>

namespace mini_opt::assert {

// Generates an exception w/ a formatted string.
template <typename... Ts>
std::string format_assert(const std::string_view condition, const std::string_view file,
                          const int line, const std::string_view reason_fmt = {}, Ts&&... args) {
  std::string err = fmt::format("Assertion failed: {}\nFile: {}\nLine: {}", condition, file, line);
  if (!reason_fmt.empty()) {
    err.append("\nDetails: ");
    fmt::vformat_to(std::back_inserter(err), reason_fmt, fmt::make_format_args(args...));
  }
  return err;
}

// Version that prints args A & B as well. For binary comparisons.
template <typename A, typename B, typename... Ts>
std::string format_assert_binary(const std::string_view condition, const std::string_view file,
                                 const int line, const std::string_view a_name, A&& a,
                                 const std::string_view b_name, B&& b,
                                 const std::string_view reason_fmt = {}, Ts&&... args) {
  std::string err = fmt::format(
      "Assertion failed: {}\n"
      "Operands are: `{}` = {}, `{}` = {}\n"
      "File: {}\nLine: {}",
      condition, a_name, std::forward<A>(a), b_name, std::forward<B>(b), file, line);
  if (!reason_fmt.empty()) {
    err.append("\nDetails: ");
    fmt::vformat_to(std::back_inserter(err), reason_fmt, fmt::make_format_args(args...));
  }
  return err;
}

// The exception type that the library throws.
class default_error : public std::exception {
 public:
  explicit default_error(std::string message) noexcept : message_(std::move(message)) {}

  // Return the message string.
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

}  // namespace mini_opt::assert

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif  // __clang__

// Assertion macros.
#define _F_ASSERT_IMPL(cond, file, line, handler, ...)                                  \
  do {                                                                                  \
    if (!static_cast<bool>(cond)) {                                                     \
      throw mini_opt::assert::default_error(handler(#cond, file, line, ##__VA_ARGS__)); \
    }                                                                                   \
  } while (false)

// Macro to use when defining an assertion.
#define F_ASSERT(cond, ...) \
  _F_ASSERT_IMPL(cond, __FILE__, __LINE__, assert::format_assert, ##__VA_ARGS__)

#define F_ASSERT_EQ(a, b, ...)                                                               \
  _F_ASSERT_IMPL((a) == (b), __FILE__, __LINE__, assert::format_assert_binary, #a, a, #b, b, \
                 ##__VA_ARGS__)

#define F_ASSERT_NE(a, b, ...)                                                               \
  _F_ASSERT_IMPL((a) != (b), __FILE__, __LINE__, assert::format_assert_binary, #a, a, #b, b, \
                 ##__VA_ARGS__)

#define F_ASSERT_LT(a, b, ...)                                                              \
  _F_ASSERT_IMPL((a) < (b), __FILE__, __LINE__, assert::format_assert_binary, #a, a, #b, b, \
                 ##__VA_ARGS__)

#define F_ASSERT_GT(a, b, ...)                                                              \
  _F_ASSERT_IMPL((a) > (b), __FILE__, __LINE__, assert::format_assert_binary, #a, a, #b, b, \
                 ##__VA_ARGS__)

#define F_ASSERT_LE(a, b, ...)                                                               \
  _F_ASSERT_IMPL((a) <= (b), __FILE__, __LINE__, assert::format_assert_binary, #a, a, #b, b, \
                 ##__VA_ARGS__)

#define F_ASSERT_GE(a, b, ...)                                                               \
  _F_ASSERT_IMPL((a) >= (b), __FILE__, __LINE__, assert::format_assert_binary, #a, a, #b, b, \
                 ##__VA_ARGS__)

#ifdef __clang__
#pragma clang diagnostic pop
#endif  // __clang__
