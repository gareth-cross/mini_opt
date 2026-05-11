#pragma once
#include <algorithm>
#include <cstdint>
#include <string_view>  // std::hash
#include <type_traits>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "mini_opt/hashing.hpp"

namespace mini_opt {

static constexpr std::size_t key_size = 16;
static constexpr std::size_t key_alignment = 16;

template <typename T>
struct is_valid_key
    : std::conditional_t<std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T> &&
                             sizeof(T) <= key_size && alignof(T) <= key_alignment,
                         std::true_type, std::false_type> {};

template <typename T>
static constexpr bool is_valid_key_v = is_valid_key<T>::value;

namespace internal {
constexpr std::uint64_t u64_from_bytes(const std::uint8_t* bytes) noexcept {
  std::uint64_t result = 0;
  for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
    result <<= 8;
    result |= bytes[i];
  }
  return result;
}
}  // namespace internal

// A key for a variable in the optimization.
class key {
 public:
  template <typename T, typename = std::enable_if_t<is_valid_key_v<T>>>
  key(const T& value) noexcept(std::is_nothrow_copy_constructible_v<T>) {
    new (&data_) T(value);
    for (std::size_t i = sizeof(T); i < key_size; ++i) {
      ptr()[i] = 0;
    }
  }

  constexpr bool operator==(const key& other) const noexcept {
    return std::equal(ptr(), ptr() + key_size, other.ptr());
  }

  constexpr bool operator!=(const key& other) const noexcept { return !(*this == other); }

  constexpr std::size_t hash() const noexcept {
    static_assert(key_size == 16);
    const std::uint8_t* p = ptr();
    return hash_combine(internal::u64_from_bytes(p),
                        internal::u64_from_bytes(p + sizeof(std::uint64_t)));
  }

 private:
  friend struct fmt::formatter<mini_opt::key>;

  constexpr const std::uint8_t* ptr() const noexcept {
    return static_cast<const std::uint8_t*>(static_cast<const void*>(&data_));
  }

  constexpr std::uint8_t* ptr() noexcept {
    return static_cast<std::uint8_t*>(static_cast<void*>(&data_));
  }

  std::aligned_storage_t<key_size, key_alignment> data_;
};

}  // namespace mini_opt

template <>
struct std::hash<mini_opt::key> {
  constexpr std::size_t operator()(const mini_opt::key& k) const noexcept { return k.hash(); }
};

template <>
struct fmt::formatter<mini_opt::key> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const mini_opt::key& k, FormatContext& ctx) const {
    const std::uint8_t* p = k.ptr();
    return fmt::format_to(ctx.out(), "key([{}])", fmt::join(p, p + mini_opt::key_size, ","));
  }
};
