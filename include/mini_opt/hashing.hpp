#include <cstdint>

namespace mini_opt {

constexpr std::uint64_t hash_mix(std::uint64_t x) {
  const std::uint64_t m = 0xe9846afb1a615dull;
  x ^= x >> 32;
  x *= m;
  x ^= x >> 32;
  x *= m;
  x ^= x >> 28;
  return x;
}

constexpr std::uint64_t hash_combine(std::uint64_t seed, std::uint64_t v) {
  return hash_mix(seed + 0x9e3779b9 + v);
}

}  // namespace mini_opt
