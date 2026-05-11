#pragma once
#include <unordered_map>

#include "mini_opt/assertions.hpp"
#include "mini_opt/key.hpp"
#include "mini_opt/values.hpp"

namespace mini_opt {

class scatter {
 public:
  scatter() = default;

  // Build scatter from values.
  explicit scatter(const values& values) {
    map_.reserve(values.size());
    int row = 0;
    for (const auto& [k, v] : values) {
      map_[k] = row;
      row += v.tangent_dimension();
    }
    size_ = row;
  }

  void reserve(std::size_t size) { map_.reserve(size); }

  int at(const key& key) const {
    const auto it = map_.find(key);
    MINI_OPT_ASSERT(it != map_.end(), "Key not found: {}", key);
    return it->second;
  }

  int total_dimension() const { return size_; }

 private:
  std::unordered_map<key, int> map_;
  int size_{0};
};

}  // namespace mini_opt
