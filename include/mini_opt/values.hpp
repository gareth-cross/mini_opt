#pragma once
#include "mini_opt/key.hpp"
#include "mini_opt/type_erased_value.hpp"

namespace mini_opt {

class scatter;

// A collection of optimization values.
class values {
 public:
  using storage_type = std::unordered_map<key, type_erased_value>;

  void reserve(std::size_t size) { storage_.reserve(size); }

  std::size_t size() const { return storage_.size(); }

  // Insert a value.
  template <typename T>
  void insert(const key& key, T&& value) {
    type_erased_value v(std::forward<T>(value));
    auto [it, did_insert] = storage_.emplace(key, std::move(v));
    if (!did_insert) {
      MINI_OPT_ASSERT(!v.is_null());
      it->second = std::move(v);
    }
  }

  // Retrieve a value.
  template <typename T>
  const T& at(const key& key) const {
    const auto it = storage_.find(key);
    MINI_OPT_ASSERT(it != storage_.end(), "Key not found: {}", key);
    return static_cast<const T&>(it->second);
  }

  // Retrieve type-erased value.
  const type_erased_value& at(const key& key) const {
    const auto it = storage_.find(key);
    MINI_OPT_ASSERT(it != storage_.end(), "Key not found: {}", key);
    return it->second;
  }

  // Retract all values in this collection.
  values retract(const Eigen::VectorXd& dx) const;

  // Compute self [-] x for all values in this collection.
  // Vector is ordered according to the scatter.
  Eigen::VectorXd local_coordinates(const values& x, const scatter& s) const;

  // Compute self [-] x for all values in this collection.
  Eigen::VectorXd local_coordinates(const values& x) const;

  void swap(values& other) noexcept { storage_.swap(other.storage_); }

  auto begin() const { return storage_.begin(); }
  auto end() const { return storage_.end(); }

  auto cbegin() const { return storage_.cbegin(); }
  auto cend() const { return storage_.cend(); }

 private:
  storage_type storage_;
};

}  // namespace mini_opt
