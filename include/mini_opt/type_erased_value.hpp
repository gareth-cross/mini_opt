#pragma once
#include <memory>
#include <span>
#include <typeindex>

#include "mini_opt/assertions.hpp"
#include "mini_opt/manifold.hpp"

namespace mini_opt {

// A value used in the optimization. This is a type-erased wrapper around any manifold type that the
// user might want to use as an optimization variable.
class type_erased_value {
 public:
  // Construct from manifold type `T`.
  template <typename T, typename U = std::decay_t<T>,
            typename =
                std::enable_if_t<implements_manifold_v<U> && !std::is_same_v<U, type_erased_value>>>
  explicit type_erased_value(T&& value) noexcept(
      std::is_nothrow_constructible_v<U, decltype(value)>)
      : impl_(std::make_unique<const model<U>>(std::forward<T>(value))) {}

  type_erased_value(const type_erased_value& other) : impl_(other.impl_->clone()) {}
  type_erased_value(type_erased_value&&) noexcept = default;

  type_erased_value& operator=(const type_erased_value& other) {
    if (this != &other) {
      impl_ = other.impl_->clone();
    }
    return *this;
  }
  type_erased_value& operator=(type_erased_value&&) noexcept = default;

  int tangent_dimension() const { return impl_->tangent_dimension(); }

  // Return self [+] dx.
  type_erased_value retract(std::span<const double> dx) const { return impl_->retract(dx); }

  // Write self [-] x into `dx`.
  void local_coordinates(const type_erased_value& x, std::span<double> dx) const {
    impl_->local_coordinates(x, dx);
  }

  // Cast to type `T`. Throw if the cast is invalid.
  template <typename T>
  operator const T&() const {
    MINI_OPT_ASSERT(impl_ && impl_->type() == typeid(T), "Invalid cast from type {} to type {}",
                    impl_ ? impl_->type().name() : "null", typeid(T).name());
    return static_cast<const model<T>*>(impl_.get())->value();
  }

 private:
  friend class values;

  bool is_null() const { return !static_cast<bool>(impl_); }

  class concept_ {
   public:
    template <typename T>
    explicit concept_(std::in_place_type_t<T>) : type_(typeid(T)) {}
    virtual ~concept_() = default;

    std::type_index type() const { return type_; }

    virtual std::unique_ptr<concept_> clone() const = 0;

    virtual int tangent_dimension() const = 0;

    // self [+] dx.
    virtual type_erased_value retract(std::span<const double> dx) const = 0;

    // self [-] x.
    virtual void local_coordinates(const type_erased_value& x, std::span<double> dx) const = 0;

   private:
    std::type_index type_;
  };

  template <typename T>
  class model final : public concept_ {
   public:
    using manifold = manifold_trait<T>;

    template <typename U>
    explicit model(U&& value) noexcept(std::is_nothrow_constructible_v<T, decltype(value)>)
        : concept_(std::in_place_type<T>), value_(std::forward<U>(value)) {}

    const auto& value() const { return value_; }

    std::unique_ptr<concept_> clone() const override { return std::make_unique<model<T>>(*this); }

    int tangent_dimension() const override { return manifold::tangent_dimension(value_); }

    // self [+] dx.
    type_erased_value retract(std::span<const double> dx) const override {
      const int dx_dim = manifold::tangent_dimension(value_);
      MINI_OPT_ASSERT_GE(dx.size(), static_cast<std::size_t>(dx_dim));

      return type_erased_value(manifold::retract(
          value_, Eigen::Map<const typename manifold::tangent_vector>(dx.data(), dx_dim)));
    }

    // self [-] x.
    void local_coordinates(const type_erased_value& x, std::span<double> dx) const override {
      const int dx_dim = manifold::tangent_dimension(value_);
      MINI_OPT_ASSERT_GE(dx.size(), static_cast<std::size_t>(dx_dim));

      Eigen::Map<typename manifold::tangent_vector>(dx.data(), dx_dim) =
          manifold::local_coordinates(value_, static_cast<const T&>(x));
    }

   private:
    T value_;
  };

  // TODO: SBO.
  std::unique_ptr<const concept_> impl_;
};

}  // namespace mini_opt
