#include "mini_opt/values.hpp"
#include "mini_opt/scatter.hpp"

namespace mini_opt {

values values::retract(const Eigen::VectorXd& dx) const {
  values out;
  int row = 0;
  for (const auto& [k, v] : storage_) {
    const int dim = v.tangent_dimension();
    MINI_OPT_ASSERT_LE(row + dim, dx.size());
    out.insert(k, v.retract({&dx[row], static_cast<std::size_t>(dim)}));
    row += dim;
  }
  return out;
}

Eigen::VectorXd values::local_coordinates(const values& x, const scatter& s) const {
  Eigen::VectorXd out;
  out.resize(s.total_dimension());
  out.setZero();
  for (const auto& [k, v] : storage_) {
    const int dim = v.tangent_dimension();
    const int output_row = s.at(k);
    v.local_coordinates(x.at(k), {&out[output_row], static_cast<std::size_t>(dim)});
  }
  return out;
}

Eigen::VectorXd values::local_coordinates(const values& x) const {
  int total_dim = 0;
  for (const auto& [k, v] : storage_) {
    total_dim += v.tangent_dimension();
  }

  Eigen::VectorXd out;
  out.resize(total_dim);
  total_dim = 0;
  for (const auto& [k, v] : storage_) {
    const int dim = v.tangent_dimension();
    v.local_coordinates(x.at(k), {&out[total_dim], static_cast<std::size_t>(dim)});
    total_dim += dim;
  }
  return out;
}

}  // namespace mini_opt
