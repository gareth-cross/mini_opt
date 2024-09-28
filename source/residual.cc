// Copyright 2021 Gareth Cross
#include "mini_opt/residual.hpp"

namespace mini_opt {

double ResidualBase::QuadraticError(const Eigen::VectorXd& params) const {
  Eigen::VectorXd b(Dimension());
  ErrorVector(params, b.head(Dimension()));
  return b.squaredNorm();
}

}  // namespace mini_opt
