// Copyright 2021 Gareth Cross
#include "mini_opt/residual.hpp"

namespace mini_opt {

double Residual::QuadraticError(const Eigen::VectorXd& params) const {
  const int dim = Dimension();
  Eigen::VectorXd b(dim);
  ErrorVector(params, b.head(dim));
  return 0.5 * b.squaredNorm();
}

}  // namespace mini_opt
