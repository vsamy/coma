/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace Eigen {

using Vector6f = Matrix<float, 6, 1>;
using Vector6d = Matrix<double, 6, 1>;
using Matrix6f = Matrix<float, 6, 6>;
using Matrix6d = Matrix<double, 6, 6>;

} // namespace Eigen

namespace coma {

constexpr int Dynamic = Eigen::Dynamic; // == -1
using Eigen::Index;

} // namespace coma
