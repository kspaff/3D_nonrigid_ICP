#pragma once

#include <vector>

#include "src/lib/cuda/cuda_transform.hpp"

namespace nricp::cuda {

struct CudaWeightedJacobianTriplet {
  int row{};
  int col{};
  float value{};
};

struct CudaWeightedJacobian {
  int num_rows{};
  int num_cols{};
  std::vector<CudaWeightedJacobianTriplet> triplets;
};

CudaWeightedJacobian BuildWeightedJacobian(const std::vector<float>& moving_points_xyz,
                                           const std::vector<float>& normal_x,
                                           const std::vector<float>& normal_y,
                                           const std::vector<float>& normal_z,
                                           const CudaGridShape& grid_shape,
                                           int num_grid_vals_per_component);

}  // namespace nricp::cuda
