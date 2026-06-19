#pragma once

#include <vector>

namespace nricp::cuda {

struct CudaGridShape {
  float origin_x{};
  float origin_y{};
  float origin_z{};
  int x_num_voxels{};
  int y_num_voxels{};
  int z_num_voxels{};
  float voxel_size{};
};

std::vector<float> ApplyTranslationGrids(const std::vector<float>& points_xyz,
                                         const std::vector<float>& x_coefficients,
                                         const std::vector<float>& y_coefficients,
                                         const std::vector<float>& z_coefficients,
                                         const CudaGridShape& grid_shape);

}  // namespace nricp::cuda
