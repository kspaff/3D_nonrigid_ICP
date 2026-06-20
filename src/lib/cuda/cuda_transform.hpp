#pragma once

#include <memory>
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

class CudaTransformWorkspace {
 public:
  CudaTransformWorkspace(const std::vector<float>& points_xyz, int num_grid_vals_per_component);
  ~CudaTransformWorkspace();

  CudaTransformWorkspace(const CudaTransformWorkspace&) = delete;
  CudaTransformWorkspace& operator=(const CudaTransformWorkspace&) = delete;

  int num_points() const;
  int num_grid_vals_per_component() const;
  const std::vector<float>& Apply(const std::vector<float>& coefficients_xyz,
                                  const CudaGridShape& grid_shape);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nricp::cuda
