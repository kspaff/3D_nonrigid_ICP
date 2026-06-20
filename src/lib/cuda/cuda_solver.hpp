#pragma once

#include <memory>
#include <vector>

#include "src/lib/cuda/cuda_transform.hpp"

namespace nricp::cuda {

struct CudaCgSolveResult {
  bool success{};
  int iterations{};
  float relative_error{};
  std::vector<float> solution;
};

class CudaPcgWorkspace {
 public:
  CudaPcgWorkspace(int max_num_points, int num_grid_vals_per_component);
  ~CudaPcgWorkspace();

  CudaPcgWorkspace(const CudaPcgWorkspace&) = delete;
  CudaPcgWorkspace& operator=(const CudaPcgWorkspace&) = delete;

  int max_num_points() const;
  int num_grid_vals_per_component() const;
  int num_unknowns() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend CudaCgSolveResult SolveNormalEquationsPcg(CudaPcgWorkspace& workspace,
                                                   const std::vector<float>& moving_points_xyz,
                                                   const std::vector<float>& normal_x,
                                                   const std::vector<float>& normal_y,
                                                   const std::vector<float>& normal_z,
                                                   const std::vector<float>& point_to_plane_dists,
                                                   const std::vector<float>& regularization_diag,
                                                   const std::vector<float>& initial_guess,
                                                   const CudaGridShape& grid_shape,
                                                   int max_iterations,
                                                   float tolerance);
};

CudaCgSolveResult SolveNormalEquationsPcg(CudaPcgWorkspace& workspace,
                                          const std::vector<float>& moving_points_xyz,
                                          const std::vector<float>& normal_x,
                                          const std::vector<float>& normal_y,
                                          const std::vector<float>& normal_z,
                                          const std::vector<float>& point_to_plane_dists,
                                          const std::vector<float>& regularization_diag,
                                          const std::vector<float>& initial_guess,
                                          const CudaGridShape& grid_shape,
                                          int max_iterations,
                                          float tolerance);

CudaCgSolveResult SolveNormalEquationsPcg(const std::vector<float>& moving_points_xyz,
                                          const std::vector<float>& normal_x,
                                          const std::vector<float>& normal_y,
                                          const std::vector<float>& normal_z,
                                          const std::vector<float>& point_to_plane_dists,
                                          const std::vector<float>& regularization_diag,
                                          const std::vector<float>& initial_guess,
                                          const CudaGridShape& grid_shape,
                                          int num_grid_vals_per_component,
                                          int max_iterations,
                                          float tolerance);

}  // namespace nricp::cuda
