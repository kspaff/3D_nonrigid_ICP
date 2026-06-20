#include <gtest/gtest.h>

#include "src/lib/cuda/cuda_jacobian.hpp"
#include "src/lib/cuda/cuda_runtime.hpp"
#include "src/lib/cuda/cuda_transform.hpp"
#include "src/lib/pt_cloud.hpp"

namespace {

std::vector<float> FlattenGrid(const TranslationGrid& grid) {
  std::vector<float> coefficients;
  coefficients.reserve(static_cast<size_t>(grid.num_grid_vals()));
  const auto& grid_vals = grid.grid_vals();
  for (int x = 0; x < grid.x_num_voxels() + 1; ++x) {
    for (int y = 0; y < grid.y_num_voxels() + 1; ++y) {
      for (int z = 0; z < grid.z_num_voxels() + 1; ++z) {
        const auto& vals = grid_vals[x][y][z];
        coefficients.push_back(vals.f);
        coefficients.push_back(vals.fx);
        coefficients.push_back(vals.fy);
        coefficients.push_back(vals.fz);
        coefficients.push_back(vals.fxy);
        coefficients.push_back(vals.fxz);
        coefficients.push_back(vals.fyz);
        coefficients.push_back(vals.fxyz);
      }
    }
  }
  return coefficients;
}

std::vector<float> FlattenPoints(const MatrixX3& points) {
  std::vector<float> points_xyz;
  points_xyz.reserve(static_cast<size_t>(points.rows()) * 3);
  for (Eigen::Index row = 0; row < points.rows(); ++row) {
    points_xyz.push_back(points(row, 0));
    points_xyz.push_back(points(row, 1));
    points_xyz.push_back(points(row, 2));
  }
  return points_xyz;
}

std::vector<float> CopyVector(const VectorX& values) {
  std::vector<float> copied;
  copied.reserve(static_cast<size_t>(values.size()));
  for (Eigen::Index row = 0; row < values.size(); ++row) {
    copied.push_back(values(row));
  }
  return copied;
}

}  // namespace

TEST(CudaRuntime, UsesCuda13AndFindsDevice) {
  const auto info = nricp::cuda::QueryCudaRuntime();

  EXPECT_EQ(info.runtime_version / 1000, 13);
  EXPECT_GE(info.device_count, 1);
  EXPECT_FALSE(info.first_device_name.empty());
  EXPECT_GE(info.first_device_compute_major, 1);
  EXPECT_FALSE(nricp::cuda::CudaVersionString(info.runtime_version).empty());
}

TEST(CudaTransform, MatchesCpuTricubicTransformEvaluator) {
  RowVector3 origin{};
  origin << Scalar{0}, Scalar{0}, Scalar{0};
  TranslationGrid x_grid;
  TranslationGrid y_grid;
  TranslationGrid z_grid;
  x_grid.Initialize(origin, 1, 1, 1, Scalar{1}, 0);
  y_grid.Initialize(origin, 1, 1, 1, Scalar{1}, x_grid.num_grid_vals());
  z_grid.Initialize(origin, 1, 1, 1, Scalar{1}, x_grid.num_grid_vals() + y_grid.num_grid_vals());

  for (int corner = 0; corner < 8; ++corner) {
    const int x = corner & 1;
    const int y = (corner >> 1) & 1;
    const int z = (corner >> 2) & 1;

    GridVals x_vals{};
    x_vals.f = Scalar{0.01f * (1 + corner)};
    x_vals.fx = Scalar{0.02f * (1 + corner)};
    x_vals.fy = Scalar{-0.01f * (1 + corner)};
    x_vals.fz = Scalar{0.015f * (1 + corner)};
    x_vals.fxy = Scalar{0.003f * (1 + corner)};
    x_vals.fxz = Scalar{-0.002f * (1 + corner)};
    x_vals.fyz = Scalar{0.004f * (1 + corner)};
    x_vals.fxyz = Scalar{0.001f * (1 + corner)};
    x_grid.UpdateVoxelGridVals(x, y, z, x_vals);

    GridVals y_vals = x_vals;
    y_vals.f += Scalar{0.05f};
    y_vals.fx -= Scalar{0.01f};
    y_grid.UpdateVoxelGridVals(x, y, z, y_vals);

    GridVals z_vals = x_vals;
    z_vals.f -= Scalar{0.03f};
    z_vals.fz += Scalar{0.02f};
    z_grid.UpdateVoxelGridVals(x, y, z, z_vals);
  }

  MatrixX3 points(5, 3);
  points << Scalar{0}, Scalar{0}, Scalar{0}, Scalar{0.25f}, Scalar{0.5f}, Scalar{0.75f},
      Scalar{0.5f}, Scalar{0.25f}, Scalar{0.125f}, Scalar{0.9f}, Scalar{0.1f}, Scalar{0.6f},
      Scalar{0.999f}, Scalar{0.999f}, Scalar{0.999f};
  const auto cpu_transformed = ApplyTranslationGrids(points, x_grid, y_grid, z_grid);

  const auto points_xyz = FlattenPoints(points);

  const nricp::cuda::CudaGridShape grid_shape{0.0f, 0.0f, 0.0f, 1, 1, 1, 1.0f};
  const auto gpu_transformed = nricp::cuda::ApplyTranslationGrids(
      points_xyz, FlattenGrid(x_grid), FlattenGrid(y_grid), FlattenGrid(z_grid), grid_shape);

  ASSERT_EQ(gpu_transformed.size(), points_xyz.size());
  for (Eigen::Index row = 0; row < points.rows(); ++row) {
    for (Eigen::Index col = 0; col < 3; ++col) {
      EXPECT_NEAR(gpu_transformed[static_cast<size_t>(row * 3 + col)], cpu_transformed(row, col),
                  1e-5f);
    }
  }
}

TEST(CudaFitting, WeightedJacobianMatchesCpuTriplets) {
  RowVector3 origin{};
  origin << Scalar{0}, Scalar{0}, Scalar{0};
  TranslationGrid x_grid;
  TranslationGrid y_grid;
  TranslationGrid z_grid;
  x_grid.Initialize(origin, 1, 1, 1, Scalar{1}, 0);
  y_grid.Initialize(origin, 1, 1, 1, Scalar{1}, x_grid.num_grid_vals());
  z_grid.Initialize(origin, 1, 1, 1, Scalar{1}, x_grid.num_grid_vals() + y_grid.num_grid_vals());

  MatrixX3 points(3, 3);
  points << Scalar{0.125f}, Scalar{0.25f}, Scalar{0.5f}, Scalar{0.5f}, Scalar{0.75f}, Scalar{0.25f},
      Scalar{0.875f}, Scalar{0.125f}, Scalar{0.625f};
  VectorX normal_x(3);
  VectorX normal_y(3);
  VectorX normal_z(3);
  normal_x << Scalar{1.0f}, Scalar{-0.5f}, Scalar{0.25f};
  normal_y << Scalar{0.0f}, Scalar{0.75f}, Scalar{-1.0f};
  normal_z << Scalar{0.5f}, Scalar{0.25f}, Scalar{0.125f};

  std::vector<Triplet> cpu_triplets;
  cpu_triplets.reserve(static_cast<size_t>(points.rows()) * 64 * 3);
  x_grid.AppendJTriplets(points, normal_x, cpu_triplets);
  y_grid.AppendJTriplets(points, normal_y, cpu_triplets);
  z_grid.AppendJTriplets(points, normal_z, cpu_triplets);

  const nricp::cuda::CudaGridShape grid_shape{0.0f, 0.0f, 0.0f, 1, 1, 1, 1.0f};
  const auto gpu_jacobian = nricp::cuda::BuildWeightedJacobian(
      FlattenPoints(points), CopyVector(normal_x), CopyVector(normal_y), CopyVector(normal_z),
      grid_shape, x_grid.num_grid_vals());

  ASSERT_EQ(gpu_jacobian.num_rows, points.rows());
  ASSERT_EQ(gpu_jacobian.num_cols,
            x_grid.num_grid_vals() + y_grid.num_grid_vals() + z_grid.num_grid_vals());
  ASSERT_EQ(gpu_jacobian.triplets.size(), cpu_triplets.size());
  for (size_t i = 0; i < cpu_triplets.size(); ++i) {
    EXPECT_EQ(gpu_jacobian.triplets[i].row, cpu_triplets[i].row());
    EXPECT_EQ(gpu_jacobian.triplets[i].col, cpu_triplets[i].col());
    EXPECT_NEAR(gpu_jacobian.triplets[i].value, cpu_triplets[i].value(), 1e-6f);
  }
}
