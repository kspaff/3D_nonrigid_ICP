#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "src/lib/cuda/cuda_jacobian.hpp"

namespace {

void CheckCuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

__device__ float CubicHermiteBasis(const float x, const int corner, const bool derivative) {
  const float x2{x * x};
  const float x3{x2 * x};

  if (derivative) {
    return corner == 0 ? x3 - 2.0f * x2 + x : x3 - x2;
  }

  return corner == 0 ? 2.0f * x3 - 3.0f * x2 + 1.0f : -2.0f * x3 + 3.0f * x2;
}

__device__ float HermiteWeight(const int coefficient_idx, const float local_x, const float local_y,
                               const float local_z) {
  const int channel = coefficient_idx / 8;
  const int corner = coefficient_idx % 8;
  const int derivative_x[8]{0, 1, 0, 0, 1, 1, 0, 1};
  const int derivative_y[8]{0, 0, 1, 0, 1, 0, 1, 1};
  const int derivative_z[8]{0, 0, 0, 1, 0, 1, 1, 1};

  const int corner_x = corner & 1;
  const int corner_y = (corner >> 1) & 1;
  const int corner_z = (corner >> 2) & 1;
  return CubicHermiteBasis(local_x, corner_x, derivative_x[channel] != 0) *
         CubicHermiteBasis(local_y, corner_y, derivative_y[channel] != 0) *
         CubicHermiteBasis(local_z, corner_z, derivative_z[channel] != 0);
}

__global__ void BuildWeightedJacobianKernel(const float* moving_points_xyz, const float* normal_x,
                                            const float* normal_y, const float* normal_z,
                                            nricp::cuda::CudaWeightedJacobianTriplet* triplets,
                                            int* first_error_point, const int num_points,
                                            const nricp::cuda::CudaGridShape grid_shape,
                                            const int num_grid_vals_per_component) {
  const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) return;

  const int point_offset = point_idx * 3;
  const float x = moving_points_xyz[point_offset + 0];
  const float y = moving_points_xyz[point_offset + 1];
  const float z = moving_points_xyz[point_offset + 2];

  const float normalized_x = (x - grid_shape.origin_x) / grid_shape.voxel_size;
  const float normalized_y = (y - grid_shape.origin_y) / grid_shape.voxel_size;
  const float normalized_z = (z - grid_shape.origin_z) / grid_shape.voxel_size;
  const int x_voxel_idx = static_cast<int>(floorf(normalized_x));
  const int y_voxel_idx = static_cast<int>(floorf(normalized_y));
  const int z_voxel_idx = static_cast<int>(floorf(normalized_z));

  if (x_voxel_idx < 0 || x_voxel_idx >= grid_shape.x_num_voxels || y_voxel_idx < 0 ||
      y_voxel_idx >= grid_shape.y_num_voxels || z_voxel_idx < 0 ||
      z_voxel_idx >= grid_shape.z_num_voxels) {
    atomicCAS(first_error_point, -1, point_idx);
    return;
  }

  const float local_x = normalized_x - static_cast<float>(x_voxel_idx);
  const float local_y = normalized_y - static_cast<float>(y_voxel_idx);
  const float local_z = normalized_z - static_cast<float>(z_voxel_idx);
  const int y_nodes = grid_shape.y_num_voxels + 1;
  const int z_nodes = grid_shape.z_num_voxels + 1;

  for (int component = 0; component < 3; ++component) {
    const float row_weight = component == 0
                                 ? normal_x[point_idx]
                                 : (component == 1 ? normal_y[point_idx] : normal_z[point_idx]);
    for (int coefficient_idx = 0; coefficient_idx < 64; ++coefficient_idx) {
      const int channel = coefficient_idx / 8;
      const int corner = coefficient_idx % 8;
      const int node_x = x_voxel_idx + (corner & 1);
      const int node_y = y_voxel_idx + ((corner >> 1) & 1);
      const int node_z = z_voxel_idx + ((corner >> 2) & 1);
      const int node_index = (node_x * y_nodes + node_y) * z_nodes + node_z;
      const int output_idx = (component * num_points + point_idx) * 64 + coefficient_idx;

      triplets[output_idx].row = point_idx;
      triplets[output_idx].col = component * num_grid_vals_per_component + node_index * 8 + channel;
      triplets[output_idx].value =
          HermiteWeight(coefficient_idx, local_x, local_y, local_z) * row_weight;
    }
  }
}

}  // namespace

namespace nricp::cuda {

CudaWeightedJacobian BuildWeightedJacobian(const std::vector<float>& moving_points_xyz,
                                           const std::vector<float>& normal_x,
                                           const std::vector<float>& normal_y,
                                           const std::vector<float>& normal_z,
                                           const CudaGridShape& grid_shape,
                                           const int num_grid_vals_per_component) {
  if (moving_points_xyz.size() % 3 != 0) {
    throw std::invalid_argument("moving_points_xyz must contain x/y/z triples");
  }
  const auto num_points = static_cast<int>(moving_points_xyz.size() / 3);
  if (normal_x.size() != static_cast<size_t>(num_points) ||
      normal_y.size() != static_cast<size_t>(num_points) ||
      normal_z.size() != static_cast<size_t>(num_points)) {
    throw std::invalid_argument("normal vectors must have one value per moving point");
  }
  if (grid_shape.x_num_voxels <= 0 || grid_shape.y_num_voxels <= 0 ||
      grid_shape.z_num_voxels <= 0 || grid_shape.voxel_size <= 0.0f ||
      num_grid_vals_per_component <= 0) {
    throw std::invalid_argument("invalid grid shape for CUDA weighted Jacobian assembly");
  }

  CudaWeightedJacobian result{};
  result.num_rows = num_points;
  result.num_cols = num_grid_vals_per_component * 3;
  result.triplets.resize(static_cast<size_t>(num_points) * 64 * 3);
  if (num_points == 0) return result;

  float* device_points{};
  float* device_normal_x{};
  float* device_normal_y{};
  float* device_normal_z{};
  CudaWeightedJacobianTriplet* device_triplets{};
  int* device_first_error_point{};

  const size_t points_bytes = moving_points_xyz.size() * sizeof(float);
  const size_t normals_bytes = normal_x.size() * sizeof(float);
  const size_t triplets_bytes = result.triplets.size() * sizeof(CudaWeightedJacobianTriplet);
  const int no_error = -1;

  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_points), points_bytes),
            "cudaMalloc(points)");
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_normal_x), normals_bytes),
            "cudaMalloc(normal_x)");
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_normal_y), normals_bytes),
            "cudaMalloc(normal_y)");
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_normal_z), normals_bytes),
            "cudaMalloc(normal_z)");
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_triplets), triplets_bytes),
            "cudaMalloc(triplets)");
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_first_error_point), sizeof(int)),
            "cudaMalloc(first_error_point)");

  try {
    CheckCuda(
        cudaMemcpy(device_points, moving_points_xyz.data(), points_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(points)");
    CheckCuda(cudaMemcpy(device_normal_x, normal_x.data(), normals_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(normal_x)");
    CheckCuda(cudaMemcpy(device_normal_y, normal_y.data(), normals_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(normal_y)");
    CheckCuda(cudaMemcpy(device_normal_z, normal_z.data(), normals_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(normal_z)");
    CheckCuda(cudaMemcpy(device_first_error_point, &no_error, sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy(first_error_point)");

    constexpr int kThreadsPerBlock = 256;
    const int num_blocks = (num_points + kThreadsPerBlock - 1) / kThreadsPerBlock;
    BuildWeightedJacobianKernel<<<num_blocks, kThreadsPerBlock>>>(
        device_points, device_normal_x, device_normal_y, device_normal_z, device_triplets,
        device_first_error_point, num_points, grid_shape, num_grid_vals_per_component);
    CheckCuda(cudaGetLastError(), "BuildWeightedJacobianKernel");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    int first_error_point = no_error;
    CheckCuda(cudaMemcpy(&first_error_point, device_first_error_point, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(first_error_point)");
    if (first_error_point != no_error) {
      throw std::out_of_range("moving point " + std::to_string(first_error_point) +
                              " is outside the CUDA transformation domain");
    }

    CheckCuda(
        cudaMemcpy(result.triplets.data(), device_triplets, triplets_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(triplets)");
  } catch (...) {
    cudaFree(device_first_error_point);
    cudaFree(device_triplets);
    cudaFree(device_normal_z);
    cudaFree(device_normal_y);
    cudaFree(device_normal_x);
    cudaFree(device_points);
    throw;
  }

  CheckCuda(cudaFree(device_first_error_point), "cudaFree(first_error_point)");
  CheckCuda(cudaFree(device_triplets), "cudaFree(triplets)");
  CheckCuda(cudaFree(device_normal_z), "cudaFree(normal_z)");
  CheckCuda(cudaFree(device_normal_y), "cudaFree(normal_y)");
  CheckCuda(cudaFree(device_normal_x), "cudaFree(normal_x)");
  CheckCuda(cudaFree(device_points), "cudaFree(points)");

  return result;
}

}  // namespace nricp::cuda
