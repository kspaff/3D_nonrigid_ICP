#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "src/lib/cuda/cuda_transform.hpp"

namespace {

void CheckCuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

__device__ float CubicHermiteBasis(const float x, const int corner, const bool derivative) {
  if (!derivative) {
    if (corner == 0) return 1.0f - 3.0f * x * x + 2.0f * x * x * x;
    return 3.0f * x * x - 2.0f * x * x * x;
  }

  if (corner == 0) return x - 2.0f * x * x + x * x * x;
  return -x * x + x * x * x;
}

__device__ float EvaluateComponent(const float* coefficients,
                                   const nricp::cuda::CudaGridShape grid_shape,
                                   const int x_voxel_idx, const int y_voxel_idx,
                                   const int z_voxel_idx, const float local_x, const float local_y,
                                   const float local_z) {
  constexpr int kDerivativeX[8]{0, 1, 0, 0, 1, 1, 0, 1};
  constexpr int kDerivativeY[8]{0, 0, 1, 0, 1, 0, 1, 1};
  constexpr int kDerivativeZ[8]{0, 0, 0, 1, 0, 1, 1, 1};

  const int y_nodes = grid_shape.y_num_voxels + 1;
  const int z_nodes = grid_shape.z_num_voxels + 1;

  float value = 0.0f;
  for (int channel = 0; channel < 8; ++channel) {
    for (int corner = 0; corner < 8; ++corner) {
      const int corner_x = corner & 1;
      const int corner_y = (corner >> 1) & 1;
      const int corner_z = (corner >> 2) & 1;
      const int node_x = x_voxel_idx + corner_x;
      const int node_y = y_voxel_idx + corner_y;
      const int node_z = z_voxel_idx + corner_z;
      const int node_index = (node_x * y_nodes + node_y) * z_nodes + node_z;
      const float weight = CubicHermiteBasis(local_x, corner_x, kDerivativeX[channel] != 0) *
                           CubicHermiteBasis(local_y, corner_y, kDerivativeY[channel] != 0) *
                           CubicHermiteBasis(local_z, corner_z, kDerivativeZ[channel] != 0);
      value += weight * coefficients[node_index * 8 + channel];
    }
  }

  return value;
}

__global__ void ApplyTranslationGridsKernel(const float* points_xyz, float* transformed_xyz,
                                            const int num_points, const float* x_coefficients,
                                            const float* y_coefficients,
                                            const float* z_coefficients,
                                            const nricp::cuda::CudaGridShape grid_shape) {
  const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) return;

  const int offset = point_idx * 3;
  const float x = points_xyz[offset + 0];
  const float y = points_xyz[offset + 1];
  const float z = points_xyz[offset + 2];

  const float normalized_x = (x - grid_shape.origin_x) / grid_shape.voxel_size;
  const float normalized_y = (y - grid_shape.origin_y) / grid_shape.voxel_size;
  const float normalized_z = (z - grid_shape.origin_z) / grid_shape.voxel_size;
  const int x_voxel_idx = static_cast<int>(floorf(normalized_x));
  const int y_voxel_idx = static_cast<int>(floorf(normalized_y));
  const int z_voxel_idx = static_cast<int>(floorf(normalized_z));
  const float local_x = normalized_x - x_voxel_idx;
  const float local_y = normalized_y - y_voxel_idx;
  const float local_z = normalized_z - z_voxel_idx;

  transformed_xyz[offset + 0] =
      x + EvaluateComponent(x_coefficients, grid_shape, x_voxel_idx, y_voxel_idx, z_voxel_idx,
                            local_x, local_y, local_z);
  transformed_xyz[offset + 1] =
      y + EvaluateComponent(y_coefficients, grid_shape, x_voxel_idx, y_voxel_idx, z_voxel_idx,
                            local_x, local_y, local_z);
  transformed_xyz[offset + 2] =
      z + EvaluateComponent(z_coefficients, grid_shape, x_voxel_idx, y_voxel_idx, z_voxel_idx,
                            local_x, local_y, local_z);
}

__global__ void ApplyPackedTranslationGridsKernel(const float* points_xyz, float* transformed_xyz,
                                                  const int num_points,
                                                  const float* coefficients_xyz,
                                                  const int num_grid_vals_per_component,
                                                  const nricp::cuda::CudaGridShape grid_shape) {
  const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) return;

  const int offset = point_idx * 3;
  const float x = points_xyz[offset + 0];
  const float y = points_xyz[offset + 1];
  const float z = points_xyz[offset + 2];

  const float normalized_x = (x - grid_shape.origin_x) / grid_shape.voxel_size;
  const float normalized_y = (y - grid_shape.origin_y) / grid_shape.voxel_size;
  const float normalized_z = (z - grid_shape.origin_z) / grid_shape.voxel_size;
  const int x_voxel_idx = static_cast<int>(floorf(normalized_x));
  const int y_voxel_idx = static_cast<int>(floorf(normalized_y));
  const int z_voxel_idx = static_cast<int>(floorf(normalized_z));
  const float local_x = normalized_x - x_voxel_idx;
  const float local_y = normalized_y - y_voxel_idx;
  const float local_z = normalized_z - z_voxel_idx;

  const float* x_coefficients = coefficients_xyz;
  const float* y_coefficients = coefficients_xyz + num_grid_vals_per_component;
  const float* z_coefficients = coefficients_xyz + 2 * num_grid_vals_per_component;
  transformed_xyz[offset + 0] =
      x + EvaluateComponent(x_coefficients, grid_shape, x_voxel_idx, y_voxel_idx, z_voxel_idx,
                            local_x, local_y, local_z);
  transformed_xyz[offset + 1] =
      y + EvaluateComponent(y_coefficients, grid_shape, x_voxel_idx, y_voxel_idx, z_voxel_idx,
                            local_x, local_y, local_z);
  transformed_xyz[offset + 2] =
      z + EvaluateComponent(z_coefficients, grid_shape, x_voxel_idx, y_voxel_idx, z_voxel_idx,
                            local_x, local_y, local_z);
}

}  // namespace

namespace nricp::cuda {

struct CudaTransformWorkspace::Impl {
  Impl(const std::vector<float>& points_xyz_in, const int num_grid_vals_per_component_in)
      : num_points{static_cast<int>(points_xyz_in.size() / 3)},
        num_grid_vals_per_component{num_grid_vals_per_component_in},
        points_bytes{points_xyz_in.size() * sizeof(float)},
        coefficients_bytes{static_cast<size_t>(num_grid_vals_per_component_in) * 3 * sizeof(float)},
        transformed_xyz(points_xyz_in.size()) {
    if (points_xyz_in.size() % 3 != 0) {
      throw std::invalid_argument("points_xyz must contain x/y/z triples");
    }
    if (num_grid_vals_per_component <= 0) {
      throw std::invalid_argument("num_grid_vals_per_component must be positive");
    }
    if (num_points == 0) {
      throw std::invalid_argument("CudaTransformWorkspace requires at least one point");
    }

    try {
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_points), points_bytes),
                "cudaMalloc(transform_points)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_transformed), points_bytes),
                "cudaMalloc(transform_transformed)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_coefficients), coefficients_bytes),
                "cudaMalloc(transform_coefficients)");
      CheckCuda(
          cudaMemcpy(device_points, points_xyz_in.data(), points_bytes, cudaMemcpyHostToDevice),
          "cudaMemcpy(transform_points)");
    } catch (...) {
      Release();
      throw;
    }
  }

  ~Impl() { Release(); }

  void Release() {
    cudaFree(device_coefficients);
    cudaFree(device_transformed);
    cudaFree(device_points);
    device_coefficients = nullptr;
    device_transformed = nullptr;
    device_points = nullptr;
  }

  int num_points{};
  int num_grid_vals_per_component{};
  size_t points_bytes{};
  size_t coefficients_bytes{};
  std::vector<float> transformed_xyz;
  float* device_points{};
  float* device_transformed{};
  float* device_coefficients{};
};

CudaTransformWorkspace::CudaTransformWorkspace(const std::vector<float>& points_xyz,
                                               const int num_grid_vals_per_component)
    : impl_{std::make_unique<Impl>(points_xyz, num_grid_vals_per_component)} {}

CudaTransformWorkspace::~CudaTransformWorkspace() = default;

int CudaTransformWorkspace::num_points() const { return impl_->num_points; }

int CudaTransformWorkspace::num_grid_vals_per_component() const {
  return impl_->num_grid_vals_per_component;
}

const std::vector<float>& CudaTransformWorkspace::Apply(const std::vector<float>& coefficients_xyz,
                                                        const CudaGridShape& grid_shape) {
  Impl& w = *impl_;
  const size_t expected_coefficients = static_cast<size_t>(w.num_grid_vals_per_component) * 3;
  if (coefficients_xyz.size() != expected_coefficients) {
    throw std::invalid_argument("coefficient vector size does not match transform workspace");
  }
  const size_t shape_coefficients = static_cast<size_t>(grid_shape.x_num_voxels + 1) *
                                    static_cast<size_t>(grid_shape.y_num_voxels + 1) *
                                    static_cast<size_t>(grid_shape.z_num_voxels + 1) * 8;
  if (shape_coefficients != static_cast<size_t>(w.num_grid_vals_per_component)) {
    throw std::invalid_argument("grid shape does not match transform workspace");
  }

  CheckCuda(cudaMemcpy(w.device_coefficients, coefficients_xyz.data(), w.coefficients_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(transform_coefficients)");
  constexpr int kThreadsPerBlock = 256;
  const int num_blocks = (w.num_points + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ApplyPackedTranslationGridsKernel<<<num_blocks, kThreadsPerBlock>>>(
      w.device_points, w.device_transformed, w.num_points, w.device_coefficients,
      w.num_grid_vals_per_component, grid_shape);
  CheckCuda(cudaGetLastError(), "ApplyPackedTranslationGridsKernel");
  CheckCuda(cudaMemcpy(w.transformed_xyz.data(), w.device_transformed, w.points_bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(transform_transformed)");
  return w.transformed_xyz;
}

std::vector<float> ApplyTranslationGrids(const std::vector<float>& points_xyz,
                                         const std::vector<float>& x_coefficients,
                                         const std::vector<float>& y_coefficients,
                                         const std::vector<float>& z_coefficients,
                                         const CudaGridShape& grid_shape) {
  if (points_xyz.size() % 3 != 0) {
    throw std::invalid_argument("points_xyz must contain x/y/z triples");
  }
  if (x_coefficients.size() != y_coefficients.size() ||
      x_coefficients.size() != z_coefficients.size()) {
    throw std::invalid_argument("x/y/z coefficient vectors must have equal sizes");
  }
  const size_t expected_coefficients = static_cast<size_t>(grid_shape.x_num_voxels + 1) *
                                       static_cast<size_t>(grid_shape.y_num_voxels + 1) *
                                       static_cast<size_t>(grid_shape.z_num_voxels + 1) * 8;
  if (x_coefficients.size() != expected_coefficients) {
    throw std::invalid_argument("coefficient vector size does not match grid shape");
  }

  const int num_points = static_cast<int>(points_xyz.size() / 3);
  std::vector<float> transformed_xyz(points_xyz.size());
  if (num_points == 0) return transformed_xyz;

  float* device_points{};
  float* device_transformed{};
  float* device_x_coefficients{};
  float* device_y_coefficients{};
  float* device_z_coefficients{};
  const size_t points_bytes = points_xyz.size() * sizeof(float);
  const size_t coefficients_bytes = x_coefficients.size() * sizeof(float);

  CheckCuda(cudaMalloc(&device_points, points_bytes), "cudaMalloc(points)");
  CheckCuda(cudaMalloc(&device_transformed, points_bytes), "cudaMalloc(transformed)");
  CheckCuda(cudaMalloc(&device_x_coefficients, coefficients_bytes), "cudaMalloc(x_coefficients)");
  CheckCuda(cudaMalloc(&device_y_coefficients, coefficients_bytes), "cudaMalloc(y_coefficients)");
  CheckCuda(cudaMalloc(&device_z_coefficients, coefficients_bytes), "cudaMalloc(z_coefficients)");

  try {
    CheckCuda(cudaMemcpy(device_points, points_xyz.data(), points_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(points)");
    CheckCuda(cudaMemcpy(device_x_coefficients, x_coefficients.data(), coefficients_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(x_coefficients)");
    CheckCuda(cudaMemcpy(device_y_coefficients, y_coefficients.data(), coefficients_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(y_coefficients)");
    CheckCuda(cudaMemcpy(device_z_coefficients, z_coefficients.data(), coefficients_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(z_coefficients)");

    constexpr int kThreadsPerBlock = 256;
    const int num_blocks = (num_points + kThreadsPerBlock - 1) / kThreadsPerBlock;
    ApplyTranslationGridsKernel<<<num_blocks, kThreadsPerBlock>>>(
        device_points, device_transformed, num_points, device_x_coefficients, device_y_coefficients,
        device_z_coefficients, grid_shape);
    CheckCuda(cudaGetLastError(), "ApplyTranslationGridsKernel");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    CheckCuda(cudaMemcpy(transformed_xyz.data(), device_transformed, points_bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(transformed)");
  } catch (...) {
    cudaFree(device_z_coefficients);
    cudaFree(device_y_coefficients);
    cudaFree(device_x_coefficients);
    cudaFree(device_transformed);
    cudaFree(device_points);
    throw;
  }

  CheckCuda(cudaFree(device_z_coefficients), "cudaFree(z_coefficients)");
  CheckCuda(cudaFree(device_y_coefficients), "cudaFree(y_coefficients)");
  CheckCuda(cudaFree(device_x_coefficients), "cudaFree(x_coefficients)");
  CheckCuda(cudaFree(device_transformed), "cudaFree(transformed)");
  CheckCuda(cudaFree(device_points), "cudaFree(points)");

  return transformed_xyz;
}

}  // namespace nricp::cuda
