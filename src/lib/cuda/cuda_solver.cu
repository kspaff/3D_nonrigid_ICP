#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "src/lib/cuda/cuda_solver.hpp"

namespace {

void CheckCuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

void CheckCublas(const cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
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

__device__ bool ComputeGridReference(const float* moving_points_xyz, const int point_idx,
                                     const nricp::cuda::CudaGridShape grid_shape, int& x_voxel_idx,
                                     int& y_voxel_idx, int& z_voxel_idx, float& local_x,
                                     float& local_y, float& local_z) {
  const int point_offset = point_idx * 3;
  const float x = moving_points_xyz[point_offset + 0];
  const float y = moving_points_xyz[point_offset + 1];
  const float z = moving_points_xyz[point_offset + 2];

  const float normalized_x = (x - grid_shape.origin_x) / grid_shape.voxel_size;
  const float normalized_y = (y - grid_shape.origin_y) / grid_shape.voxel_size;
  const float normalized_z = (z - grid_shape.origin_z) / grid_shape.voxel_size;
  x_voxel_idx = static_cast<int>(floorf(normalized_x));
  y_voxel_idx = static_cast<int>(floorf(normalized_y));
  z_voxel_idx = static_cast<int>(floorf(normalized_z));

  if (x_voxel_idx < 0 || x_voxel_idx >= grid_shape.x_num_voxels || y_voxel_idx < 0 ||
      y_voxel_idx >= grid_shape.y_num_voxels || z_voxel_idx < 0 ||
      z_voxel_idx >= grid_shape.z_num_voxels) {
    return false;
  }

  local_x = normalized_x - static_cast<float>(x_voxel_idx);
  local_y = normalized_y - static_cast<float>(y_voxel_idx);
  local_z = normalized_z - static_cast<float>(z_voxel_idx);
  return true;
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

__device__ int CoefficientColumn(const int coefficient_idx, const int component,
                                 const int x_voxel_idx, const int y_voxel_idx,
                                 const int z_voxel_idx, const nricp::cuda::CudaGridShape grid_shape,
                                 const int num_grid_vals_per_component) {
  const int channel = coefficient_idx / 8;
  const int corner = coefficient_idx % 8;
  const int node_x = x_voxel_idx + (corner & 1);
  const int node_y = y_voxel_idx + ((corner >> 1) & 1);
  const int node_z = z_voxel_idx + ((corner >> 2) & 1);
  const int y_nodes = grid_shape.y_num_voxels + 1;
  const int z_nodes = grid_shape.z_num_voxels + 1;
  const int node_index = (node_x * y_nodes + node_y) * z_nodes + node_z;
  return component * num_grid_vals_per_component + node_index * 8 + channel;
}

__device__ float ComponentWeight(const int component, const int point_idx, const float* normal_x,
                                 const float* normal_y, const float* normal_z) {
  return component == 0 ? normal_x[point_idx]
                        : (component == 1 ? normal_y[point_idx] : normal_z[point_idx]);
}

__global__ void PrecomputeWeightedJacobianMetadataKernel(
    const float* moving_points_xyz, const float* normal_x, const float* normal_y,
    const float* normal_z, float* weighted_values, int* columns, int* first_error_point,
    const int num_points, const nricp::cuda::CudaGridShape grid_shape,
    const int num_grid_vals_per_component) {
  const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) return;

  int x_voxel_idx{};
  int y_voxel_idx{};
  int z_voxel_idx{};
  float local_x{};
  float local_y{};
  float local_z{};
  if (!ComputeGridReference(moving_points_xyz, point_idx, grid_shape, x_voxel_idx, y_voxel_idx,
                            z_voxel_idx, local_x, local_y, local_z)) {
    atomicCAS(first_error_point, -1, point_idx);
    return;
  }

  for (int component = 0; component < 3; ++component) {
    const float component_weight =
        ComponentWeight(component, point_idx, normal_x, normal_y, normal_z);
    for (int coefficient_idx = 0; coefficient_idx < 64; ++coefficient_idx) {
      const int metadata_idx = (point_idx * 3 + component) * 64 + coefficient_idx;
      weighted_values[metadata_idx] =
          HermiteWeight(coefficient_idx, local_x, local_y, local_z) * component_weight;
      columns[metadata_idx] =
          CoefficientColumn(coefficient_idx, component, x_voxel_idx, y_voxel_idx, z_voxel_idx,
                            grid_shape, num_grid_vals_per_component);
    }
  }
}

__global__ void BuildRhsAndPreconditionerFromMetadataKernel(const float* weighted_values,
                                                            const int* columns,
                                                            const float* point_to_plane_dists,
                                                            float* rhs, float* preconditioner_diag,
                                                            const int num_points) {
  const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) return;

  const float residual = -point_to_plane_dists[point_idx];
  for (int component = 0; component < 3; ++component) {
    for (int coefficient_idx = 0; coefficient_idx < 64; ++coefficient_idx) {
      const int metadata_idx = (point_idx * 3 + component) * 64 + coefficient_idx;
      const float value = weighted_values[metadata_idx];
      const int col = columns[metadata_idx];
      atomicAdd(&rhs[col], value * residual);
      atomicAdd(&preconditioner_diag[col], value * value);
    }
  }
}

__global__ void ApplyRegularizationKernel(float* output, const float* input,
                                          const float* regularization_diag,
                                          const int num_unknowns) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_unknowns) return;
  output[idx] = regularization_diag[idx] * input[idx];
}

__global__ void ApplyNormalMatrixFromMetadataKernel(const float* weighted_values,
                                                    const int* columns, const float* input,
                                                    float* output, const int num_points) {
  const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) return;

  float projected_value = 0.0f;
  for (int component = 0; component < 3; ++component) {
    for (int coefficient_idx = 0; coefficient_idx < 64; ++coefficient_idx) {
      const int metadata_idx = (point_idx * 3 + component) * 64 + coefficient_idx;
      projected_value += weighted_values[metadata_idx] * input[columns[metadata_idx]];
    }
  }

  for (int component = 0; component < 3; ++component) {
    for (int coefficient_idx = 0; coefficient_idx < 64; ++coefficient_idx) {
      const int metadata_idx = (point_idx * 3 + component) * 64 + coefficient_idx;
      atomicAdd(&output[columns[metadata_idx]], weighted_values[metadata_idx] * projected_value);
    }
  }
}

__global__ void ApplyInversePreconditionerKernel(float* z, const float* r,
                                                 const float* preconditioner_diag,
                                                 const int num_unknowns) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_unknowns) return;
  const float diag = preconditioner_diag[idx];
  z[idx] = diag > 0.0f ? r[idx] / diag : r[idx];
}

__global__ void UpdateConjugateDirectionKernel(float* p, const float* z, const float beta,
                                               const int num_unknowns) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_unknowns) return;
  p[idx] = z[idx] + beta * p[idx];
}

void ApplyNormalEquationsFromMetadata(const float* device_weighted_values,
                                      const int* device_columns,
                                      const float* device_regularization_diag,
                                      const float* device_input, float* device_output,
                                      const int num_points, const int num_unknowns) {
  constexpr int kThreadsPerBlock = 256;
  const int unknown_blocks = (num_unknowns + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ApplyRegularizationKernel<<<unknown_blocks, kThreadsPerBlock>>>(
      device_output, device_input, device_regularization_diag, num_unknowns);
  CheckCuda(cudaGetLastError(), "ApplyRegularizationKernel");

  const int point_blocks = (num_points + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ApplyNormalMatrixFromMetadataKernel<<<point_blocks, kThreadsPerBlock>>>(
      device_weighted_values, device_columns, device_input, device_output, num_points);
  CheckCuda(cudaGetLastError(), "ApplyNormalMatrixFromMetadataKernel");
}

}  // namespace

namespace nricp::cuda {

struct CudaPcgWorkspace::Impl {
  Impl(const int max_num_points_in, const int num_grid_vals_per_component_in)
      : max_num_points{max_num_points_in},
        num_grid_vals_per_component{num_grid_vals_per_component_in},
        num_unknowns{num_grid_vals_per_component_in * 3},
        points_bytes{static_cast<size_t>(max_num_points_in) * 3 * sizeof(float)},
        point_scalars_bytes{static_cast<size_t>(max_num_points_in) * sizeof(float)},
        unknowns_bytes{static_cast<size_t>(num_grid_vals_per_component_in) * 3 * sizeof(float)},
        metadata_entries{static_cast<size_t>(max_num_points_in) * 3 * 64},
        metadata_values_bytes{metadata_entries * sizeof(float)},
        metadata_columns_bytes{metadata_entries * sizeof(int)} {
    if (max_num_points <= 0 || num_grid_vals_per_component <= 0) {
      throw std::invalid_argument("invalid CUDA PCG workspace dimensions");
    }

    try {
      CheckCublas(cublasCreate(&cublas_handle), "cublasCreate");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_points), points_bytes),
                "cudaMalloc(points)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_normal_x), point_scalars_bytes),
                "cudaMalloc(normal_x)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_normal_y), point_scalars_bytes),
                "cudaMalloc(normal_y)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_normal_z), point_scalars_bytes),
                "cudaMalloc(normal_z)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_dists), point_scalars_bytes),
                "cudaMalloc(point_to_plane_dists)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_regularization_diag), unknowns_bytes),
                "cudaMalloc(regularization_diag)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_preconditioner_diag), unknowns_bytes),
                "cudaMalloc(preconditioner_diag)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_rhs), unknowns_bytes),
                "cudaMalloc(rhs)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_x), unknowns_bytes), "cudaMalloc(x)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_r), unknowns_bytes), "cudaMalloc(r)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_z), unknowns_bytes), "cudaMalloc(z)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_p), unknowns_bytes), "cudaMalloc(p)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_ap), unknowns_bytes), "cudaMalloc(Ap)");
      CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&device_weighted_values), metadata_values_bytes),
          "cudaMalloc(weighted_values)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_columns), metadata_columns_bytes),
                "cudaMalloc(columns)");
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_first_error_point), sizeof(int)),
                "cudaMalloc(first_error_point)");
    } catch (...) {
      Free();
      throw;
    }
  }

  ~Impl() { Free(); }

  void Free() {
    cudaFree(device_first_error_point);
    cudaFree(device_columns);
    cudaFree(device_weighted_values);
    cudaFree(device_ap);
    cudaFree(device_p);
    cudaFree(device_z);
    cudaFree(device_r);
    cudaFree(device_x);
    cudaFree(device_rhs);
    cudaFree(device_preconditioner_diag);
    cudaFree(device_regularization_diag);
    cudaFree(device_dists);
    cudaFree(device_normal_z);
    cudaFree(device_normal_y);
    cudaFree(device_normal_x);
    cudaFree(device_points);
    device_first_error_point = nullptr;
    device_columns = nullptr;
    device_weighted_values = nullptr;
    device_ap = nullptr;
    device_p = nullptr;
    device_z = nullptr;
    device_r = nullptr;
    device_x = nullptr;
    device_rhs = nullptr;
    device_preconditioner_diag = nullptr;
    device_regularization_diag = nullptr;
    device_dists = nullptr;
    device_normal_z = nullptr;
    device_normal_y = nullptr;
    device_normal_x = nullptr;
    device_points = nullptr;
    if (cublas_handle != nullptr) {
      cublasDestroy(cublas_handle);
      cublas_handle = nullptr;
    }
  }

  int max_num_points{};
  int num_grid_vals_per_component{};
  int num_unknowns{};
  size_t points_bytes{};
  size_t point_scalars_bytes{};
  size_t unknowns_bytes{};
  size_t metadata_entries{};
  size_t metadata_values_bytes{};
  size_t metadata_columns_bytes{};
  float* device_points{};
  float* device_normal_x{};
  float* device_normal_y{};
  float* device_normal_z{};
  float* device_dists{};
  float* device_regularization_diag{};
  float* device_preconditioner_diag{};
  float* device_rhs{};
  float* device_x{};
  float* device_r{};
  float* device_z{};
  float* device_p{};
  float* device_ap{};
  float* device_weighted_values{};
  int* device_columns{};
  int* device_first_error_point{};
  cublasHandle_t cublas_handle{};
};

CudaPcgWorkspace::CudaPcgWorkspace(const int max_num_points, const int num_grid_vals_per_component)
    : impl_{std::make_unique<Impl>(max_num_points, num_grid_vals_per_component)} {}

CudaPcgWorkspace::~CudaPcgWorkspace() = default;

int CudaPcgWorkspace::max_num_points() const { return impl_->max_num_points; }

int CudaPcgWorkspace::num_grid_vals_per_component() const {
  return impl_->num_grid_vals_per_component;
}

int CudaPcgWorkspace::num_unknowns() const { return impl_->num_unknowns; }

CudaCgSolveResult SolveNormalEquationsPcg(
    CudaPcgWorkspace& workspace, const std::vector<float>& moving_points_xyz,
    const std::vector<float>& normal_x, const std::vector<float>& normal_y,
    const std::vector<float>& normal_z, const std::vector<float>& point_to_plane_dists,
    const std::vector<float>& regularization_diag, const std::vector<float>& initial_guess,
    const CudaGridShape& grid_shape, const int max_iterations, const float tolerance) {
  if (moving_points_xyz.size() % 3 != 0) {
    throw std::invalid_argument("moving_points_xyz must contain x/y/z triples");
  }
  const int num_points = static_cast<int>(moving_points_xyz.size() / 3);
  auto& w = *workspace.impl_;
  const int num_unknowns = w.num_unknowns;
  if (normal_x.size() != static_cast<size_t>(num_points) ||
      normal_y.size() != static_cast<size_t>(num_points) ||
      normal_z.size() != static_cast<size_t>(num_points) ||
      point_to_plane_dists.size() != static_cast<size_t>(num_points)) {
    throw std::invalid_argument("normal and distance vectors must have one value per point");
  }
  if (regularization_diag.size() != static_cast<size_t>(num_unknowns) ||
      initial_guess.size() != static_cast<size_t>(num_unknowns)) {
    throw std::invalid_argument("solver vectors do not match the CUDA normal system size");
  }
  if (num_points > w.max_num_points) {
    throw std::invalid_argument("CUDA PCG workspace is too small for the correspondence count");
  }
  if (num_points == 0 || num_unknowns <= 0 || max_iterations <= 0 || tolerance <= 0.0f) {
    throw std::invalid_argument("invalid CUDA PCG solve dimensions or tolerance");
  }

  CudaCgSolveResult result{};
  result.solution.resize(static_cast<size_t>(num_unknowns));

  const size_t points_bytes = moving_points_xyz.size() * sizeof(float);
  const size_t point_scalars_bytes = normal_x.size() * sizeof(float);
  const int no_error = -1;

  CheckCuda(
      cudaMemcpy(w.device_points, moving_points_xyz.data(), points_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(points)");
  CheckCuda(
      cudaMemcpy(w.device_normal_x, normal_x.data(), point_scalars_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(normal_x)");
  CheckCuda(
      cudaMemcpy(w.device_normal_y, normal_y.data(), point_scalars_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(normal_y)");
  CheckCuda(
      cudaMemcpy(w.device_normal_z, normal_z.data(), point_scalars_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(normal_z)");
  CheckCuda(cudaMemcpy(w.device_dists, point_to_plane_dists.data(), point_scalars_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(point_to_plane_dists)");
  CheckCuda(cudaMemcpy(w.device_regularization_diag, regularization_diag.data(), w.unknowns_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(regularization_diag)");
  CheckCuda(cudaMemcpy(w.device_preconditioner_diag, regularization_diag.data(), w.unknowns_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(preconditioner_diag)");
  CheckCuda(cudaMemset(w.device_rhs, 0, w.unknowns_bytes), "cudaMemset(rhs)");
  CheckCuda(cudaMemcpy(w.device_x, initial_guess.data(), w.unknowns_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(x)");
  CheckCuda(cudaMemcpy(w.device_first_error_point, &no_error, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy(first_error_point)");

  constexpr int kThreadsPerBlock = 256;
  const int point_blocks = (num_points + kThreadsPerBlock - 1) / kThreadsPerBlock;
  PrecomputeWeightedJacobianMetadataKernel<<<point_blocks, kThreadsPerBlock>>>(
      w.device_points, w.device_normal_x, w.device_normal_y, w.device_normal_z,
      w.device_weighted_values, w.device_columns, w.device_first_error_point, num_points,
      grid_shape, w.num_grid_vals_per_component);
  CheckCuda(cudaGetLastError(), "PrecomputeWeightedJacobianMetadataKernel");
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  int first_error_point = no_error;
  CheckCuda(cudaMemcpy(&first_error_point, w.device_first_error_point, sizeof(int),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(first_error_point)");
  if (first_error_point != no_error) {
    throw std::out_of_range("moving point " + std::to_string(first_error_point) +
                            " is outside the CUDA transformation domain");
  }

  BuildRhsAndPreconditionerFromMetadataKernel<<<point_blocks, kThreadsPerBlock>>>(
      w.device_weighted_values, w.device_columns, w.device_dists, w.device_rhs,
      w.device_preconditioner_diag, num_points);
  CheckCuda(cudaGetLastError(), "BuildRhsAndPreconditionerFromMetadataKernel");

  ApplyNormalEquationsFromMetadata(w.device_weighted_values, w.device_columns,
                                   w.device_regularization_diag, w.device_x, w.device_ap,
                                   num_points, num_unknowns);
  CheckCuda(cudaMemcpy(w.device_r, w.device_rhs, w.unknowns_bytes, cudaMemcpyDeviceToDevice),
            "cudaMemcpy(r=rhs)");
  float minus_one = -1.0f;
  CheckCublas(cublasSaxpy(w.cublas_handle, num_unknowns, &minus_one, w.device_ap, 1, w.device_r, 1),
              "cublasSaxpy(r-=Ap)");

  float rhs_norm = 0.0f;
  CheckCublas(cublasSnrm2(w.cublas_handle, num_unknowns, w.device_rhs, 1, &rhs_norm),
              "cublasSnrm2(rhs)");
  const float normalization = rhs_norm > 0.0f ? rhs_norm : 1.0f;

  float residual_norm = 0.0f;
  CheckCublas(cublasSnrm2(w.cublas_handle, num_unknowns, w.device_r, 1, &residual_norm),
              "cublasSnrm2(r)");
  result.relative_error = residual_norm / normalization;
  if (result.relative_error <= tolerance) {
    result.success = true;
    CheckCuda(
        cudaMemcpy(result.solution.data(), w.device_x, w.unknowns_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(solution)");
    return result;
  }

  const int unknown_blocks = (num_unknowns + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ApplyInversePreconditionerKernel<<<unknown_blocks, kThreadsPerBlock>>>(
      w.device_z, w.device_r, w.device_preconditioner_diag, num_unknowns);
  CheckCuda(cudaGetLastError(), "ApplyInversePreconditionerKernel");
  CheckCuda(cudaMemcpy(w.device_p, w.device_z, w.unknowns_bytes, cudaMemcpyDeviceToDevice),
            "cudaMemcpy(p=z)");

  float rz_old = 0.0f;
  CheckCublas(cublasSdot(w.cublas_handle, num_unknowns, w.device_r, 1, w.device_z, 1, &rz_old),
              "cublasSdot(r,z)");

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    ApplyNormalEquationsFromMetadata(w.device_weighted_values, w.device_columns,
                                     w.device_regularization_diag, w.device_p, w.device_ap,
                                     num_points, num_unknowns);

    float p_ap = 0.0f;
    CheckCublas(cublasSdot(w.cublas_handle, num_unknowns, w.device_p, 1, w.device_ap, 1, &p_ap),
                "cublasSdot(p,Ap)");
    if (p_ap <= 0.0f || rz_old == 0.0f) {
      result.success = false;
      break;
    }

    const float alpha = rz_old / p_ap;
    CheckCublas(cublasSaxpy(w.cublas_handle, num_unknowns, &alpha, w.device_p, 1, w.device_x, 1),
                "cublasSaxpy(x+=alpha*p)");
    const float negative_alpha = -alpha;
    CheckCublas(
        cublasSaxpy(w.cublas_handle, num_unknowns, &negative_alpha, w.device_ap, 1, w.device_r, 1),
        "cublasSaxpy(r-=alpha*Ap)");

    CheckCublas(cublasSnrm2(w.cublas_handle, num_unknowns, w.device_r, 1, &residual_norm),
                "cublasSnrm2(r iter)");
    result.iterations = iteration + 1;
    result.relative_error = residual_norm / normalization;
    if (result.relative_error <= tolerance) {
      result.success = true;
      break;
    }

    ApplyInversePreconditionerKernel<<<unknown_blocks, kThreadsPerBlock>>>(
        w.device_z, w.device_r, w.device_preconditioner_diag, num_unknowns);
    CheckCuda(cudaGetLastError(), "ApplyInversePreconditionerKernel");

    float rz_new = 0.0f;
    CheckCublas(cublasSdot(w.cublas_handle, num_unknowns, w.device_r, 1, w.device_z, 1, &rz_new),
                "cublasSdot(r,z iter)");
    const float beta = rz_new / rz_old;
    UpdateConjugateDirectionKernel<<<unknown_blocks, kThreadsPerBlock>>>(w.device_p, w.device_z,
                                                                         beta, num_unknowns);
    CheckCuda(cudaGetLastError(), "UpdateConjugateDirectionKernel");
    rz_old = rz_new;
  }

  CheckCuda(
      cudaMemcpy(result.solution.data(), w.device_x, w.unknowns_bytes, cudaMemcpyDeviceToHost),
      "cudaMemcpy(solution)");
  return result;
}

CudaCgSolveResult SolveNormalEquationsPcg(
    const std::vector<float>& moving_points_xyz, const std::vector<float>& normal_x,
    const std::vector<float>& normal_y, const std::vector<float>& normal_z,
    const std::vector<float>& point_to_plane_dists, const std::vector<float>& regularization_diag,
    const std::vector<float>& initial_guess, const CudaGridShape& grid_shape,
    const int num_grid_vals_per_component, const int max_iterations, const float tolerance) {
  if (moving_points_xyz.size() % 3 != 0) {
    throw std::invalid_argument("moving_points_xyz must contain x/y/z triples");
  }
  CudaPcgWorkspace workspace{static_cast<int>(moving_points_xyz.size() / 3),
                             num_grid_vals_per_component};
  return SolveNormalEquationsPcg(workspace, moving_points_xyz, normal_x, normal_y, normal_z,
                                 point_to_plane_dists, regularization_diag, initial_guess,
                                 grid_shape, max_iterations, tolerance);
}

}  // namespace nricp::cuda
