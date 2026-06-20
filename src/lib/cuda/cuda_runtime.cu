#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "src/lib/cuda/cuda_runtime.hpp"

namespace {

void CheckCuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

}  // namespace

namespace nricp::cuda {

std::string CudaVersionString(const int version) {
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

CudaRuntimeInfo QueryCudaRuntime() {
  CudaRuntimeInfo info{};
  CheckCuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
  CheckCuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
  CheckCuda(cudaGetDeviceCount(&info.device_count), "cudaGetDeviceCount");

  if (info.device_count > 0) {
    cudaDeviceProp device_prop{};
    CheckCuda(cudaGetDeviceProperties(&device_prop, 0), "cudaGetDeviceProperties");
    info.first_device_name = device_prop.name;
    info.first_device_compute_major = device_prop.major;
    info.first_device_compute_minor = device_prop.minor;
  }

  return info;
}

}  // namespace nricp::cuda
