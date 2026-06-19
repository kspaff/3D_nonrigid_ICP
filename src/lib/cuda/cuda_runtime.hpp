#pragma once

#include <string>

namespace nricp::cuda {

struct CudaRuntimeInfo {
  int runtime_version{};
  int driver_version{};
  int device_count{};
  std::string first_device_name{};
  int first_device_compute_major{};
  int first_device_compute_minor{};
};

CudaRuntimeInfo QueryCudaRuntime();
std::string CudaVersionString(int version);

}  // namespace nricp::cuda
