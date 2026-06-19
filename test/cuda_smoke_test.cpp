#include <gtest/gtest.h>

#include "src/lib/cuda/cuda_runtime.hpp"

TEST(CudaRuntime, UsesCuda13AndFindsDevice) {
  const auto info = nricp::cuda::QueryCudaRuntime();

  EXPECT_EQ(info.runtime_version / 1000, 13);
  EXPECT_GE(info.device_count, 1);
  EXPECT_FALSE(info.first_device_name.empty());
  EXPECT_GE(info.first_device_compute_major, 1);
  EXPECT_FALSE(nricp::cuda::CudaVersionString(info.runtime_version).empty());
}
