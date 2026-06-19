#pragma once

#include <Eigen/Sparse>

#include "correspondences.hpp"

struct OptimizationResults {
  bool success{};
  int num_observations{};
  int num_unknowns{};
};

class Optimization {
 public:
  Optimization();
  static OptimizationResults Solve(Correspondences& correspondences,
                                   const std::vector<Scalar>& weights_zero_observations);
};
