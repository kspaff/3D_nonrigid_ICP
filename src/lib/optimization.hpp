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
  static OptimizationResults SolveGpu(Correspondences& correspondences,
                                      const std::vector<Scalar>& weights_zero_observations);
};

VectorX BuildZeroObservationWeights(const int num_unknowns,
                                    const int num_grid_vals_per_component,
                                    const std::vector<Scalar>& weights);
