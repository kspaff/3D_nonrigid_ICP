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

 private:
  static std::vector<Triplet> SparseIdentity(const int& n);
  static std::vector<Triplet> MultiplyWithComponentsOfNormalVectors(
      const std::vector<Triplet>& triplets_in, const VectorX& n_component);
  static void AddSubblockTriplets(const int& first_row, const int& first_col,
                                  const std::vector<Triplet>& subblock_triplets,
                                  std::vector<Triplet>& triplets);
};
