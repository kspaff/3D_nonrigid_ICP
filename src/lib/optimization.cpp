#include "optimization.hpp"

#include <stdexcept>

namespace {

Scalar WeightForDerivativeChannel(const int channel, const std::vector<Scalar>& weights) {
  if (weights.size() != 4) {
    throw std::invalid_argument("Expected four zero-observation weights: f, first, second, third");
  }

  if (channel == 0) return weights[0];
  if (channel >= 1 && channel <= 3) return weights[1];
  if (channel >= 4 && channel <= 6) return weights[2];
  return weights[3];
}

}  // namespace

VectorX BuildZeroObservationWeights(const int num_unknowns,
                                    const int num_grid_vals_per_component,
                                    const std::vector<Scalar>& weights) {
  VectorX direct_obs_weights(num_unknowns);
  for (int unknown_idx = 0; unknown_idx < num_unknowns; ++unknown_idx) {
    const int component_local_idx = unknown_idx % num_grid_vals_per_component;
    const int derivative_channel = component_local_idx % 8;
    direct_obs_weights(unknown_idx) = WeightForDerivativeChannel(derivative_channel, weights);
  }
  return direct_obs_weights;
}

Optimization::Optimization() = default;

OptimizationResults Optimization::Solve(Correspondences& correspondences,
                                        const std::vector<Scalar>& weights_zero_observations) {
  OptimizationResults optimization_results{};  // returned

  CorrespondencesPointsWithAttributes X{correspondences.GetCorrespondences()};

  int num_unknowns{correspondences.pc_mov().x_translation_grid().num_grid_vals() +
                   correspondences.pc_mov().y_translation_grid().num_grid_vals() +
                   correspondences.pc_mov().z_translation_grid().num_grid_vals()};

  std::vector<Triplet> J_triplets;
  J_triplets.reserve(static_cast<size_t>(X.num) * 64 * 3);
  correspondences.pc_mov().x_translation_grid().AppendJTriplets(X.pc_mov_X, X.pc_fix_nx,
                                                                J_triplets);
  correspondences.pc_mov().y_translation_grid().AppendJTriplets(X.pc_mov_X, X.pc_fix_ny,
                                                                J_triplets);
  correspondences.pc_mov().z_translation_grid().AppendJTriplets(X.pc_mov_X, X.pc_fix_nz,
                                                                J_triplets);

  SparseMatrix J(X.num, num_unknowns);
  J.setFromTriplets(J_triplets.begin(), J_triplets.end());

  const int num_grid_vals_per_component{correspondences.pc_mov().x_translation_grid().num_grid_vals()};
  const VectorX direct_obs_weights{
      BuildZeroObservationWeights(num_unknowns, num_grid_vals_per_component, weights_zero_observations)};
  SparseMatrix normal_matrix{J.transpose() * J};
  for (int unknown_idx = 0; unknown_idx < num_unknowns; ++unknown_idx) {
    normal_matrix.coeffRef(unknown_idx, unknown_idx) += direct_obs_weights(unknown_idx);
  }
  normal_matrix.makeCompressed();

  const VectorX rhs{J.transpose() * (-correspondences.point_to_plane_dists().dists)};
  VectorX initial_guess{VectorX::Zero(num_unknowns)};
  correspondences.pc_mov().x_translation_grid().CopyAllGridValsToVector(initial_guess);
  correspondences.pc_mov().y_translation_grid().CopyAllGridValsToVector(initial_guess);
  correspondences.pc_mov().z_translation_grid().CopyAllGridValsToVector(initial_guess);

  // Solve!
  VectorX xhat(num_unknowns);
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> solver;
  solver.compute(normal_matrix);
  if (solver.info() != Eigen::Success) {
    optimization_results.success = false;
    return optimization_results;
  }
  xhat = solver.solveWithGuess(rhs, initial_guess);
  if (solver.info() != Eigen::Success) {
    optimization_results.success = false;
    return optimization_results;
  }
  optimization_results.success = true;

  // auto v{J * xhat - l};

  // Save estimated unknowns to translation grids
  correspondences.pc_mov().x_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().y_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().z_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().UpdateXt();
  correspondences.ComputeDists();

  optimization_results.num_observations = X.num + num_unknowns;
  optimization_results.num_unknowns = num_unknowns;

  return optimization_results;
}
