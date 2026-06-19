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

VectorX BuildDirectObservationWeights(const int num_unknowns, const int num_grid_vals_per_component,
                                      const std::vector<Scalar>& weights) {
  VectorX direct_obs_weights(num_unknowns);
  for (int unknown_idx = 0; unknown_idx < num_unknowns; ++unknown_idx) {
    const int component_local_idx = unknown_idx % num_grid_vals_per_component;
    const int derivative_channel = component_local_idx % 8;
    direct_obs_weights(unknown_idx) = WeightForDerivativeChannel(derivative_channel, weights);
  }
  return direct_obs_weights;
}

}  // namespace

Optimization::Optimization() = default;

OptimizationResults Optimization::Solve(Correspondences& correspondences,
                                        const std::vector<Scalar>& weights_zero_observations) {
  OptimizationResults optimization_results{};  // returned

  CorrespondencesPointsWithAttributes X{correspondences.GetCorrespondences()};

  auto J_pc_mov_x_triplets{correspondences.pc_mov().x_translation_grid().J(X.pc_mov_X)};
  auto J_pc_mov_y_triplets{correspondences.pc_mov().y_translation_grid().J(X.pc_mov_X)};
  auto J_pc_mov_z_triplets{correspondences.pc_mov().z_translation_grid().J(X.pc_mov_X)};

  auto J_pc_mov_x_nx_triplets{
      Optimization::MultiplyWithComponentsOfNormalVectors(J_pc_mov_x_triplets, X.pc_fix_nx)};
  auto J_pc_mov_y_ny_triplets{
      Optimization::MultiplyWithComponentsOfNormalVectors(J_pc_mov_y_triplets, X.pc_fix_ny)};
  auto J_pc_mov_z_nz_triplets{
      Optimization::MultiplyWithComponentsOfNormalVectors(J_pc_mov_z_triplets, X.pc_fix_nz)};

  J_pc_mov_x_triplets.clear();
  J_pc_mov_y_triplets.clear();
  J_pc_mov_z_triplets.clear();

  int num_unknowns{correspondences.pc_mov().x_translation_grid().num_grid_vals() +
                   correspondences.pc_mov().y_translation_grid().num_grid_vals() +
                   correspondences.pc_mov().z_translation_grid().num_grid_vals()};

  std::vector<Triplet> J_triplets;
  J_triplets.reserve(J_pc_mov_x_nx_triplets.size() + J_pc_mov_y_ny_triplets.size() +
                     J_pc_mov_z_nz_triplets.size());

  // clang-format off
  Optimization::AddSubblockTriplets(0,
                      0,
                      J_pc_mov_x_nx_triplets,
                      J_triplets);
  Optimization::AddSubblockTriplets(0,
                      0,
                      J_pc_mov_y_ny_triplets,
                      J_triplets);
  Optimization::AddSubblockTriplets(0,
                      0,
                      J_pc_mov_z_nz_triplets,
                      J_triplets);
  // clang-format on

  SparseMatrix J(X.num, num_unknowns);
  J.setFromTriplets(J_triplets.begin(), J_triplets.end());

  const int num_grid_vals_per_component{correspondences.pc_mov().x_translation_grid().num_grid_vals()};
  const VectorX direct_obs_weights{
      BuildDirectObservationWeights(num_unknowns, num_grid_vals_per_component, weights_zero_observations)};
  SparseMatrix normal_matrix{J.transpose() * J};
  for (int unknown_idx = 0; unknown_idx < num_unknowns; ++unknown_idx) {
    normal_matrix.coeffRef(unknown_idx, unknown_idx) += direct_obs_weights(unknown_idx);
  }
  normal_matrix.makeCompressed();

  const VectorX rhs{J.transpose() * (-correspondences.point_to_plane_dists().dists)};

  // Solve!
  VectorX xhat(num_unknowns);
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> solver;
  solver.compute(normal_matrix);
  if (solver.info() != Eigen::Success) {
    optimization_results.success = false;
    return optimization_results;
  }
  xhat = solver.solve(rhs);
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

std::vector<Triplet> Optimization::MultiplyWithComponentsOfNormalVectors(
    const std::vector<Triplet>& triplets_in, const VectorX& n_component) {
  std::vector<Triplet> triplets_out;
  triplets_out.reserve(triplets_in.size());
  for (auto const& triplet : triplets_in) {
    int row{triplet.row()};
    int col{triplet.col()};
    Scalar val{triplet.value() * n_component(triplet.row())};
    triplets_out.emplace_back(row, col, val);
  }

  return triplets_out;
}

void Optimization::AddSubblockTriplets(const int& first_row, const int& first_col,
                                       const std::vector<Triplet>& subblock_triplets,
                                       std::vector<Triplet>& triplets) {
  for (auto const& triplet : subblock_triplets) {
    int row{first_row + triplet.row()};
    int col{first_col + triplet.col()};
    Scalar val{triplet.value()};
    triplets.emplace_back(row, col, val);
  }
}
