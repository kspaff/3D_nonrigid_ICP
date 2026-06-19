#include "optimization.hpp"

#include <stdexcept>

#ifdef NRICP_ENABLE_CUDA
#include "src/lib/cuda/cuda_jacobian.hpp"
#endif

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

OptimizationResults SolveWeightedJacobian(Correspondences& correspondences,
                                          const SparseMatrix& J,
                                          const int num_correspondences,
                                          const std::vector<Scalar>& weights_zero_observations) {
  OptimizationResults optimization_results{};

  const int num_unknowns{static_cast<int>(J.cols())};
  const int num_grid_vals_per_component{
      correspondences.pc_mov().x_translation_grid().num_grid_vals()};
  const VectorX direct_obs_weights{
      BuildZeroObservationWeights(num_unknowns, num_grid_vals_per_component,
                                  weights_zero_observations)};
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

  correspondences.pc_mov().x_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().y_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().z_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().UpdateXt();
  correspondences.ComputeDists();

  optimization_results.num_observations = num_correspondences + num_unknowns;
  optimization_results.num_unknowns = num_unknowns;

  return optimization_results;
}

#ifdef NRICP_ENABLE_CUDA
std::vector<float> FlattenMatrixX3(const MatrixX3& matrix) {
  std::vector<float> values;
  values.reserve(static_cast<size_t>(matrix.rows()) * 3);
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    values.push_back(matrix(row, 0));
    values.push_back(matrix(row, 1));
    values.push_back(matrix(row, 2));
  }
  return values;
}

std::vector<float> CopyVector(const VectorX& vector) {
  std::vector<float> values;
  values.reserve(static_cast<size_t>(vector.size()));
  for (Eigen::Index row = 0; row < vector.size(); ++row) {
    values.push_back(vector(row));
  }
  return values;
}

nricp::cuda::CudaGridShape CudaShapeFromGrid(const TranslationGrid& grid) {
  return nricp::cuda::CudaGridShape{
      grid.grid_origin()(0),
      grid.grid_origin()(1),
      grid.grid_origin()(2),
      grid.x_num_voxels(),
      grid.y_num_voxels(),
      grid.z_num_voxels(),
      grid.voxel_size()};
}
#endif

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

  return SolveWeightedJacobian(correspondences, J, X.num, weights_zero_observations);
}

OptimizationResults Optimization::SolveGpu(
    Correspondences& correspondences,
    const std::vector<Scalar>& weights_zero_observations) {
#ifndef NRICP_ENABLE_CUDA
  throw std::runtime_error(
      "GPU execution backend is not available in this build; rerun with --execution_backend cpu.");
#else
  CorrespondencesPointsWithAttributes X{correspondences.GetCorrespondences()};
  const int num_grid_vals_per_component{
      correspondences.pc_mov().x_translation_grid().num_grid_vals()};

  const auto cuda_jacobian = nricp::cuda::BuildWeightedJacobian(
      FlattenMatrixX3(X.pc_mov_X), CopyVector(X.pc_fix_nx), CopyVector(X.pc_fix_ny),
      CopyVector(X.pc_fix_nz), CudaShapeFromGrid(correspondences.pc_mov().x_translation_grid()),
      num_grid_vals_per_component);

  std::vector<Triplet> J_triplets;
  J_triplets.reserve(cuda_jacobian.triplets.size());
  for (const auto& triplet : cuda_jacobian.triplets) {
    J_triplets.emplace_back(triplet.row, triplet.col, static_cast<Scalar>(triplet.value));
  }

  SparseMatrix J(cuda_jacobian.num_rows, cuda_jacobian.num_cols);
  J.setFromTriplets(J_triplets.begin(), J_triplets.end());
  return SolveWeightedJacobian(correspondences, J, X.num, weights_zero_observations);
#endif
}
