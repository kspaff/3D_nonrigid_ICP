#include "optimization.hpp"

#include <limits>
#include <memory>
#include <stdexcept>

#ifdef NRICP_ENABLE_CUDA
#include "src/lib/cuda/cuda_solver.hpp"
#include "src/lib/cuda/cuda_transform.hpp"
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

OptimizationResults SolveWeightedJacobian(Correspondences& correspondences, const SparseMatrix& J,
                                          const int num_correspondences,
                                          const std::vector<Scalar>& weights_zero_observations) {
  OptimizationResults optimization_results{};

  const int num_unknowns{static_cast<int>(J.cols())};
  const int num_grid_vals_per_component{
      correspondences.pc_mov().x_translation_grid().num_grid_vals()};
  const VectorX direct_obs_weights{BuildZeroObservationWeights(
      num_unknowns, num_grid_vals_per_component, weights_zero_observations)};
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

  correspondences.pc_mov().x_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().y_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().z_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().UpdateXt();
  correspondences.ComputeDists();

  optimization_results.success = true;
  optimization_results.num_observations = num_correspondences + num_unknowns;
  optimization_results.num_unknowns = num_unknowns;

  return optimization_results;
}

#ifdef NRICP_ENABLE_CUDA
template <typename MatrixT>
std::vector<float> FlattenMatrixX3(const MatrixT& matrix) {
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
      grid.grid_origin()(0), grid.grid_origin()(1), grid.grid_origin()(2), grid.x_num_voxels(),
      grid.y_num_voxels(),   grid.z_num_voxels(),   grid.voxel_size()};
}
#endif

}  // namespace

VectorX BuildZeroObservationWeights(const int num_unknowns, const int num_grid_vals_per_component,
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

#ifndef NRICP_ENABLE_CUDA
OptimizationResults Optimization::SolveGpu(Correspondences& correspondences,
                                           const std::vector<Scalar>& weights_zero_observations) {
  (void)correspondences;
  (void)weights_zero_observations;
  throw std::runtime_error(
      "GPU execution backend is not available in this build; rerun with --execution_backend cpu.");
}
#else
OptimizationResults Optimization::SolveGpu(
    Correspondences& correspondences, const std::vector<Scalar>& weights_zero_observations,
    nricp::cuda::CudaPcgWorkspace* workspace,
    nricp::cuda::CudaTransformWorkspace* transform_workspace) {
  CorrespondencesPointsWithAttributes X{correspondences.GetCorrespondences()};
  const int num_grid_vals_per_component{
      correspondences.pc_mov().x_translation_grid().num_grid_vals()};
  const int num_unknowns{num_grid_vals_per_component * 3};
  VectorX initial_guess{VectorX::Zero(num_unknowns)};
  correspondences.pc_mov().x_translation_grid().CopyAllGridValsToVector(initial_guess);
  correspondences.pc_mov().y_translation_grid().CopyAllGridValsToVector(initial_guess);
  correspondences.pc_mov().z_translation_grid().CopyAllGridValsToVector(initial_guess);
  const VectorX regularization_diag{BuildZeroObservationWeights(
      num_unknowns, num_grid_vals_per_component, weights_zero_observations)};

  std::unique_ptr<nricp::cuda::CudaPcgWorkspace> local_workspace;
  if (workspace == nullptr) {
    local_workspace =
        std::make_unique<nricp::cuda::CudaPcgWorkspace>(X.num, num_grid_vals_per_component);
    workspace = local_workspace.get();
  }
  std::unique_ptr<nricp::cuda::CudaTransformWorkspace> local_transform_workspace;
  if (transform_workspace == nullptr) {
    local_transform_workspace = std::make_unique<nricp::cuda::CudaTransformWorkspace>(
        FlattenMatrixX3(correspondences.pc_mov().X()), num_grid_vals_per_component);
    transform_workspace = local_transform_workspace.get();
  }

  const auto moving_points_xyz = FlattenMatrixX3(X.pc_mov_X);
  const auto normal_x = CopyVector(X.pc_fix_nx);
  const auto normal_y = CopyVector(X.pc_fix_ny);
  const auto normal_z = CopyVector(X.pc_fix_nz);
  const auto point_to_plane_dists = CopyVector(correspondences.point_to_plane_dists().dists);
  const auto regularization = CopyVector(regularization_diag);
  const auto initial = CopyVector(initial_guess);

  const auto solve_result = nricp::cuda::SolveNormalEquationsPcg(
      *workspace, moving_points_xyz, normal_x, normal_y, normal_z, point_to_plane_dists,
      regularization, initial, CudaShapeFromGrid(correspondences.pc_mov().x_translation_grid()),
      num_unknowns, std::numeric_limits<Scalar>::epsilon());

  OptimizationResults optimization_results{};
  optimization_results.num_observations = X.num + num_unknowns;
  optimization_results.num_unknowns = num_unknowns;
  if (!solve_result.success) {
    optimization_results.success = false;
    return optimization_results;
  }

  VectorX xhat(num_unknowns);
  for (int unknown_idx = 0; unknown_idx < num_unknowns; ++unknown_idx) {
    xhat(unknown_idx) = solve_result.solution[static_cast<size_t>(unknown_idx)];
  }

  correspondences.pc_mov().x_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().y_translation_grid().UpdateAllGridValsFromVector(xhat);
  correspondences.pc_mov().z_translation_grid().UpdateAllGridValsFromVector(xhat);
  const auto& transformed_xyz = transform_workspace->Apply(
      solve_result.solution, CudaShapeFromGrid(correspondences.pc_mov().x_translation_grid()));
  correspondences.pc_mov().SetXtFromFlatXYZ(transformed_xyz);
  correspondences.ComputeTransformedPointToPlaneReportStats();

  optimization_results.success = true;
  return optimization_results;
}
#endif
