#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tuple>
#include <vector>

#include "scalar_types.hpp"

typedef Eigen::Matrix<int, 64, 1> Vector64i;
typedef Eigen::Matrix<int, 8, 1> Vector8i;

struct GridVals {
  Scalar f{0};
  Scalar fx{0};
  Scalar fy{0};
  Scalar fz{0};
  Scalar fxy{0};
  Scalar fxz{0};
  Scalar fyz{0};
  Scalar fxyz{0};
};

struct GridIdxAdj {
  int f{0};
  int fx{0};
  int fy{0};
  int fz{0};
  int fxy{0};
  int fxz{0};
  int fyz{0};
  int fxyz{0};
};

class TranslationGrid {
 public:
  void Initialize(const RowVector3& grid_origin, const int& x_num_voxels,
                  const int& y_num_voxels, const int& z_num_voxels, const Scalar& voxel_size,
                  const int& first_idx_adj);
  VectorX p(const MatrixX3& X);
  // This version of p() can be used to save computation time if >1 translation grid is used, e.g.
  // for x, y, z
  VectorX p(const MatrixX3& X, const MatrixX64& X_weights,
            const Eigen::MatrixX3i& X_voxel_idx);
  std::vector<Triplet> J(const MatrixX3& X);
  void CopyAllGridValsToVector(VectorX& grid_vals_vector) const;
  void UpdateAllGridValsFromVector(const VectorX& grid_vals_new);
  void UpdateVoxelGridVals(const int& x_voxel_idx, const int& y_voxel_idx, const int& z_voxel_idx,
                           const GridVals grid_vals_new);
  static MatrixX64 ComputeHermiteWeights(const MatrixX3& Xn_voxel);
  static MatrixX64 Compute_X_power(const MatrixX3& Xn_voxel);
  std::tuple<Eigen::MatrixX3i, MatrixX3> GetGridReference(const MatrixX3& X);

  // Getters
  const RowVector3& grid_origin() const;
  const Scalar& voxel_size() const;
  const int& x_num_voxels() const;
  const int& y_num_voxels() const;
  const int& z_num_voxels() const;
  const int& num_grid_vals() const;
  const int& min_idx_adj() const;
  const int& max_idx_adj() const;
  const std::vector<std::vector<std::vector<GridVals>>>& grid_vals() const;

 private:
  std::tuple<Vector64, Vector64i> Get_f(const Eigen::RowVector3i& X_voxel_idx);

  RowVector3 grid_origin_;
  Scalar voxel_size_;
  std::vector<std::vector<std::vector<GridVals>>> grid_vals_;
  std::vector<std::vector<std::vector<GridIdxAdj>>> grid_idx_adj_;
  Matrix64 inv_A_;
  int x_num_voxels_;
  int y_num_voxels_;
  int z_num_voxels_;
  int num_grid_vals_;
  int min_idx_adj_;
  int max_idx_adj_;
};
