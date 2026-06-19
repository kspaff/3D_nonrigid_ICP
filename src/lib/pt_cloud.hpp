#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include "translation_grid.hpp"

class PtCloud {
 public:
  PtCloud(MatrixX X);

  void SetNormals(VectorX nx, VectorX ny, VectorX nz);
  void SetCorrespondenceId(VectorX correspondence_id);
  void InitializeTranslationGrids(const Scalar& voxel_size, const uint32_t& buffer_voxels,
                                  const std::vector<Scalar>& grid_limits);
  void ImportTranslationGrids(const std::string& filepath);
  void ExportTranslationGrids(const std::string& filepath);
  void UpdateXt();
  void InitMatricesForUpdateXt();

  long NumPts();
  Scalar x_min();
  Scalar x_max();
  Scalar y_min();
  Scalar y_max();
  Scalar z_min();
  Scalar z_max();

  // Getters
  const MatrixX& X();
  const MatrixX& Xt();
  const VectorX& nx();
  const VectorX& ny();
  const VectorX& nz();
  const VectorX& correspondence_id();
  TranslationGrid& x_translation_grid();
  TranslationGrid& y_translation_grid();
  TranslationGrid& z_translation_grid();

 private:
  MatrixX X_;
  MatrixX Xt_;

  // Point attributes
  VectorX nx_;
  VectorX ny_;
  VectorX nz_;

  // Correspondence id
  VectorX correspondence_id_;

  // Translation grids
  TranslationGrid x_translation_grid_;
  TranslationGrid y_translation_grid_;
  TranslationGrid z_translation_grid_;
  Eigen::MatrixX3i X_voxel_idx_;
  MatrixX3 Xn_voxel_;
  MatrixX64 X_power_;
};

struct HeaderInfo {
  char identifier[10]{"nricp"};
  int fileversion{1};
  const int length{1000};  // bytes
};
