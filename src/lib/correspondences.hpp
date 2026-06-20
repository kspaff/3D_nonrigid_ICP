#pragma once

#include <algorithm>
#include <fstream>
#include <random>

#include "pt_cloud.hpp"

struct Dists {
  VectorX dists{};
  double mean{NAN};
  double median{NAN};
  double std{NAN};
  double std_mad{NAN};
};

struct CorrespondencesPointsWithAttributes {
  int num{};
  MatrixX3 pc_fix_X{};
  VectorX pc_fix_nx{};
  VectorX pc_fix_ny{};
  VectorX pc_fix_nz{};
  MatrixX3 pc_mov_X{};
  MatrixX3 pc_mov_Xt{};
};

class Correspondences {
 public:
  Correspondences(PtCloud& pc_fix, PtCloud& pc_mov);
  void SelectPointsByRandomSampling(const uint32_t& num_correspondences);
  void MatchPointsByNearestNeighbor();
  void MatchPointsByCorrespondenceId();
  void RejectMaxEuclideanDistanceCriteria(const Scalar& max_euclidean_distance);
  void RejectStdMadCriteria();
  CorrespondencesPointsWithAttributes GetCorrespondences();
  void ComputeDists();
  void ComputeTransformedPointToPlaneReportStats();
  void SetSelectedPoints(std::vector<int> idx_pc_fix);
  void ExportCorrespondences(const std::string& debug_file_name);

  uint64_t num();
  // Getters
  PtCloud& pc_fix();
  PtCloud& pc_mov();
  const Dists& point_to_plane_dists();
  const Dists& point_to_plane_dists_t();
  const Dists& euclidean_dists();
  const Dists& euclidean_dists_t();
  std::vector<int> GetSelectedPoints();

 private:
  MatrixX GetSelectedPoints_();
  MatrixX GetSelectedCorrespondenceIds_();

  PtCloud& pc_fix_;
  PtCloud& pc_mov_;
  std::vector<int> idx_pc_fix_;
  std::vector<int> idx_pc_mov_;
  Dists point_to_plane_dists_;
  Dists point_to_plane_dists_t_;
  Dists euclidean_dists_;
  Dists euclidean_dists_t_;
};

Eigen::MatrixXi KnnSearch(const MatrixX& X, const MatrixX& X_query, const int& k = 1);

template <typename T>
std::vector<T> KeepSubsetOfVector(const std::vector<T>& old_vector, const std::vector<bool>& keep);

std::vector<int> RandInt(const int& min_val, const int& max_val, const uint32_t& n);

double Median(const VectorX& v);

// Median of absolute differences (mad) with respect to the median
double MAD(const VectorX& v);

double Std(const VectorX& v);

template <typename T>
std::vector<T> Range(T start, T stop, T step = 1);
