#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "src/lib/io_utils.hpp"
#include "src/lib/correspondences.hpp"
#include "src/lib/optimization.hpp"
#include "src/lib/pt_cloud.hpp"
#include "src/lib/scalar_types.hpp"

#ifndef NRICP_ADDITIONAL_TRANSFORMS
#define NRICP_ADDITIONAL_TRANSFORMS ""
#endif

#ifndef NRICP_ADDITIONAL_GOLDENS
#define NRICP_ADDITIONAL_GOLDENS ""
#endif

#ifndef NRICP_SMOKE_TRANSFORM_MOVING_XYZ
#define NRICP_SMOKE_TRANSFORM_MOVING_XYZ ""
#endif

#ifndef NRICP_SMOKE_TRANSFORM_REPEAT
#define NRICP_SMOKE_TRANSFORM_REPEAT ""
#endif

#ifndef NRICP_SMOKE_TRANSFORM_GPU
#define NRICP_SMOKE_TRANSFORM_GPU ""
#endif

#ifndef NRICP_SMOKE_TRANSFORMED
#define NRICP_SMOKE_TRANSFORMED ""
#endif

#ifndef NRICP_SMOKE_TRANSFORMED_SINGLE_CHUNK
#define NRICP_SMOKE_TRANSFORMED_SINGLE_CHUNK ""
#endif

namespace {

template <typename T>
T ReadValue(std::ifstream& file) {
  T value{};
  file.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

struct NricpHeaderForTest {
  char identifier[10]{};
  int fileversion{};
  double origin_x{};
  double origin_y{};
  double origin_z{};
  int x_num_voxels{};
  int y_num_voxels{};
  int z_num_voxels{};
  double voxel_size{};
};

NricpHeaderForTest ReadNricpHeader(std::ifstream& file) {
  NricpHeaderForTest header{};
  file.read(header.identifier, sizeof(header.identifier));
  header.fileversion = ReadValue<int>(file);
  header.origin_x = ReadValue<double>(file);
  header.origin_y = ReadValue<double>(file);
  header.origin_z = ReadValue<double>(file);
  header.x_num_voxels = ReadValue<int>(file);
  header.y_num_voxels = ReadValue<int>(file);
  header.z_num_voxels = ReadValue<int>(file);
  header.voxel_size = ReadValue<double>(file);
  return header;
}

std::uint64_t HashFileFnv1a64(const std::filesystem::path& path) {
  constexpr std::uint64_t kOffsetBasis = 14695981039346656037ull;
  constexpr std::uint64_t kPrime = 1099511628211ull;

  std::ifstream file{path, std::ios::binary};
  if (!file.is_open()) return 0;

  std::uint64_t hash = kOffsetBasis;
  std::array<char, 8192> buffer{};
  while (file) {
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto bytes_read = file.gcount();
    for (std::streamsize i = 0; i < bytes_read; ++i) {
      hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
      hash *= kPrime;
    }
  }
  return hash;
}

std::vector<std::filesystem::path> SplitPathList(const std::string& paths) {
  std::vector<std::filesystem::path> result;
  std::stringstream stream{paths};
  std::string item;
  while (std::getline(stream, item, '|')) {
    if (!item.empty()) {
      result.emplace_back(item);
    }
  }
  return result;
}

testing::AssertionResult NricpFilesNear(const std::filesystem::path& actual_path,
                                        const std::filesystem::path& expected_path,
                                        const double epsilon) {
  constexpr std::uintmax_t kHeaderBytes = 1000;

  if (!std::filesystem::exists(actual_path)) {
    return testing::AssertionFailure() << "actual file does not exist: " << actual_path.string();
  }
  if (!std::filesystem::exists(expected_path)) {
    return testing::AssertionFailure() << "expected file does not exist: "
                                       << expected_path.string();
  }
  if (std::filesystem::file_size(actual_path) != std::filesystem::file_size(expected_path)) {
    return testing::AssertionFailure()
           << "file sizes differ: actual=" << std::filesystem::file_size(actual_path)
           << " expected=" << std::filesystem::file_size(expected_path);
  }

  std::ifstream actual{actual_path, std::ios::binary};
  std::ifstream expected{expected_path, std::ios::binary};
  if (!actual.is_open() || !expected.is_open()) {
    return testing::AssertionFailure() << "failed to open files for comparison";
  }

  const auto actual_header = ReadNricpHeader(actual);
  const auto expected_header = ReadNricpHeader(expected);
  if (std::strcmp(actual_header.identifier, expected_header.identifier) != 0 ||
      actual_header.fileversion != expected_header.fileversion ||
      actual_header.x_num_voxels != expected_header.x_num_voxels ||
      actual_header.y_num_voxels != expected_header.y_num_voxels ||
      actual_header.z_num_voxels != expected_header.z_num_voxels) {
    return testing::AssertionFailure() << "non-floating .nricp header fields differ";
  }

  const auto header_double_near = [epsilon](const double actual_value,
                                            const double expected_value) {
    return std::abs(actual_value - expected_value) <= epsilon;
  };
  if (!header_double_near(actual_header.origin_x, expected_header.origin_x) ||
      !header_double_near(actual_header.origin_y, expected_header.origin_y) ||
      !header_double_near(actual_header.origin_z, expected_header.origin_z) ||
      !header_double_near(actual_header.voxel_size, expected_header.voxel_size)) {
    return testing::AssertionFailure() << "floating .nricp header fields differ by more than "
                                       << epsilon;
  }

  const std::uintmax_t payload_bytes = std::filesystem::file_size(actual_path) - kHeaderBytes;
  if (payload_bytes % sizeof(double) != 0) {
    return testing::AssertionFailure() << "payload size is not a multiple of double bytes";
  }
  const std::uintmax_t payload_values = payload_bytes / sizeof(double);

  actual.seekg(static_cast<std::streamoff>(kHeaderBytes), std::ios::beg);
  expected.seekg(static_cast<std::streamoff>(kHeaderBytes), std::ios::beg);

  double max_abs_diff = 0.0;
  std::uintmax_t max_abs_diff_index = 0;
  double max_actual_value = 0.0;
  double max_expected_value = 0.0;
  std::uintmax_t values_over_epsilon = 0;
  std::uintmax_t first_over_epsilon_index = 0;
  double first_actual_value = 0.0;
  double first_expected_value = 0.0;
  double first_abs_diff = 0.0;

  for (std::uintmax_t value_idx = 0; value_idx < payload_values; ++value_idx) {
    const double actual_value = ReadValue<double>(actual);
    const double expected_value = ReadValue<double>(expected);
    if (!std::isfinite(actual_value) || !std::isfinite(expected_value)) {
      return testing::AssertionFailure() << "nonfinite payload value at index " << value_idx;
    }

    const double abs_diff = std::abs(actual_value - expected_value);
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
      max_abs_diff_index = value_idx;
      max_actual_value = actual_value;
      max_expected_value = expected_value;
    }
    if (abs_diff > epsilon) {
      if (values_over_epsilon == 0) {
        first_over_epsilon_index = value_idx;
        first_actual_value = actual_value;
        first_expected_value = expected_value;
        first_abs_diff = abs_diff;
      }
      ++values_over_epsilon;
    }
  }

  if (values_over_epsilon != 0) {
    return testing::AssertionFailure()
           << values_over_epsilon << " payload values differ by more than epsilon " << epsilon
           << "; first index=" << first_over_epsilon_index << " actual=" << first_actual_value
           << " expected=" << first_expected_value << " abs_diff=" << first_abs_diff
           << "; max index=" << max_abs_diff_index << " actual=" << max_actual_value
           << " expected=" << max_expected_value << " abs_diff=" << max_abs_diff;
  }

  return testing::AssertionSuccess()
         << "payload values are within epsilon " << epsilon << "; max_abs_diff=" << max_abs_diff
         << " at payload index " << max_abs_diff_index;
}

void ExpectLockedFitMatchesGolden(const std::filesystem::path& transform_path) {
  const std::filesystem::path golden_path{NRICP_GOLDEN_TRANSFORM};
  ASSERT_TRUE(std::filesystem::exists(transform_path));
  ASSERT_TRUE(std::filesystem::exists(golden_path));

  constexpr std::uintmax_t kExpectedSize = 85480;
  constexpr std::uint64_t kExpectedFnv1a64 = 0x79DAB4243AB1C0B8ull;
  // Direct Hermite weights and the SPD solve reorder float arithmetic relative to the reference.
  constexpr double kPayloadEpsilon = 1e-4;
  EXPECT_EQ(std::filesystem::file_size(golden_path), kExpectedSize);
  EXPECT_EQ(HashFileFnv1a64(golden_path), kExpectedFnv1a64);
  EXPECT_EQ(std::filesystem::file_size(transform_path), kExpectedSize);

  if (HashFileFnv1a64(transform_path) != kExpectedFnv1a64) {
    const auto comparison = NricpFilesNear(transform_path, golden_path, kPayloadEpsilon);
    EXPECT_TRUE(comparison) << comparison.message();
  }
}

}  // namespace

TEST(NricpDataSmoke, UsesSinglePrecisionInMemory) {
  static_assert(std::is_same_v<Scalar, float>);
  EXPECT_EQ(sizeof(Scalar), sizeof(float));

  const auto fixed = ImportFileToMatrix(NRICP_FIXED_PC, true, false);
  const auto moving = ImportFileToMatrix(NRICP_MOVING_PC, true, false);
  EXPECT_EQ(fixed.rows(), 31080);
  EXPECT_EQ(fixed.cols(), 6);
  EXPECT_EQ(moving.rows(), 27771);
  EXPECT_EQ(moving.cols(), 6);
  EXPECT_STREQ(typeid(MatrixX::Scalar).name(), typeid(float).name());
}

TEST(NricpDataSmoke, GeneratedFileHasExpectedV1Layout) {
  const std::filesystem::path transform_path{NRICP_SMOKE_TRANSFORM};
  ASSERT_TRUE(std::filesystem::exists(transform_path));

  constexpr int kXNumVoxels = 10;
  constexpr int kYNumVoxels = 7;
  constexpr int kZNumVoxels = 4;
  constexpr std::uintmax_t kHeaderBytes = 1000;
  constexpr std::uintmax_t kPayloadBytes =
      (kXNumVoxels + 1) * (kYNumVoxels + 1) * (kZNumVoxels + 1) * 3 * 8 * sizeof(double);
  EXPECT_EQ(std::filesystem::file_size(transform_path), kHeaderBytes + kPayloadBytes);

  std::ifstream file{transform_path, std::ios::binary};
  ASSERT_TRUE(file.is_open());

  char identifier[10]{};
  file.read(identifier, sizeof(identifier));
  EXPECT_STREQ(identifier, "nricp");
  EXPECT_EQ(ReadValue<int>(file), 1);
  EXPECT_DOUBLE_EQ(ReadValue<double>(file), -50.0);
  EXPECT_DOUBLE_EQ(ReadValue<double>(file), -50.0);
  EXPECT_DOUBLE_EQ(ReadValue<double>(file), -19.0);
  EXPECT_EQ(ReadValue<int>(file), kXNumVoxels);
  EXPECT_EQ(ReadValue<int>(file), kYNumVoxels);
  EXPECT_EQ(ReadValue<int>(file), kZNumVoxels);
  EXPECT_DOUBLE_EQ(ReadValue<double>(file), 50.0);

  file.seekg(static_cast<std::streamoff>(kHeaderBytes), std::ios::beg);
  const double first_payload_value = ReadValue<double>(file);
  file.seekg(-static_cast<std::streamoff>(sizeof(double)), std::ios::end);
  const double last_payload_value = ReadValue<double>(file);
  EXPECT_TRUE(std::isfinite(first_payload_value));
  EXPECT_TRUE(std::isfinite(last_payload_value));
}

TEST(NricpDataSmoke, GeneratedFileMatchesGoldenFitOutput) {
  ExpectLockedFitMatchesGolden(NRICP_SMOKE_TRANSFORM);
}

TEST(NricpDataSmoke, MovingCloudWithoutNormalsMatchesGoldenFitOutput) {
  ExpectLockedFitMatchesGolden(NRICP_SMOKE_TRANSFORM_MOVING_XYZ);
}

TEST(NricpDataSmoke, RepeatedFitMatchesPrimaryFitOutput) {
  const std::filesystem::path primary_path{NRICP_SMOKE_TRANSFORM};
  const std::filesystem::path repeat_path{NRICP_SMOKE_TRANSFORM_REPEAT};
  ASSERT_TRUE(std::filesystem::exists(primary_path));
  ASSERT_TRUE(std::filesystem::exists(repeat_path));
  EXPECT_EQ(std::filesystem::file_size(primary_path), std::filesystem::file_size(repeat_path));

  if (HashFileFnv1a64(primary_path) != HashFileFnv1a64(repeat_path)) {
    const auto comparison = NricpFilesNear(repeat_path, primary_path, 1e-4);
    EXPECT_TRUE(comparison) << comparison.message();
  }
}

TEST(NricpDataSmoke, GpuGeneratedFileMatchesGoldenFitOutput) {
  const std::filesystem::path transform_path{NRICP_SMOKE_TRANSFORM_GPU};
  if (transform_path.empty()) {
    GTEST_SKIP() << "CUDA fit output is not configured for this build.";
  }
  ExpectLockedFitMatchesGolden(transform_path);
}

TEST(NricpAdditionalFits, GeneratedFilesMatchGoldens) {
  const auto transform_paths = SplitPathList(NRICP_ADDITIONAL_TRANSFORMS);
  const auto golden_paths = SplitPathList(NRICP_ADDITIONAL_GOLDENS);
  ASSERT_FALSE(transform_paths.empty());
  ASSERT_EQ(transform_paths.size(), golden_paths.size());

  constexpr double kPayloadEpsilon = 1e-4;
  for (size_t i = 0; i < transform_paths.size(); ++i) {
    SCOPED_TRACE("actual=" + transform_paths[i].string() + " golden=" + golden_paths[i].string());
    ASSERT_TRUE(std::filesystem::exists(transform_paths[i]));
    ASSERT_TRUE(std::filesystem::exists(golden_paths[i]));

    if (HashFileFnv1a64(transform_paths[i]) != HashFileFnv1a64(golden_paths[i])) {
      const auto comparison = NricpFilesNear(transform_paths[i], golden_paths[i], kPayloadEpsilon);
      EXPECT_TRUE(comparison) << comparison.message();
    }
  }
}

TEST(NricpDataSmoke, GeneratedFileCanBeImportedAndApplied) {
  const auto moving = ImportFileToMatrix(NRICP_MOVING_PC, false, false);
  MatrixX coords = moving.leftCols(3);
  PtCloud cloud{coords};

  ASSERT_NO_THROW(cloud.ImportTranslationGrids(NRICP_SMOKE_TRANSFORM));
  ASSERT_NO_THROW(cloud.InitMatricesForUpdateXt());
  ASSERT_NO_THROW(cloud.UpdateXt());

  ASSERT_EQ(cloud.Xt().rows(), moving.rows());
  ASSERT_EQ(cloud.Xt().cols(), 3);
  EXPECT_TRUE(cloud.Xt().array().isFinite().all());

  const Scalar max_displacement = (cloud.Xt() - cloud.X()).rowwise().norm().maxCoeff();
  EXPECT_GT(max_displacement, Scalar{0});
  EXPECT_LT(max_displacement, Scalar{10});
}

TEST(NricpTransformOutputs, ChunkedAndSingleChunkOutputsMatchExactly) {
  const std::filesystem::path chunked_path{NRICP_SMOKE_TRANSFORMED};
  const std::filesystem::path single_chunk_path{NRICP_SMOKE_TRANSFORMED_SINGLE_CHUNK};
  ASSERT_TRUE(std::filesystem::exists(chunked_path));
  ASSERT_TRUE(std::filesystem::exists(single_chunk_path));
  EXPECT_EQ(std::filesystem::file_size(chunked_path), std::filesystem::file_size(single_chunk_path));
  EXPECT_EQ(HashFileFnv1a64(chunked_path), HashFileFnv1a64(single_chunk_path));
}

TEST(CorrespondenceStats, MedianAndMadIncludeFirstElement) {
  VectorX median_values(3);
  median_values << Scalar{10}, Scalar{1}, Scalar{2};
  EXPECT_DOUBLE_EQ(Median(median_values), 2.0);

  VectorX mad_values(3);
  mad_values << Scalar{100}, Scalar{101}, Scalar{0};
  EXPECT_DOUBLE_EQ(Median(mad_values), 100.0);
  EXPECT_DOUBLE_EQ(MAD(mad_values), 1.0);
}

TEST(CorrespondenceSampling, SelectsSinglePointWhenRequestExceedsCloudSize) {
  const auto selected = RandInt(0, 0, 35000);

  ASSERT_EQ(selected.size(), 1);
  EXPECT_EQ(selected[0], 0);
}

TEST(CorrespondenceSampling, SelectsAllFixedPointsWhenRequestExceedsCloudSize) {
  MatrixX3 fixed_points(3, 3);
  fixed_points << Scalar{0}, Scalar{0}, Scalar{0},
      Scalar{1}, Scalar{0}, Scalar{0},
      Scalar{2}, Scalar{0}, Scalar{0};
  MatrixX3 moving_points(3, 3);
  moving_points << Scalar{0}, Scalar{0}, Scalar{0},
      Scalar{1}, Scalar{0}, Scalar{0},
      Scalar{2}, Scalar{0}, Scalar{0};

  PtCloud fixed{fixed_points};
  PtCloud moving{moving_points};
  Correspondences correspondences{fixed, moving};
  correspondences.SelectPointsByRandomSampling(35000);

  const auto selected = correspondences.GetSelectedPoints();
  ASSERT_EQ(selected.size(), 3);
  EXPECT_EQ(selected[0], 0);
  EXPECT_EQ(selected[1], 1);
  EXPECT_EQ(selected[2], 2);
}

TEST(CorrespondenceSampling, PartialSelectionIsDeterministicAndSorted) {
  const auto first = RandInt(0, 9, 4);
  const auto second = RandInt(0, 9, 4);

  EXPECT_EQ(first, second);
  ASSERT_EQ(first.size(), 4);
  EXPECT_TRUE(std::is_sorted(first.begin(), first.end()));
  for (const int index : first) {
    EXPECT_GE(index, 0);
    EXPECT_LE(index, 9);
  }
}

TEST(CorrespondenceSampling, RejectsZeroRequestedCorrespondences) {
  EXPECT_THROW(RandInt(0, 2, 0), std::invalid_argument);
}

TEST(OptimizationRegularization, EqualWeightsBuildUniformRidgeDiagonal) {
  const auto regularization =
      BuildZeroObservationWeights(48, 16, {Scalar{0.01f}, Scalar{0.01f}, Scalar{0.01f},
                                           Scalar{0.01f}});

  ASSERT_EQ(regularization.size(), 48);
  for (Eigen::Index i = 0; i < regularization.size(); ++i) {
    EXPECT_EQ(regularization(i), Scalar{0.01f});
  }
}

TEST(OptimizationRegularization, MapsDerivativeClassesToWeightSlots) {
  const auto regularization =
      BuildZeroObservationWeights(24, 8, {Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}});
  const Scalar expected_component_weights[8]{Scalar{1}, Scalar{2}, Scalar{2}, Scalar{2},
                                             Scalar{3}, Scalar{3}, Scalar{3}, Scalar{4}};

  ASSERT_EQ(regularization.size(), 24);
  for (Eigen::Index i = 0; i < regularization.size(); ++i) {
    EXPECT_EQ(regularization(i), expected_component_weights[i % 8]);
  }
}

TEST(OptimizationRegularization, RejectsNonFourWeightVector) {
  EXPECT_THROW(BuildZeroObservationWeights(8, 8, {Scalar{0.01f}}), std::invalid_argument);
}

TEST(CorrespondenceMatching, KnnSearchFindsExactNearestNeighbors) {
  MatrixX points(4, 3);
  points << Scalar{0}, Scalar{0}, Scalar{0},
      Scalar{2}, Scalar{0}, Scalar{0},
      Scalar{0}, Scalar{3}, Scalar{0},
      Scalar{0}, Scalar{0}, Scalar{4};

  MatrixX queries(3, 3);
  queries << Scalar{1.75f}, Scalar{0.1f}, Scalar{0},
      Scalar{0.1f}, Scalar{2.75f}, Scalar{0},
      Scalar{0}, Scalar{0.1f}, Scalar{3.75f};

  const auto idx = KnnSearch(points, queries, 1);
  ASSERT_EQ(idx.rows(), 3);
  ASSERT_EQ(idx.cols(), 1);
  EXPECT_EQ(idx(0, 0), 1);
  EXPECT_EQ(idx(1, 0), 2);
  EXPECT_EQ(idx(2, 0), 3);
}

TEST(CorrespondenceMatching, MaxEuclideanDistanceKeepsValuesAtThreshold) {
  MatrixX3 fixed_points(3, 3);
  fixed_points << Scalar{0}, Scalar{0}, Scalar{0},
      Scalar{0}, Scalar{0}, Scalar{0},
      Scalar{0}, Scalar{0}, Scalar{0};
  MatrixX3 moving_points(3, 3);
  moving_points << Scalar{1.999f}, Scalar{0}, Scalar{0},
      Scalar{2.0f}, Scalar{0}, Scalar{0},
      Scalar{2.001f}, Scalar{0}, Scalar{0};

  PtCloud fixed{fixed_points};
  PtCloud moving{moving_points};
  VectorX normal_x = VectorX::Ones(3);
  VectorX normal_y = VectorX::Zero(3);
  VectorX normal_z = VectorX::Zero(3);
  fixed.SetNormals(normal_x, normal_y, normal_z);
  VectorX ids(3);
  ids << Scalar{1}, Scalar{2}, Scalar{3};
  fixed.SetCorrespondenceId(ids);
  moving.SetCorrespondenceId(ids);
  moving.InitializeTranslationGrids(Scalar{1}, 1, {0, 0, 0, 0, 0, 0});

  Correspondences correspondences{fixed, moving};
  correspondences.SelectPointsByRandomSampling(3);
  correspondences.MatchPointsByCorrespondenceId();
  ASSERT_EQ(correspondences.num(), 3);

  correspondences.RejectMaxEuclideanDistanceCriteria(Scalar{2});
  EXPECT_EQ(correspondences.num(), 2);
  EXPECT_LE(correspondences.euclidean_dists_t().dists.maxCoeff(), Scalar{2});
}

TEST(CorrespondenceMatching, PreservesOriginalMovingCoordinatesAfterTransformUpdate) {
  MatrixX3 fixed_points(1, 3);
  fixed_points << Scalar{1}, Scalar{0}, Scalar{0};
  MatrixX3 moving_points(1, 3);
  moving_points << Scalar{0}, Scalar{0}, Scalar{0};

  PtCloud fixed{fixed_points};
  PtCloud moving{moving_points};
  VectorX normal_x = VectorX::Ones(1);
  VectorX normal_y = VectorX::Zero(1);
  VectorX normal_z = VectorX::Zero(1);
  fixed.SetNormals(normal_x, normal_y, normal_z);
  VectorX ids(1);
  ids << Scalar{1};
  fixed.SetCorrespondenceId(ids);
  moving.SetCorrespondenceId(ids);

  moving.InitializeTranslationGrids(Scalar{1}, 0, {0, 0, 0, 1, 1, 1});
  GridVals x_corner_displacement{};
  x_corner_displacement.f = Scalar{1};
  moving.x_translation_grid().UpdateVoxelGridVals(0, 0, 0, x_corner_displacement);
  moving.InitMatricesForUpdateXt();
  moving.UpdateXt();

  Correspondences correspondences{fixed, moving};
  correspondences.SelectPointsByRandomSampling(1);
  correspondences.MatchPointsByCorrespondenceId();
  const auto matched = correspondences.GetCorrespondences();

  ASSERT_EQ(matched.num, 1);
  EXPECT_EQ(matched.pc_mov_X(0, 0), Scalar{0});
  EXPECT_EQ(matched.pc_mov_Xt(0, 0), Scalar{1});
  EXPECT_EQ(correspondences.point_to_plane_dists().dists(0), Scalar{-1});
  EXPECT_EQ(correspondences.point_to_plane_dists_t().dists(0), Scalar{0});
}

TEST(TranslationGridBasis, HermiteWeightsMatchReferenceMatrixEvaluator) {
  TranslationGrid grid;
  RowVector3 origin{};
  origin << Scalar{0}, Scalar{0}, Scalar{0};
  grid.Initialize(origin, 1, 1, 1, Scalar{1}, 0);

  Vector64 f_vals{};
  for (int corner = 0; corner < 8; ++corner) {
    GridVals vals{};
    vals.f = Scalar{0.125f * (1 + corner)};
    vals.fx = Scalar{0.125f * (9 + corner)};
    vals.fy = Scalar{0.125f * (17 + corner)};
    vals.fz = Scalar{0.125f * (25 + corner)};
    vals.fxy = Scalar{0.125f * (33 + corner)};
    vals.fxz = Scalar{0.125f * (41 + corner)};
    vals.fyz = Scalar{0.125f * (49 + corner)};
    vals.fxyz = Scalar{0.125f * (57 + corner)};

    const int x = corner & 1;
    const int y = (corner >> 1) & 1;
    const int z = (corner >> 2) & 1;
    grid.UpdateVoxelGridVals(x, y, z, vals);

    f_vals(8 * 0 + corner) = vals.f;
    f_vals(8 * 1 + corner) = vals.fx;
    f_vals(8 * 2 + corner) = vals.fy;
    f_vals(8 * 3 + corner) = vals.fz;
    f_vals(8 * 4 + corner) = vals.fxy;
    f_vals(8 * 5 + corner) = vals.fxz;
    f_vals(8 * 6 + corner) = vals.fyz;
    f_vals(8 * 7 + corner) = vals.fxyz;
  }

  MatrixX3 points(6, 3);
  points << Scalar{0}, Scalar{0}, Scalar{0},
      Scalar{0.25f}, Scalar{0.5f}, Scalar{0.75f},
      Scalar{0.5f}, Scalar{0.25f}, Scalar{0.125f},
      Scalar{0.9f}, Scalar{0.1f}, Scalar{0.6f},
      Scalar{0.125f}, Scalar{0.875f}, Scalar{0.375f},
      Scalar{0.999f}, Scalar{0.999f}, Scalar{0.999f};

  const auto reference = grid.p(points);
  const auto weights = TranslationGrid::ComputeHermiteWeights(points);
  for (int i = 0; i < points.rows(); ++i) {
    const Scalar hermite_value = weights.row(i).dot(f_vals);
    EXPECT_NEAR(reference(i), hermite_value, 1e-5f) << "point index " << i;
  }
}

TEST(TranslationGridBasis, CopyAllGridValsToVectorUsesGlobalIndices) {
  RowVector3 origin{};
  origin << Scalar{0}, Scalar{0}, Scalar{0};

  TranslationGrid x_grid;
  TranslationGrid y_grid;
  x_grid.Initialize(origin, 1, 1, 1, Scalar{1}, 0);
  y_grid.Initialize(origin, 1, 1, 1, Scalar{1}, x_grid.num_grid_vals());

  GridVals x_vals{};
  x_vals.f = Scalar{1};
  x_vals.fx = Scalar{2};
  x_vals.fy = Scalar{3};
  x_vals.fz = Scalar{4};
  x_vals.fxy = Scalar{5};
  x_vals.fxz = Scalar{6};
  x_vals.fyz = Scalar{7};
  x_vals.fxyz = Scalar{8};
  x_grid.UpdateVoxelGridVals(0, 0, 0, x_vals);

  GridVals y_vals{};
  y_vals.f = Scalar{9};
  y_vals.fx = Scalar{10};
  y_vals.fy = Scalar{11};
  y_vals.fz = Scalar{12};
  y_vals.fxy = Scalar{13};
  y_vals.fxz = Scalar{14};
  y_vals.fyz = Scalar{15};
  y_vals.fxyz = Scalar{16};
  y_grid.UpdateVoxelGridVals(0, 0, 0, y_vals);

  VectorX coefficients = VectorX::Constant(x_grid.num_grid_vals() + y_grid.num_grid_vals(),
                                           Scalar{-1});
  x_grid.CopyAllGridValsToVector(coefficients);
  y_grid.CopyAllGridValsToVector(coefficients);

  EXPECT_EQ(coefficients(0), Scalar{1});
  EXPECT_EQ(coefficients(1), Scalar{2});
  EXPECT_EQ(coefficients(7), Scalar{8});
  EXPECT_EQ(coefficients(x_grid.num_grid_vals()), Scalar{9});
  EXPECT_EQ(coefficients(x_grid.num_grid_vals() + 1), Scalar{10});
  EXPECT_EQ(coefficients(x_grid.num_grid_vals() + 7), Scalar{16});
}

TEST(TranslationGridBasis, AppendJTripletsMatchesWeightedReferenceJ) {
  TranslationGrid grid;
  RowVector3 origin{};
  origin << Scalar{0}, Scalar{0}, Scalar{0};
  grid.Initialize(origin, 1, 1, 1, Scalar{1}, 17);

  MatrixX3 points(2, 3);
  points << Scalar{0.25f}, Scalar{0.5f}, Scalar{0.75f},
      Scalar{0.9f}, Scalar{0.1f}, Scalar{0.6f};
  VectorX row_weights(2);
  row_weights << Scalar{2}, Scalar{-0.5f};

  const auto reference_triplets = grid.J(points);
  std::vector<Triplet> appended_triplets;
  grid.AppendJTriplets(points, row_weights, appended_triplets);

  ASSERT_EQ(appended_triplets.size(), reference_triplets.size());
  for (size_t i = 0; i < reference_triplets.size(); ++i) {
    EXPECT_EQ(appended_triplets[i].row(), reference_triplets[i].row());
    EXPECT_EQ(appended_triplets[i].col(), reference_triplets[i].col());
    EXPECT_NEAR(appended_triplets[i].value(),
                reference_triplets[i].value() * row_weights(reference_triplets[i].row()),
                1e-6f);
  }
}
