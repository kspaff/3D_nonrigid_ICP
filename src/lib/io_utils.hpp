#pragma once

#include <string>

#include "named_column_matrix.hpp"
#include "scalar_types.hpp"

// Import a CSV file into a NamedColumnMatrix.
// The file is expected to have columns in the following order:
// x, y, z (if with_normals = false, with_correspondence_id = false)
// x, y, z, nx, ny, nz (if with_normals = true)
// x, y, z, nx, ny, nz, correspondence_id (if both are true)
// (Headers are supported and will be automatically skipped if non-numeric).
NamedColumnMatrix<MatrixX> ImportFileToMatrix(const std::string& path,
                                              bool with_normals,
                                              bool with_correspondence_id);

// Save point cloud to a CSV file.
// Updates only the x, y, and z coordinates from the updated matrix.
// All other original attributes (normals, correspondence IDs, etc.) remain unchanged.
void SaveMatrixToFile(const NamedColumnMatrix<MatrixX>& x_updated, 
                      const std::string& path_in,
                      const std::string& path_out);
