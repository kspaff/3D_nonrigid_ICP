#include "io_utils.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iomanip>

NamedColumnMatrix<MatrixX> ImportFileToMatrix(const std::string& path,
                                              bool with_normals,
                                              bool with_correspondence_id) 
													  {
  int expected_cols = 3;
  if (with_normals) expected_cols += 3;
  if (with_correspondence_id) expected_cols += 1;

  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  std::vector<std::vector<Scalar>> data;
  std::string line;
  bool first_line = true;

  while (std::getline(file, line)) {
    // Trim carriage returns and spaces
    line.erase(line.find_last_not_of(" \r\n\t") + 1);
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string cell;
    std::vector<Scalar> row;
    bool is_header = false;

    while (std::getline(ss, cell, ',')) {
      // Trim whitespaces around cell
      cell.erase(0, cell.find_first_not_of(" \t"));
      cell.erase(cell.find_last_not_of(" \t") + 1);
      if (cell.empty()) continue;

      try {
        row.push_back(static_cast<Scalar>(std::stod(cell)));
      } catch (const std::invalid_argument&) {
        if (first_line) {
          is_header = true;
          break; // It's likely a header string, stop parsing this row
        } else {
          throw std::runtime_error("Invalid numeric data in CSV at row " + 
                                   std::to_string(data.size() + 2));
        }
      }
    }

    if (is_header) {
      first_line = false;
      continue;
    }

    if (row.size() < static_cast<size_t>(expected_cols)) {
      throw std::runtime_error("Row in CSV does not have enough columns. Expected " + 
                               std::to_string(expected_cols) + ", got " + std::to_string(row.size()));
    }

    data.push_back(row);
    first_line = false;
  }

  // Define named columns
  std::vector<std::string> col_names = {"x", "y", "z"};
  if (with_normals) {
    col_names.emplace_back("nx");
    col_names.emplace_back("ny");
    col_names.emplace_back("nz");
  }
  if (with_correspondence_id) {
    col_names.emplace_back("correspondence_id");
  }

  // Populate Eigen matrix
  MatrixX mat(data.size(), col_names.size());
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < col_names.size(); ++j) {
      mat(i, j) = data[i][j];
    }
  }

  return NamedColumnMatrix<MatrixX>(mat, col_names);
}

void SaveMatrixToFile(const NamedColumnMatrix<MatrixX>& x_updated,
                      const std::string& path_in, 
                      const std::string& path_out) {
  std::ifstream fin(path_in);
  if (!fin.is_open()) {
    throw std::runtime_error("Cannot open input file: " + path_in);
  }

  std::ofstream fout(path_out);
  if (!fout.is_open()) {
    throw std::runtime_error("Cannot open output file: " + path_out);
  }

  // Use a high level of precision to prevent truncating coordinate values (essential for UTM formats)
  fout << std::setprecision(12);

  std::string line;
  bool first_line = true;
  size_t row_idx = 0;

  Eigen::Index x_col = x_updated.namedColIndex("x");
  Eigen::Index y_col = x_updated.namedColIndex("y");
  Eigen::Index z_col = x_updated.namedColIndex("z");

  while (std::getline(fin, line)) {
    // Strip trailing newlines/carriage returns
    line.erase(line.find_last_not_of(" \r\n") + 1);
    if (line.empty()) {
      fout << "\n";
      continue;
    }

    // Determine if header by checking if the first token is a valid double
    std::stringstream ss(line);
    std::string cell;
    std::getline(ss, cell, ',');
    cell.erase(0, cell.find_first_not_of(" \t"));
    cell.erase(cell.find_last_not_of(" \t") + 1);

    bool is_header = false;
    try {
      (void)std::stod(cell);
    } catch (const std::invalid_argument&) {
      if (first_line) is_header = true;
    }

    if (is_header) {
      fout << line << "\n"; // pass header directly to the output
      first_line = false;
      continue;
    }

    if (row_idx < static_cast<size_t>(x_updated.rows())) {
      // Find the comma boundaries for the first 3 variables to swap them out
      size_t pos1 = line.find(',');
      size_t pos2 = pos1 != std::string::npos ? line.find(',', pos1 + 1) : std::string::npos;
      size_t pos3 = pos2 != std::string::npos ? line.find(',', pos2 + 1) : std::string::npos;

      fout << x_updated(row_idx, x_col) << ","
           << x_updated(row_idx, y_col) << ","
           << x_updated(row_idx, z_col);

      // Print remainder of original CSV sequence unmodified
      if (pos3 != std::string::npos) {
        fout << line.substr(pos3);
      }
      fout << "\n";
      row_idx++;
    } else {
      // Catch-all: output trailing unrelated lines, if they somehow exist
      fout << line << "\n";
    }

    first_line = false;
  }
}
