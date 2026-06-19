#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

using Scalar = float;
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixX3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;
using MatrixX64 = Eigen::Matrix<Scalar, Eigen::Dynamic, 64>;
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using RowVector3 = Eigen::Matrix<Scalar, 1, 3>;
using Vector64 = Eigen::Matrix<Scalar, 64, 1>;
using Matrix64 = Eigen::Matrix<Scalar, 64, 64>;
using SparseMatrix = Eigen::SparseMatrix<Scalar>;
using Triplet = Eigen::Triplet<Scalar>;
