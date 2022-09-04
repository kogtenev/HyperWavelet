#include <cmath>
#include <Eigen/Dense>

#include "helpers.h"

#define HaarBuffer Eigen::Matrix<typename Vector::Scalar, Eigen::Dynamic, 1>  

template <typename Vector>
void HaarElem(size_t dim, Vector& x, HaarBuffer& tmp) {
    const double sqrt2 = sqrt(2.);
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[i] = (x[2 * i] + x[2 * i + 1]) / sqrt2;
    }
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[dim / 2 + i] = (x[2 * i] - x[2 * i + 1]) / sqrt2;
    }
    for (size_t i = 0; i < dim; i++) {
        x[i] = tmp[i];
    }
}

template <typename Vector>
void HaarElemInverse(size_t dim, Vector& x, HaarBuffer& tmp) {
    const double sqrt2 = sqrt(2.);
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[2 * i] = (x[i] + x[dim / 2 + i]) / sqrt2;
    }
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[2 * i + 1] = (x[i] - x[dim / 2 + i]) / sqrt2;
    }
    for (size_t i = 0; i < dim; i++) {
        x[i] = tmp[i];
    }
}

template <typename Vector>
void Haar(Vector& x) {
    size_t dim = x.size();
    HaarBuffer tmp(dim);
    while (dim > 1) {
        HaarElem(dim, x, tmp);
        dim /= 2;
    }
}

template <typename Vector>
void HaarInverse(Vector& x) {
    size_t dim = 2;
    HaarBuffer tmp(x.size());
    while (dim <= x.size()) {
        HaarElemInverse(dim, x, tmp);
        dim *= 2;
    }
}

template <typename Vector>
void Haar2D(Vector& x, size_t size) {
    auto& mat = x.reshaped(size, size);
    for (size_t j = 0; j < size; j++) {
        Haar(mat.col(j));
    }
    for (size_t i = 0; i < size; i++) {
        Haar(mat.row(i));
    }
}

template <typename Vector>
void HaarInverse2D(Vector& x, size_t size) {
    auto& mat = x.reshaped(size, size);
    for (size_t i = size - 1; i >= 0; i--) {
        Haar(mat.row(i));
    }
    for (size_t j = size - 1; j >= 0; j--) {
        Haar(mat.col(j));
    }
}
