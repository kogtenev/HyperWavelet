#include <cmath>
#include <Eigen/Dense>

#define HaarBufferType Eigen::Matrix<typename VectorType::Scalar, Eigen::Dynamic, 1>  

template <typename VectorType>
void HaarElem(size_t dim, VectorType& x, HaarBufferType& tmp) {
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

template <typename VectorType>
void HaarElemInverse(size_t dim, VectorType& x, HaarBufferType& tmp) {
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

template <typename VectorType>
void Haar(VectorType& x) {
    size_t dim = x.size();
    HaarBufferType tmp(dim);
    while (dim > 1) {
        HaarElem(dim, x, tmp);
        dim /= 2;
    }
}

template <typename VectorType>
void HaarInverse(VectorType& x) {
    size_t dim = 2;
    HaarBufferType tmp(x.size());
    while (dim <= x.size()) {
        HaarElemInverse(dim, x, tmp);
        dim *= 2;
    }
}