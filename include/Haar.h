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
class Row { 
public:
    Row(Vector data, size_t size, size_t row): data(data), _size(size), row(row) {}

    auto operator[](size_t i) {
        return data[row + i * _size];
    } 

    size_t size() {return _size;}

    using Scalar = typename Vector::Scalar;

public:
    Vector data;
    size_t _size;
    size_t row;
};

template <typename Vector>
class Col { 
public:
    Col(Vector data, size_t size, size_t col): data(data), _size(size), col(col) {}

    auto operator[](size_t i) {
        return data[_size * col + i];
    }

    size_t size() {return _size;}

    using Scalar = typename Vector::Scalar;

public:
    Vector data;
    size_t _size;
    size_t col;
};

template <typename Vector>
void Haar2D(Vector& x, size_t size) {
    for (size_t j = 0; j < size; j++) {
        Col col(x, size, j);
        Haar(col);
    }
    for (size_t i = 0; i < size; i++) {
        Row row(x, size, i);
        Haar(row);
    }
}

template <typename Vector>
void HaarInverse2D(Vector& x, size_t size) {
    for (size_t i = 1; i <= size; i++) {
        Row row(x, size, size - i);
        HaarInverse(row);
    }
    for (size_t j = 1; j <= size; j++) {
        Col col(x, size, size - j);
        HaarInverse(col);
    }
}

template <typename Vector>
class Subvector2D {
public:
    Subvector2D(Vector data, size_t size, int even): data(data), _size(size), even(even) {}

    typename Vector::Scalar operator[](size_t i) {
        return data[2 * i + even];
    }

    size_t size() {return _size;}

    using Scalar = typename Vector::Scalar;

public:  
    Vector data;
    size_t _size;
    int even;
};

