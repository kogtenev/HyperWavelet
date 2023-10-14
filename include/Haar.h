#include <cmath>
#include <Eigen/Dense>

#include "helpers.h"
#include "surface.h"

#define HaarBuffer Eigen::Matrix<typename Vector::Scalar, Eigen::Dynamic, 1>  

using hyper_wavelet_2d::WaveletMatrix;

template <typename Vector>
void HaarElem(size_t dim, Vector& x, HaarBuffer& tmp, double weight) {
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[i] = (x[2 * i] + x[2 * i + 1]) / weight;
    }
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[dim / 2 + i] = (x[2 * i] - x[2 * i + 1]) / weight;
    }
    for (size_t i = 0; i < dim; i++) {
        x[i] = tmp[i];
    }
}

template <typename Vector>
void HaarElemInverse(size_t dim, Vector& x, HaarBuffer& tmp, double weight) {
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[2 * i] = (x[i] + x[dim / 2 + i]) / weight;
    }
    for (size_t i = 0; i < dim / 2; i++) {
        tmp[2 * i + 1] = (x[i] - x[dim / 2 + i]) / weight;
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
        HaarElem(dim, x, tmp, 2.);
        dim /= 2;
    }
}

template <typename Vector>
void HaarUnweighted(Vector& x) {
    size_t dim = x.size();
    HaarBuffer tmp(dim);
    while (dim > 1) {
        HaarElem(dim, x, tmp, 1.);
        dim /= 2;
    }
}

template <typename Vector>
void HaarInverse(Vector& x) {
    size_t dim = 2;
    HaarBuffer tmp(x.size());
    const double sqrt2 = sqrt(2.);
    while (dim <= x.size()) {
        HaarElemInverse(dim, x, tmp, 1.);
        dim *= 2;
    }
}

template <typename Vector>
class Row { 
public:
    Row(Vector &data, size_t ncols, size_t nrows, size_t row)
    : data(data), ncols(ncols), nrows(nrows), row(row) {}

    auto& operator[](size_t i) {
        return data[row + i * nrows];
    } 

    size_t size() { return ncols; }

    using Scalar = typename Vector::Scalar;

private:
    Vector& data;
    size_t nrows;
    size_t ncols;
    size_t row;
};

template <typename Vector>
class Col { 
public:
    Col(Vector& data, size_t ncols, size_t nrows, size_t col)
    : data(data), ncols(ncols), nrows(nrows), col(col) {}

    auto& operator[](size_t i) {
        return data[nrows * col + i];
    }

    size_t size() { return nrows; }

    using Scalar = typename Vector::Scalar;

private:
    Vector& data;
    size_t nrows;
    size_t ncols;
    size_t col;
};

template <typename Vector>
void Haar2D(Vector& x, size_t ncols, size_t nrows) {
    #pragma omp parallel for
    for (size_t j = 0; j < ncols; j++) {
        Col col(x, ncols, nrows, j);
        Haar(col);
    }
    #pragma omp parallel for
    for (size_t i = 0; i < nrows; i++) {
        Row row(x, ncols, nrows, i);
        Haar(row);
    }
}

template <typename Vector>
void HaarInverse2D(Vector& x, size_t ncols, size_t nrows) {
    #pragma omp parallel for
    for (size_t i = 1; i <= nrows; i++) {
        Row row(x, ncols, nrows, nrows - i);
        HaarInverse(row);
    }
    #pragma omp parallel for
    for (size_t j = 1; j <= ncols; j++) {
        Col col(x, ncols, nrows, ncols - j);
        HaarInverse(col);
    }
}

template <typename Vector>
class Subvector2D {
public:
    Subvector2D(Vector& data, size_t size, int even): data(data), _size(size), even(even) {}

    typename Vector::Scalar& operator[](size_t i) {
        return data[2 * i + even];
    }

    size_t size() {return _size;}

    using Scalar = typename Vector::Scalar;

private:  
    Vector& data;
    size_t _size;
    int even;
};

template <typename Vector> 
void SurphaseWavelet(Vector& x, const WaveletMatrix& wmatrix) {
    Eigen::VectorXcd y(x.size());
    y.fill(0.);
    #pragma omp parallel for
    for (int row = 0; row < x.size(); ++row) {
        double N1 = wmatrix.medians[row] - wmatrix.starts[row];
        double N2 = wmatrix.ends[row] - wmatrix.medians[row];
        double left  = (row > 0) ?  1. * N2 / std::sqrt(N1*N1*N2 + N2*N2*N1) : 1. / sqrt(N1);
        double right = (row > 0) ? -1. * N1 / std::sqrt(N1*N1*N2 + N2*N2*N1) : 0.;
        for (int col = wmatrix.starts[row]; col < wmatrix.medians[row]; ++col) {
            y[row] += left * x[col];
        }
        for (int col = wmatrix.medians[row]; col < wmatrix.ends[row]; ++col) {
            y[row] += right * x[col];
        }
    }
    for (int i = 0; i < x.size(); ++i) {
        x[i] = y[i];
    }
}

template <typename Vector> 
void SurphaseWaveletInverse(Vector& x, const WaveletMatrix& wmatrix) {
    Eigen::VectorXcd y(x.size());
    y.fill(0.);
    for (int col = 0; col < x.size(); ++col) {
        double N1 = wmatrix.medians[col] - wmatrix.starts[col];
        double N2 = wmatrix.ends[col] - wmatrix.medians[col];
        double left  = (col > 0) ?  1. * N2 / std::sqrt(N1*N1*N2 + N2*N2*N1) : 1. / sqrt(N1);
        double right = (col > 0) ? -1. * N1 / std::sqrt(N1*N1*N2 + N2*N2*N1) : 0.;
        #pragma omp parallel for
        for (int row = wmatrix.starts[col]; row < wmatrix.medians[col]; ++row) {
            y[row] += left * x[col];
        }
        #pragma omp parallel for
        for (int row = wmatrix.medians[col]; row < wmatrix.ends[col]; ++row) {
            y[row] += right * x[col];
        }
    }
    for (int i = 0; i < x.size(); ++i) {
        x[i] = y[i];
    }
}
