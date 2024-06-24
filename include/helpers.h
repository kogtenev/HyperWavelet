#pragma once

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <numeric>

#ifdef USE_MKL_PARDISO
    #define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include "metis.h"

namespace hyper_wavelet {

// return a^{power}
int IntPow(int a, int power);

class Profiler {
public:
    Profiler();
    void Tic();
    double Toc();

private:
    double _time;
};


template <typename T>
void PrintSparsityTable(const Eigen::MatrixBase<T>& matrix) {
    const size_t size = matrix.rows() * matrix.cols();
    const double norm = matrix.template lpNorm<Eigen::Infinity>();
    double tol = 1.;
    std::cout << "Number of elements: " << size << std::endl;
    std::cout << "Sparsity table" << std::endl;
    std::cout << "Tolerance" << std::setw(15) << "Proportion" << std::endl;  
    for (int k = 0; k < 20; ++k) {
        size_t count = 0;
        for (size_t i = 0; i < matrix.rows(); ++i) {
            for (size_t j = 0; j < matrix.cols(); ++j) {
                if (std::abs(matrix(i, j)) / norm <= tol) {
                    ++count;
                }
            }
        }
        if (count == 0) {
            break;
        }
        std::cout << tol << std::setw(15) << 1. - 1. * count / size << std::endl;
        tol /= 10.;
    }
    std::cout << std::endl;
}

class SegmentTree {
public:
    SegmentTree(const double a, const double b, int nSegments);
    int Find(double x) const;
    double Median(int i) { return (_data[i] + _data[i+1]) / 2; }
private:
    const Eigen::ArrayXd _data;
};

void CartesianToSphere(const Eigen::Vector3d& n, double& phi, double& theta);

class Hedgehog {
public:
    Hedgehog(const double phi, const double theta);
    Eigen::Vector3d Comb(const Eigen::Vector3d& n);

private:
    Eigen::Vector3d e1, e2, e3;
    const double eps = 1e-14;
};

int DistanceToTrue(const int row, const int col, const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& array);

class Logger {
public:
    void Open(const std::string& fname) { _stream.open(fname, std::ios::out); }
    std::ofstream& Stream() { return _stream; }
private:
    std::ofstream _stream;
};

template<typename T>
Logger& operator<<(Logger& logger, const T& t) {
    std::cout << t;
    logger.Stream() << t;
    return logger;
}

}