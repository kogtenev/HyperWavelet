#pragma once

#include <cmath>
#include <iostream>
#include <iomanip>
#include <ctime>

#include <Eigen/Dense>

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
            for (size_t j = 0; j < matrix.cols(); j++) {
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

template <typename Vector>
class SubVector {
private:
    Vector* data;
    size_t dim;
    int start;
    int inc;

public:
    SubVector(Vector& x, int start, int inc): 
        data(&x), start(start), inc(inc), dim(x.size() / inc) {} 

    size_t size() const {return dim;}

    typename Vector::Scalar& operator[](size_t i) { 
        return data->operator[](start + i * inc); 
    }

};

}