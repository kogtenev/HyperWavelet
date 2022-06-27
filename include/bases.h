#pragma once

#include <vector>

namespace hyper_wavelet {

class LinearFunction {
public:
    LinearFunction(double A, double B, double a, double b):
        _A(A), _B(B), _a(a), _b(b) {}

    double HyperSingularIntegral(double x0) const;
    double operator() (double x) const;
private:
    // Boundary of the support
    double _a, _b;
    // F(x) = Ax + B
    double _A, _B;
};


class PeaceWiseLinearFuntion {
public:
    PeaceWiseLinearFuntion(
        double A, double B, double C, double D, double a, double b
    ): _left(A, B, a, (a + b) / 2), _right(C, D, (a + b) / 2, b) {}

    double HyperSingularIntegral(double x0) const;
    double operator() (double x) const;
private:
    LinearFunction _left;
    LinearFunction _right;
};

class Basis {
public:
    Basis(double a, double b, int dim);
    const std::vector<PeaceWiseLinearFuntion>& Data() const {
        return _data;
    }

private:
    std::vector<PeaceWiseLinearFuntion> _data;
};

}

