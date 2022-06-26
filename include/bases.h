#pragma once

#include <vector>

namespace hyper_wavelet {

struct PeacewiseLinearFunction {
    PeacewiseLinearFunction(
        double a, double b, double A, double B, double C, double D
    ): a(a), b(b), A(A), B(B), C(C), D(D) {}

    double operator() (double x) const;
    // Support of the function is [a, b]
    double a, b;
    // F(x) = Ax + C, x \in [a, (a + b) / 2]
    // F(x) = Bx + D, x \in ((a + b) / 2, b]
    double A, B, C, D;
};

class UnitIntervalBasis {
public:
    UnitIntervalBasis(int numLevels);
    const std::vector<PeacewiseLinearFunction>& Data() const;
    int NumLevels() const;
    int Dimension() const;

private:
    int _numLevels;
    int _dimension;
    std::vector<PeacewiseLinearFunction> _basis;
};

}

