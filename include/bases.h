#pragma once

#include <vector>

namespace hyper_wavelet {

class LinearFunction {
public:
    LinearFunction(): _A(0), _B(0), _a(0), _b(0) {}

    LinearFunction(double A, double B, double a, double b):
        _A(A), _B(B), _a(a), _b(b) {}

    double HyperSingularIntegral(double x0) const;
    double operator() (double x) const;
    void SetSupport(double a, double b) {_a = a; _b = b;}
    void Normalize(double a) {_A /= a; _B /= a;}

private:
    // Boundary of the support
    double _a, _b;
    // F(x) = Ax + B
    double _A, _B;
};


class PeacewiseLinearFunction {
public:
    PeacewiseLinearFunction() = default;

    PeacewiseLinearFunction(
        double A, double B, double C, double D, double a, double b
    ): _a(a), _b(b), _left(A, B, a, (a + b) / 2), _right(C, D, (a + b) / 2, b) {}

    double HyperSingularIntegral(double x0) const;
    double operator() (double x) const;
    void SetSupport(double a, double b);
    void Normalize(double a);
    
private:
    double _a, _b;
    LinearFunction _left;
    LinearFunction _right;
};

class Basis {
public:
    Basis(double a, double b, int numLevels);
    const std::vector<PeacewiseLinearFunction>& Data() const {
        return _data;
    }
    int Dimension() const {return _dim;}
    int NumLevels() const {return _numLevels;}

private:
    int _numLevels;
    int _dim;
    std::vector<PeacewiseLinearFunction> _data;
};

int IntPow(int a, int power);

}

