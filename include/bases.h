#pragma once

#include <vector>
#include <Eigen/Dense>

namespace hyper_wavelet {

struct Interval {
    double a, b;
};

double Distance(const Interval& S1, const Interval& S2);

class LinearFunction {
public:
    LinearFunction(): _A(0.), _B(0.), _a(0.), _b(0.) {}

    LinearFunction(double A, double B, double a, double b):
        _A(A), _B(B), _a(a), _b(b) {}

    double HyperSingularIntegral(double x0) const;
    double FredholmIntegral(const std::function<double(double, double)>& K, double x0) const;

    double operator() (double x) const;

    void SetSupport(double a, double b) {_a = a; _b = b;}
    void SetIntegralPointsNumber(int n) {_numOfIntPoints = n;}
    void Normalize(double a) {_A /= a; _B /= a;}

private:
    // Boundary of the support
    double _a, _b;
    // F(x) = Ax + B
    double _A, _B;

    int _numOfIntPoints;
};


class PeacewiseLinearFunction {
public:
    PeacewiseLinearFunction() = default;

    PeacewiseLinearFunction(
        double A, double B, double C, double D, double a, double b
    ): _a(a), _b(b), _left(A, B, a, (a + b) / 2), _right(C, D, (a + b) / 2, b) {}

    double HyperSingularIntegral(double x0) const;
    double FredholmIntegral(const std::function<double(double, double)>& K, double x0) const;
    double operator() (double x) const;

    void SetIntegralPointsNumber(int n);
    void SetSupport(double a, double b);
    Interval GetSupport() const;

    void Normalize(double a);
    
private:
    double _a, _b;
    LinearFunction _left;
    LinearFunction _right;
    int _numOfIntPoints;
};


class LinearFunctional {
public:
    LinearFunctional() = default;

    LinearFunctional(
        const Eigen::Vector4d& coefs,   
        double a, double b
    );

    void SetSupport(double a, double b);
    const Interval& GetSupport() const;
    void Normalize(double norm);

    const Eigen::Vector4d& GetPoints() const {return _points;}
    const Eigen::Vector4d& GetCoefs() const {return _coefs;} 

private:
    Eigen::Vector4d _coefs;
    Eigen::Vector4d _points;
    Interval _support;    
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

}

