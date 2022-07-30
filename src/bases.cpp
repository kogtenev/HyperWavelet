#include <cmath>
#include <iostream>

#include "bases.h"
#include "helpers.h"

using namespace std;

namespace hyper_wavelet {

double Distance(const Interval& S1, const Interval& S2) {
    if (S1.b < S2.a) {
        return S2.a - S1.b;
    } else if (S2.b < S1.a) {
        return S1.a - S2.b;
    } else {
        return 0.;
    }
}

LinearFunctional::LinearFunctional(
    const Eigen::Vector4d& coefs, 
    double a, double b): _coefs(coefs) {
    
    SetSupport(a, b);
}

void LinearFunctional::SetSupport(double a, double b) {
    _support = {a, b};
    _points(0) = a + (b - a) / 6.;
    _points(1) = a + (b - a) / 3.;
    _points(2) = a + 2. * (b - a) / 3.;
    _points(3) = a + 5. * (b - a) / 6.;
}

const Interval& LinearFunctional::GetSupport() const {
    return _support;
} 

double LinearFunction::HyperSingularIntegral(double x0) const {
    return (_A * _b + _B) / (_b - x0) - (_A * _a + _B) / (_a - x0) 
            - _A * log(abs(_b - x0)) + _A * log(abs(_a - x0)); 
}

double LinearFunction::FredholmIntegral(
    const std::function<double(double, double)>& K, double x0) const {

    const double h = (_b - _a) / (_numOfIntPoints - 1);
    double result = K(x0, _a) * h / 2;
    for (int i = 1; i < _numOfIntPoints - 1; i++) {
        result += K(x0, _a + i * h) * h;
    }
    result += K(x0, _b) * h / 2;
    return result;
}

double LinearFunction::operator() (double x) const {
    if (_a <= x and x <= _b) {
        return _A * x + _B;
    } else {
        return 0.;
    }
}

double PeacewiseLinearFunction::HyperSingularIntegral(double x0) const {
    return _left.HyperSingularIntegral(x0) + _right.HyperSingularIntegral(x0);
}

double PeacewiseLinearFunction::FredholmIntegral(
    const std::function<double(double, double)>& K, double x0) const {

    return _left.FredholmIntegral(K, x0) + _right.FredholmIntegral(K, x0);
}

// TODO: refactor
double PeacewiseLinearFunction::operator() (double x) const {
    return _left(x) + _right(x);
}

void PeacewiseLinearFunction::SetSupport(double a, double b) {
    _a = a;
    _b = b;
    _left.SetSupport(a, (a + b) / 2);
    _right.SetSupport((a + b) / 2, b);
}

void PeacewiseLinearFunction::SetIntegralPointsNumber(int n) {
    _numOfIntPoints = n;
    _left.SetIntegralPointsNumber(n / 2 + 1);
    _right.SetIntegralPointsNumber(n / 2 + 1);
}

Interval PeacewiseLinearFunction::GetSupport() const {
    return {_a, _b};
}

void PeacewiseLinearFunction::Normalize(double a) {
    _left.Normalize(a);
    _right.Normalize(a);
}

Basis::Basis(double a, double b, int numLevels):
    _numLevels(numLevels), _dim(2 * IntPow(2, numLevels)) {

    _data.resize(_dim);

    const double sqrt3 = sqrt(3.);
    const PeacewiseLinearFunction W_0_0 = {-3., 2., -3., 2., 0., 1.};
    const PeacewiseLinearFunction W_0_1 = {3., -1., 3., -1., 0., 1.};
    const PeacewiseLinearFunction W_1_0 = {-4.5, 1., 1.5, -1., 0., 1.};
    const PeacewiseLinearFunction W_1_1 = {-1.5, 0.5, 4.5, -3.5, 0., 1.};    

    _data[0] = W_0_0;
    _data[1] = W_0_1;
    _data[2] = W_1_0;
    _data[3] = W_1_1;

    _data[0].SetSupport(a, b);
    _data[1].SetSupport(a, b);
    _data[2].SetSupport(a, b);
    _data[3].SetSupport(a, b);

    _data[0].SetIntegralPointsNumber(_dim + 1);
    _data[1].SetIntegralPointsNumber(_dim + 1);
    _data[2].SetIntegralPointsNumber(_dim + 1);
    _data[3].SetIntegralPointsNumber(_dim + 1);

    double norm = sqrt(b - a);
    _data[0].Normalize(norm); 
    _data[1].Normalize(norm);
    _data[2].Normalize(norm);
    _data[3].Normalize(norm);

    const double sqrt2 = sqrt(2.);
    double scale = (b - a);
    int numOfSupports = 1;
    int index = 4;

    for (int level = 2; level <= numLevels; level++) {
        norm /= sqrt2;
        numOfSupports *= 2;
        scale /= 2;
        for (int i = 0; i < numOfSupports; i++) {
            _data[index] = W_1_0;
            _data[index].SetSupport(a + scale * i, a + scale * (i + 1));
            _data[index].SetIntegralPointsNumber(_dim / numOfSupports + 1);
            _data[index].Normalize(norm);
            ++index;
            _data[index] = W_1_1;
            _data[index].SetSupport(a + scale * i, a + scale * (i + 1));
            _data[index].SetIntegralPointsNumber(_dim / numOfSupports + 1);
            _data[index].Normalize(norm);
            ++index;
        }
    } 
}

}
