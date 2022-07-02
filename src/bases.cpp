#include <cmath>
#include <iostream>

#include "bases.h"
#include "helpers.h"

using namespace std;

namespace hyper_wavelet {

LinearFunctional::LinearFunctional(
    const Eigen::Vector4d& coefs, 
    const Eigen::Vector4d& points, 
    double a, double b): _coefs(coefs), _points(points) {}
    

double LinearFunction::HyperSingularIntegral(double x0) const {
    return (_A * _b + _B) / (_b - x0) - (_A * _a + _B) / (_a - x0) 
            - _A * log(abs(_b - x0)) + _A * log(abs(_a - x0)); 
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

void PeacewiseLinearFunction::Normalize(double a) {
    _left.Normalize(a);
    _right.Normalize(a);
}

Basis::Basis(double a, double b, int numLevels):
    _numLevels(numLevels), _dim(2 * IntPow(2, numLevels)) {

    _data.resize(_dim);

    const double sqrt3 = sqrt(3.);
    const PeacewiseLinearFunction W_0_0 = {0., 1., 0., 1., 0., 1};
    const PeacewiseLinearFunction W_0_1 = {2 * sqrt3, -sqrt3, 2 * sqrt3, -sqrt3, 0., 1.};
    const PeacewiseLinearFunction W_1_0 = {-6., 1, -6., 5, 0., 1.};;
    const PeacewiseLinearFunction W_1_1 = {-4. * sqrt3, sqrt3, 4 * sqrt3, -3 * sqrt3, 0., 1.};    

    _data[0] = W_0_0;
    _data[1] = W_0_1;
    _data[2] = W_1_0;
    _data[3] = W_1_1;

    _data[0].SetSupport(a, b);
    _data[1].SetSupport(a, b);
    _data[2].SetSupport(a, b);
    _data[3].SetSupport(a, b);

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
            _data[index].Normalize(norm);
            ++index;
            _data[index] = W_1_1;
            _data[index].SetSupport(a + scale * i, a + scale * (i + 1));
            _data[index].Normalize(norm);
            ++index;
        }
    } 
}

}
