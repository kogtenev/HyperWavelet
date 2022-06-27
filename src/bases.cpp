#include <cmath>

#include "bases.h"

using namespace std;

namespace hyper_wavelet {

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

double PeaceWiseLinearFuntion::HyperSingularIntegral(double x0) const {
    return _left.HyperSingularIntegral(x0) + _right.HyperSingularIntegral(x0);
}

// TODO: refactor
double PeaceWiseLinearFuntion::operator() (double x) const {
    return _left(x) + _right(x);
}

Basis::Basis(double a, double b, int dim) {
    const double h = 2 * (b - a) / dim;
    for (int i = 0; i < dim / 2; i++) {
        const double x1 = a + i * h;
        const double x2 = x1 + h;
        _data.push_back({1, 0, 1, 0, x1, x2});
        _data.push_back({0, 1, 0, 1, x1, x2});
    }
}

}



