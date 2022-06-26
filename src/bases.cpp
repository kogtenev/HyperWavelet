#include <cmath>

#include "bases.h"

using namespace std;

double IntPow(int a, int power) {
    int result = 1;
    while (power > 0) {
        if (power % 2 == 0) {
            power /= 2;
            a *= a;
        } else {
            power--;
            result *= a;
        }
    }
    return result;
}

namespace hyper_wavelet {

double PeacewiseLinearFunction::operator() (double x) const {
    if (x >= a and x <= (a + b) / 2) {
        return A * x + C;
    } else if (x > (a + b) / 2 and x <= b) {
        return B * x + D;
    } else {
        return 0.;
    }
}

UnitIntervalBasis::UnitIntervalBasis(int numLevels): 
    _numLevels(numLevels), _dimension(IntPow(2, numLevels + 1)) {

    _basis.reserve(_dimension);

    double A = 0., B = 0., C = 1., D = 1.;
    _basis.push_back({0., 1., A, B, C, D});

    A = B = sqrt(3.) * 2; 
    C = D = sqrt(3.);
    _basis.push_back({0., 1., A, B, C, D});

    double A1 = -6., B1 = -6., C1 = -1., D1 = -5.;
    double A2 = -4 * sqrt(3.), B2 = 4 * sqrt(3.), C2 = sqrt(3.), D2 = -3 * sqrt(3.);

    int numOfSupports = 1;
    const double scale = sqrt(2.);
    for (int level = 1; level <= _numLevels; level++) {
        double step = 1. / numOfSupports;
        double a = 0., b = 1. / numOfSupports;
        for (int i = 0; i < numOfSupports; i++) {
            _basis.push_back({a, b, A1, B1, C1, D1});
            _basis.push_back({a, b, A2, B2, C2, D2});
            a += step;
            b += step;
        }
        A1 *= scale; B1 *= scale; C1 *= scale; D1 *= scale;
        A2 *= scale; B2 *= scale; C2 *= scale; D2 *= scale;
        numOfSupports *= 2; 
    }  
}

const vector<PeacewiseLinearFunction>& UnitIntervalBasis::Data() const {
    return _basis;
}

int UnitIntervalBasis::NumLevels() const {
    return _numLevels;
}

int UnitIntervalBasis::Dimension() const {
    return _dimension;
}

}



