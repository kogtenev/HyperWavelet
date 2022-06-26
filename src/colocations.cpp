#include <cmath>
#include <fstream>
#include <iostream>

#include "colocations.h"
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

ColocationsMethod::ColocationsMethod(int numLevels, double a, double b):
    _numLevels(numLevels), _dim(2 * IntPow(2, numLevels)), _a(a), _b(b) {

    cout << "Colocation method on interval: (" << a << ", " << b << ")" << endl;
    cout << "Number of refinment levels: " << numLevels << endl; 
    cout << "Dimension of linear system: " << _dim << endl << endl;
    
    int numOfSupports = _dim / 2;
    _x.resize(numOfSupports + 1);
    double h = (_b - _a) / numOfSupports;
    for (int i = 0; i < _x.size(); i++) {
        _x[i] = _a + i * h;
    }

    _x0.resize(_dim);
    for (int i = 0; i < numOfSupports; i++) {
        _x0[2 * i] = _x[i] + h / 4;
        _x0[2 * i + 1] = _x[i] + 3 * h / 4;
    }
}

void ColocationsMethod::FormFullMatrix() {
    cout << "Foming dense system matrix" << endl;

    _mat.resize(_dim, _dim);
    const double h = (_b - _a) / (_x.size() - 1);
    for (int j = 0; j < _dim; j++) {
        int i = 0;
        for (int k = 0; k < _dim / 2; k++) {
            _mat(j, i) = _x[k+1] / (_x[k+1] - _x0[j]) - _x[k] / (_x[k] - _x0[j]) -
                         log(abs(_x[k+1] - _x0[j])) + log(abs(_x[k] - _x0[j]));
            ++i;
            _mat(j, i) = 1. / (_x[k+1] - _x0[j]) - 1. / (_x[k] - _x0[j]);
            ++i;       
        }
    }
}

void ColocationsMethod::FormRhs(const function<double(double)>& f) {
    cout << "Froming rhs" << endl;
    _rhs.resize(_dim);
    for (int i = 0; i < _dim; i++) {
        _rhs(i) = f(_x0[i]);
    }
}

void ColocationsMethod::PrintSolution(const Eigen::VectorXd& x) const {
    cout << "Printing solution" << endl;
    Eigen::VectorXd solution(_dim / 2);
    int i = 0;
    for (int k = 0; k < _dim / 2; k++) {
        double v = (_x[k] + _x[k+1]) / 2;
        solution[k] = v * x(2 * k) + x(2 * k + 1);
    }
    ofstream fout("sol.txt", ios::out);
    fout << solution << endl;
}

}