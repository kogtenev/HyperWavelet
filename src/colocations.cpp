#include <cmath>
#include <fstream>
#include <iostream>

#include "colocations.h"
#include "bases.h"

using namespace std;

namespace hyper_wavelet {

ColocationsMethod::ColocationsMethod(int numLevels, double a, double b):
    _numLevels(numLevels), _dim(2 * IntPow(2, numLevels)), 
    _basis(a, b, numLevels), _a(a), _b(b) {

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
    cout << "Forming dense system matrix" << endl;

    _mat.resize(_dim, _dim);
    const double h = (_b - _a) / (_x.size() - 1);
    const auto& w = _basis.Data();
    for (int j = 0; j < _dim; j++) {
        for (int i = 0; i < _dim; i++) {
            _mat(j, i) = w[i].HyperSingularIntegral(_x0[j]);      
        }
    }
}

void ColocationsMethod::FormRhs(const function<double(double)>& f) {
    cout << "Forming rhs" << endl;
    _rhs.resize(_dim);
    for (int i = 0; i < _dim; i++) {
        _rhs(i) = f(_x0[i]);
    }
}

void ColocationsMethod::PrintSolution(const Eigen::VectorXd& x) const {
    cout << "Printing solution" << endl;
    Eigen::VectorXd solution(_dim);
    Eigen::MatrixXd valuesMatrix(_dim, _dim);
    const auto& w = _basis.Data();
    for (int i = 0; i < _dim; i++) {
        for (int j = 0; j < _dim; j++) {
            valuesMatrix(i, j) = w[j](_x0[i]);
        }
    }
    solution = valuesMatrix * x;
    ofstream fout("sol.txt", ios::out);
    fout << solution << endl;
}

}