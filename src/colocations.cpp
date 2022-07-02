#include <cmath>
#include <fstream>
#include <iostream>

#include "colocations.h"
#include "bases.h"
#include "helpers.h"

using namespace std;

namespace hyper_wavelet {

ConjugateSpace::ConjugateSpace(double a, double b, int numLevels):
    _a(a), _b(b), _numLevels(numLevels), _dim(2 * IntPow(2, numLevels)) {

    int numOfSupports = _dim / 2;
    const double h = (_b - _a) / numOfSupports;
    vector<double> _x(numOfSupports+1);
    for (int i = 0; i < _x.size(); i++) {
        _x[i] = _a + i * h;
    }

    vector<double> _x0(_dim);
    for (int i = 0; i < numOfSupports; i++) {
        _x0[2 * i] = _x[i] + h / 4;
        _x0[2 * i + 1] = _x[i] + 3 * h / 4;
    }

    _data.resize(_dim);
    for (int i = 0; i < _dim; i++) {
        Eigen::Vector4d coefs;
        coefs.fill(0.);
        coefs(3) = 1.;
        Eigen::Vector4d points;
        points.fill(_x0[i]);
        _data[i] = {coefs, points, a, b};
    }
}


ColocationsMethod::ColocationsMethod(int numLevels, double a, double b):
    _numLevels(numLevels), _dim(2 * IntPow(2, numLevels)), 
    _basis(a, b, numLevels), _conjugateSpace(a, b, numLevels), _a(a), _b(b) {

    cout << "Colocation method on interval: (" << a << ", " << b << ")" << endl;
    cout << "Number of refinment levels: " << numLevels << endl; 
    cout << "Dimension of linear system: " << _dim << endl << endl;
}

void ColocationsMethod::FormFullMatrix() {
    cout << "Forming dense system matrix" << endl;

    _mat.resize(_dim, _dim);
    const auto& w = _basis.Data();
    for (int j = 0; j < _dim; j++) {
        for (int i = 0; i < _dim; i++) {
            const auto& c = _conjugateSpace.Data()[j].GetCoefs();
            const auto& p = _conjugateSpace.Data()[j].GetPoints();

            _mat(j, i) = c(0) * w[i].HyperSingularIntegral(p(0)) +
                         c(1) * w[i].HyperSingularIntegral(p(1)) +
                         c(2) * w[i].HyperSingularIntegral(p(2)) +
                         c(3) * w[i].HyperSingularIntegral(p(3));    
        }
    }
}

void ColocationsMethod::FormRhs(const function<double(double)>& f) {
    cout << "Forming rhs" << endl;
    _rhs.resize(_dim);
    for (int i = 0; i < _dim; i++) {
        const auto& c = _conjugateSpace.Data()[i].GetCoefs();
        const auto& p = _conjugateSpace.Data()[i].GetPoints();
        _rhs(i) = c(0) * f(p(0)) + c(1) * f(p(1)) + 
                  c(2) * f(p(2)) + c(3) * f(p(3));
    }
}

void ColocationsMethod::PrintSolution(const Eigen::VectorXd& x) const {
    cout << "Printing solution" << endl;
    Eigen::VectorXd solution(_dim);
    Eigen::MatrixXd valuesMatrix(_dim, _dim);
    const auto& w = _basis.Data();
    const auto& l = _conjugateSpace.Data();
    for (int i = 0; i < _dim; i++) {
        for (int j = 0; j < _dim; j++) {
            valuesMatrix(i, j) = w[j](l[i].GetPoints()(0));
        }
    }
    solution = valuesMatrix * x;
    ofstream fout("sol.txt", ios::out);
    fout << solution << endl;
}

}