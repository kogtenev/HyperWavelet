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

    _data.resize(_dim);

    Eigen::Vector4d coefs = {0., 1., 0., 0};
    const LinearFunctional L_0_0(coefs, a, b);

    coefs = {0., 0., 1., 0};
    const LinearFunctional L_0_1(coefs, a, b);

    coefs = {1., -1.5, 0.5, 0.};
    const LinearFunctional L_1_0(coefs, a, b);

    coefs = {0., 0.5, -1.5, 1.};
    const LinearFunctional L_1_1(coefs, a, b);

    _data[0] = L_0_0;
    _data[1] = L_0_1;
    _data[2] = L_1_0;
    _data[3] = L_1_1;

    double scale = (b - a);
    int numOfSupports = 1;
    int index = 4;

    for (int level = 2; level <= numLevels; level++) {
        numOfSupports *= 2;
        scale /= 2;
        for (int i = 0; i < numOfSupports; i++) {
            _data[index] = L_1_0;
            _data[index].SetSupport(a + scale * i, a + scale * (i + 1));
            ++index;
            _data[index] = L_1_1;
            _data[index].SetSupport(a + scale * i, a + scale * (i + 1));
            ++index;
        }
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
    const double h = (_b - _a) / _dim;
    for (int i = 0; i < _dim; i++) {
        for (int j = 0; j < _dim; j++) {
            valuesMatrix(i, j) = w[j](_a + h / 2. + i * h);
        }
    }
    solution = valuesMatrix * x;
    ofstream fout("sol.txt", ios::out);
    fout << solution << endl;
}

}