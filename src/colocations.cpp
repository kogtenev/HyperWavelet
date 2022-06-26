#include <cmath>
#include <fstream>
#include <iostream>

#include "colocations.h"
#include "bases.h"

using namespace std;

namespace hyper_wavelet {

inline double MatElem(const PeacewiseLinearFunction& w, double x) {
    if (w.a == x or w.b == x) {
        cout << "a = " << w.a << endl 
             << "b = " << w.b << endl
             << "x = " << x << endl;
        throw invalid_argument("wrong integraton point");
    }

    return w(w.b) / (w.b - x) - w(w.a) / (w.a - x) +
            w.A * log(abs(x - w.a)) - w.B * log(abs(w.b - x)); 
}

ColocationMethod::ColocationMethod(int numLevels): _basis(numLevels) {
    int dim = _basis.Dimension();
    int numOfSupports = dim;
    double step = 1. / numOfSupports; 

    _points.resize(dim);

    for (int i = 0; i < dim; i++) {
        _points[i] = step / 2 + i * step;
    }

    ofstream fout("points.txt", ios::out);
    for (int i = 0; i < dim; i++) {
        fout << _points[i] << '\n';
    }
}

void ColocationMethod::FormFullMatrix() {
    int dim = _basis.Dimension();
    _mat.resize(dim, dim);
    const auto& w = _basis.Data();
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            _mat(i, j) = MatElem(w[i], _points[j]);
        }
    }
}

void ColocationMethod::FormRhs(const function<double(double)>& f) {
    int dim = _basis.Dimension();
    _rhs.resize(dim);
    for (int i = 0; i < dim; i++) {
        _rhs(i) = f(_points[i]);
    }
}

void ColocationMethod::PrintSolution(const Eigen::VectorXd& x) const {
    int dim = _basis.Dimension();
    Eigen::MatrixXd valuesMatrix(dim, dim);
    const auto& w = _basis.Data();
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            valuesMatrix(i, j) = w[j](_points[i]);
        }
    }
    Eigen::VectorXd solution = valuesMatrix * x;
    ofstream fout("solution.txt", ios::out);
    fout << solution << endl;
    fout.close();
    fout.open("values.txt", ios::out);
    fout << valuesMatrix << endl;
    fout.close();
    fout.open("x.txt", ios::out);
    fout << x << endl;
}

}