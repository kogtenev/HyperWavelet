#include <cmath>
#include <iostream>
#include <fstream>

#include "surface.h"

using complex = std::complex<double>;

namespace hyper_wavelet_2d {

RectangleMesh::RectangleMesh(int nx, int ny, 
    const std::function<Eigen::Vector3d(double, double)>& surfaceMap) {

    double hx = 1. / nx, hy = 1. / ny;
    _data.resize(nx * ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            const Eigen::Vector3d a = surfaceMap(i * hx, j * hy);
            const Eigen::Vector3d b = surfaceMap((i + 1) * hx, j * hy);
            const Eigen::Vector3d c = surfaceMap((i + 1) * hx, (j + 1) * hy);
            const Eigen::Vector3d d = surfaceMap(i * hx, (j + 1) * hy);
            _data[nx * j + i] = Rectangle(a, b, c, d);
        }
    }
}

double RectangleSurfaceSolver::_Smooth(double r) const {
    if (r < _smootherEpsilon) {
        return 3 * r * r * r / _smootherEpsilon / _smootherEpsilon / _smootherEpsilon
                 - 2 * r * r / _smootherEpsilon / _smootherEpsilon;
    } else {
        return 1.;
    }
}

Eigen::Vector3cd RectangleSurfaceSolver::
_MainKernelPart(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& x) {
    
    Eigen::Vector3cd AM = (x - a).cast<complex>();
    Eigen::Vector3cd BM = (x - b).cast<complex>();

    return (AM / AM.norm() + BM / BM.norm()) / (AM.norm() * BM.norm() + AM.dot(BM));
}

Eigen::Vector3cd K1(const Eigen::Vector3d& j, const Eigen::Vector3d& x, const Eigen::Vector3d& y, double k) {
    const double R = (x - y).norm();
    const Eigen::Vector3d& r = (x - y) / R;
    std::complex<double> i = {0, 1};
    return (1. - std::exp(i*k*R) + i*k*R * std::exp(i*k*R)) / R / R / R * 
           (j.cast<complex>() - 3. * r.dot(j) * r.cast<complex>()) +
            k * k * std::exp(i*k*R) / R * (j.cast<complex>() - r.dot(j) * r.cast<complex>());
}

Eigen::Vector3cd RectangleSurfaceSolver::
_RegularKernelPart(const Eigen::Vector3d& J, const Rectangle& X, const Eigen::Vector3d& x0) { 
    double hx = (X.b - X.a).norm() / _integralPoints;
    double hy = (X.c - X.b).norm() / _integralPoints;
    Eigen::Vector3cd result;
    result.fill({0., 0.});
    for (int i = 0; i < _integralPoints; i++) {
        for (int j = 0; j < _integralPoints; j++) {
            const Eigen::Vector3d O = X.a + (hx * i) * X.e1 + (hy * j) * X.e2;
            const Rectangle s(O, O + hx * X.e1, O + hx * X.e1 + hy * X.e2, O + hy * X.e2);
            result += _Smooth((s.center - x0).norm()) * s.area * K1(J, x0, s.center, _k);
        }
    }
    return result;
}

Eigen::Matrix2cd RectangleSurfaceSolver::_LocalMatrix(const Rectangle& X, const Rectangle& X0) {
    Eigen::Matrix2cd a;
    Eigen::Vector3cd Ke1 = _MainKernelPart(X.d, X.a, X0.center) - _MainKernelPart(X.b, X.c, X0.center);
    Eigen::Vector3cd Ke2 = _MainKernelPart(X.a, X.b, X0.center) - _MainKernelPart(X.c, X.d, X0.center);
    Ke1 += _RegularKernelPart(X.e1, X, X0.center);
    Ke2 += _RegularKernelPart(X.e2, X, X0.center);
    a(0, 0) = X0.normal.cast<complex>().cross(Ke1).dot(X0.e1.cast<complex>());
    a(1, 0) = X0.normal.cast<complex>().cross(Ke1).dot(X0.e2.cast<complex>());
    a(0, 1) = X0.normal.cast<complex>().cross(Ke2).dot(X0.e1.cast<complex>());
    a(1, 1) = X0.normal.cast<complex>().cross(Ke2).dot(X0.e2.cast<complex>());
    return a;
}

void RectangleSurfaceSolver::FormFullMatrix() {
    std::cout << "Forming full matrix" << std::endl;
    std::cout << "Matrix size: " << _dim << " x " << _dim << std::endl;
    _fullMatrix.resize(_dim, _dim);
    const auto& rectangles = _mesh.Data();
    const int n = rectangles.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            _fullMatrix.block<2, 2>(2*i, 2*j) = _LocalMatrix(rectangles[j], rectangles[i]);
        }
    }
    std::cout << "Matrix is formed" << std::endl << std::endl;
}

void RectangleSurfaceSolver::
FormRhs(const std::function<Eigen::Vector3cd(const Eigen::Vector3d&)>& f) {
    _rhs.resize(_dim);
    const auto& rectangles = _mesh.Data();
    for (int i = 0; i < rectangles.size(); i++) {
        const Eigen::Vector3cd b = f(rectangles[i].center);
        //std::cout << rectangles[i].center << std::endl;
        //std::cout << b << std::endl;
        //_rhs(2 * i    ) = b.dot(rectangles[i].e1.cast<complex>());
        //_rhs(2 * i + 1) = b.dot(rectangles[i].e2.cast<complex>());
        _rhs(2 * i    ) = {1., 0.};
        _rhs(2 * i + 1) = {0., 0.}; 
    }
}

void RectangleSurfaceSolver::PlotSolutionMap(const Eigen::VectorXcd& x) const {
    std::ofstream fout("solution.txt", std::ios::out);
    for (int i = 0; i < _dim; i++) {
        fout << std::abs(x(i)) << '\n';
    }
    fout.close();
}

}