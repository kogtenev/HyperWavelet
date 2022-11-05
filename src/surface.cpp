#include <cmath>
#include <iostream>
#include <fstream>

#include "surface.h"
#include "helpers.h"
#include "Haar.h"
#include "bases.h"

using complex = std::complex<double>;
using hyper_wavelet::Profiler;
using hyper_wavelet::Interval;
using hyper_wavelet::Distance;

namespace hyper_wavelet_2d {

RectangleMesh::RectangleMesh(int nx, int ny, 
    const std::function<Eigen::Vector3d(double, double)>& surfaceMap): 
                                    _nx(nx), _ny(ny), surfaceMap(surfaceMap) {

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

void PrepareSupports1D(std::vector<Interval>& intervals, int n) {
    intervals.resize(n);
    intervals[0] = {0., 1};
    intervals[1] = {0., 1};

    int numOfSupports = 2;
    double h = 0.5;
    int cnt = 2;

    while (numOfSupports < n) {
        for (int i = 0; i < numOfSupports; i++) {
            intervals[cnt] = {i * h, (i + 1) * h};
            cnt++; 
        }
        h /= 2;
        numOfSupports *= 2;
    }

    if (cnt != n) {
       throw std::runtime_error("Cannot prepare 1D Haar supports!");
    }
}

void RectangleMesh::HaarTransform() {
    std::cout << "Preparing mesh for Haar basis\n";

    Profiler profiler;

    std::vector<Interval> supports_x, supports_y;
    PrepareSupports1D(supports_x, _nx);
    PrepareSupports1D(supports_y, _ny);

    for (int j = 0; j < _ny; j++) {
        for (int i = 0; i < _nx; i++) {
            const double a_x = supports_x[i].a;
            const double b_x = supports_x[i].b;
            const double a_y = supports_y[j].a;
            const double b_y = supports_y[j].b;

            Eigen::Vector3d r1 = surfaceMap(a_x, a_y);
            Eigen::Vector3d r2 = surfaceMap(b_x, a_y);
            Eigen::Vector3d r3 = surfaceMap(b_x, b_y);
            Eigen::Vector3d r4 = surfaceMap(a_x, b_y);

            _data[_nx * j + i] = Rectangle(r1, r2, r3, r4);
        }
    }
        
    std::cout << "Time for preparation: " << profiler.Toc() << " s.\n\n";
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

    return (AM / AM.norm() + BM / BM.norm()) * (b - a).norm() / (AM.norm() * BM.norm() + AM.dot(BM));
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

    const Eigen::Vector3cd I_ab = _MainKernelPart(X.a, X.b, X0.center);
    const Eigen::Vector3cd I_bc = _MainKernelPart(X.b, X.c, X0.center);
    const Eigen::Vector3cd I_cd = _MainKernelPart(X.c, X.d, X0.center);
    const Eigen::Vector3cd I_da = _MainKernelPart(X.d, X.a, X0.center);

    Eigen::Vector3d t_ab = (X.b - X.a); t_ab /= t_ab.norm(); 
    Eigen::Vector3d t_bc = (X.c - X.b); t_bc /= t_bc.norm();
    Eigen::Vector3d t_cd = (X.d - X.c); t_cd /= t_cd.norm(); 
    Eigen::Vector3d t_da = (X.a - X.d); t_da /= t_da.norm();    

    Eigen::Vector3cd Ke1 = X.e1.cross(X.normal).dot(t_ab) * I_ab;
    Ke1 += X.e1.cross(X.normal).dot(t_bc) * I_bc;
    Ke1 += X.e1.cross(X.normal).dot(t_cd) * I_cd;
    Ke1 += X.e1.cross(X.normal).dot(t_da) * I_da;

    Eigen::Vector3cd Ke2 = X.e2.cross(X.normal).dot(t_ab) * I_ab;
    Ke2 += X.e2.cross(X.normal).dot(t_bc) * I_bc;
    Ke2 += X.e2.cross(X.normal).dot(t_cd) * I_cd;
    Ke2 += X.e2.cross(X.normal).dot(t_da) * I_da;

    Ke1 += _RegularKernelPart(X.e1, X, X0.center);
    Ke2 += _RegularKernelPart(X.e2, X, X0.center);

    a(0, 0) = X0.normal.cast<complex>().cross(Ke1).dot(X0.e1.cast<complex>());
    a(1, 0) = X0.normal.cast<complex>().cross(Ke1).dot(X0.e2.cast<complex>());
    a(0, 1) = X0.normal.cast<complex>().cross(Ke2).dot(X0.e1.cast<complex>());
    a(1, 1) = X0.normal.cast<complex>().cross(Ke2).dot(X0.e2.cast<complex>());

    return a;
}

void RectangleSurfaceSolver::_formBlockCol(Eigen::MatrixXcd& blockCol, int j) {
    blockCol.resize(_dim, 2);
    const auto& rectangles = _mesh.Data();
    for (int i = 0; i < _dim; i += 2) {
        blockCol.block<2, 2>(2*i, 0) = _LocalMatrix(rectangles[j], rectangles[i]);
    }

    auto V0 = blockCol.col(0);
    auto V1 = blockCol.col(1);

    Subvector2D V0_x(V0, _dim / 2, 0);
    Subvector2D V0_y(V0, _dim / 2, 1);
    Subvector2D V1_x(V0, _dim / 2, 0);
    Subvector2D V1_y(V0, _dim / 2, 1);

    Haar2D(V0_x, _nx);
    Haar2D(V0_y, _nx);
    Haar2D(V1_x, _nx);
    Haar2D(V1_y, _nx);
}

void RectangleSurfaceSolver::FormFullMatrix() {
    std::cout << "Forming full matrix" << std::endl;
    std::cout << "Matrix size: " << _dim << " x " << _dim << std::endl;
    Profiler profiler;
    _fullMatrix.resize(_dim, _dim);
    const auto& rectangles = _mesh.Data();
    const int n = rectangles.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            _fullMatrix.block<2, 2>(2*i, 2*j) = _LocalMatrix(rectangles[j], rectangles[i]);
        }
    }
    std::cout << "Matrix is formed" << std::endl;
    std::cout << "Time for forming matrix: " << profiler.Toc() << " s." << std::endl << std::endl; 
}

void RectangleSurfaceSolver::FormTruncatedMatrix(double threshold, bool print) {
    _mesh.HaarTransform();
    std::cout << "Forming truncated matrix\n";
    Profiler profiler;
    const auto& rectangles = _mesh.Data();
    const int n = rectangles.size();
    std::vector<Eigen::Triplet<complex>> triplets;
    _truncMatrix.resize(_dim, _dim);
    _truncMatrix.makeCompressed();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (PlaneParRectDist(rectangles[j], rectangles[i]) < threshold) {
                triplets.push_back({2*i, 2*j, _fullMatrix(2*i, 2*j)});
                triplets.push_back({2*i+1, 2*j, _fullMatrix(2*i+1, 2*j)});
                triplets.push_back({2*i, 2*j+1, _fullMatrix(2*i, 2*j+1)});
                triplets.push_back({2*i+1, 2*j+1, _fullMatrix(2*i+1, 2*j+1)});
            }
        }
    }
    _truncMatrix.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Time for forming truncated matrix: " << profiler.Toc() << " s.\n"; 
    std::cout << "Proportion of nonzeros: " << 1. * triplets.size() / _fullMatrix.size() << "\n";

    if (print) {
        std::ofstream fout("trunc_mat.txt", std::ios::out);
        std::cout << "Printing truncated matrix" << '\n';
        for (const auto& triplet: triplets) {
            fout << triplet.col() << ' ' << triplet.row()
                 << ' ' << std::abs(triplet.value()) << '\n';
        }
        fout.close();    
    }
    std::cout << '\n';
}

void MakeHaarMatrix1D(int n, Eigen::MatrixXd& H) {
    H.resize(n, n);
    H.fill(0.);
    for (int i = 0; i < n; i++) {
        H(i, i) = 1.;
    }
    for (int j = 0; j < n; j++) {
        auto col = H.col(j);
        Haar(col);
    }
}

void RectangleSurfaceSolver::FormMatrixCompressed(double threshold, bool print) {
    _mesh.HaarTransform();
    std::cout << "Forming truncated matrix\n";
    Profiler profiler;

    const auto& rectangles = _mesh.Data();
    const int N = rectangles.size();
    std::vector<Eigen::Triplet<complex>> triplets;

    _truncMatrix.resize(_dim, _dim);
    _truncMatrix.makeCompressed();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (PlaneParRectDist(rectangles[j], rectangles[i]) < threshold) {
                triplets.push_back({2*i, 2*j, 0.});
                triplets.push_back({2*i+1, 2*j, 0.});
                triplets.push_back({2*i, 2*j+1, 0.});
                triplets.push_back({2*i+1, 2*j+1, 0.});
            }
        }
    }

    Eigen::MatrixXd haarX, haarY;
    MakeHaarMatrix1D(_nx, haarX);
    MakeHaarMatrix1D(_nx, haarY);

    for (int k = 0; k < N; k++) {
        Eigen::MatrixXcd blockB;
        _formBlockCol(blockB, k);
        for (size_t tr = 0; tr < triplets.size(); tr += 4) {
            const int i = triplets[tr].row() / 2;
            const int j = triplets[tr].col() / 2;
            const double haar = haarX(j % _nx, k / _nx) * haarY(j / _nx, k / _nx);

            complex& C_0_0 = const_cast<complex&>(triplets[tr].value());
            complex& C_1_0 = const_cast<complex&>(triplets[tr+1].value());
            complex& C_0_1 = const_cast<complex&>(triplets[tr+2].value());
            complex& C_1_1 = const_cast<complex&>(triplets[tr+3].value());

            auto B = blockB.block<2, 2>(i, 0);
            C_0_0 += B(0, 0);
            C_1_0 += B(1, 0);
            C_0_1 += B(0, 1);
            C_1_1 += B(1, 1);
        }
    }
}

void RectangleSurfaceSolver::
FormRhs(const std::function<Eigen::Vector3cd(const Eigen::Vector3d&)>& f) {
    _rhs.resize(_dim);
    const auto& rectangles = _mesh.Data();
    for (int i = 0; i < rectangles.size(); i++) {
        const Eigen::Vector3cd b = f(rectangles[i].center);
        const auto& e1 = rectangles[i].e1.cast<complex>();
        const auto& e2 = rectangles[i].e2.cast<complex>();
        const auto& n =  rectangles[i].normal.cast<complex>();
        _rhs(2 * i    ) = -n.cross(b).dot(e1);
        _rhs(2 * i + 1) = -n.cross(b).dot(e2);
    }
}

void RectangleSurfaceSolver::PlotSolutionMap(Eigen::VectorXcd& x) const {
    std::cout << "Applying inverse Haar transfrom\n";
    Subvector2D E0(x, _nx, 0);
    HaarInverse2D(E0, _nx);
    Subvector2D E1(x, _nx, 1);
    HaarInverse2D(E1, _nx);
    std::cout << "Printing solution\n";
    std::ofstream fout("solution.txt", std::ios::out);
    for (int i = 0; i < _dim; i++) {
        fout << std::abs(x(i)) << '\n';
    }
    fout.close();
}

void RectangleSurfaceSolver::HaarTransform() {
    for (int i = 0; i < _dim; i++) {
        auto col = _fullMatrix.col(i);
        Subvector2D E0(col, _dim / 2, 0);
        Haar2D(E0, _nx);
        Subvector2D E1(col, _dim / 2, 1);
        Haar2D(E1, _nx);
    }
    for (int i = 0; i < _dim; i++) {
        auto row = _fullMatrix.row(i);
        Subvector2D E0(row, _dim / 2, 0);
        Haar2D(E0, _nx);
        Subvector2D E1(row, _dim / 2, 1);
        Haar2D(E1, _nx);
    }
    Subvector2D f0(_rhs, _dim / 2, 0);
    Haar2D(f0, _nx);
    Subvector2D f1(_rhs, _dim / 2, 1);
    Haar2D(f1, _nx);   
}

void RectangleSurfaceSolver::PrintFullMatrix(const std::string& file) const {
    std::ofstream fout(file, std::ios::out);
    for (int i = 0; i < _dim; i++) {
        for (int j = 0; j < _dim; j++) {
            fout << std::abs(_fullMatrix(i, j)) << ' ';
        }
        fout << '\n';
    }
}

double PlaneParRectDist(const Rectangle& A, const Rectangle& B) {
    Interval A1 = {A.a[0], A.b[0]};
    Interval B1 = {B.a[0], B.b[0]};
    Interval A2 = {A.b[1], A.c[1]};
    Interval B2 = {B.b[1], B.c[1]};
    const double dx = Distance(A1, B1);
    const double dy = Distance(A2, B2);
    return std::sqrt(dx*dx + dy*dy);
}

}