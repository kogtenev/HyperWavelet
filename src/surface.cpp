#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <numeric>

#include "metis.h"

#include "surface.h"
#include "helpers.h"
#include "Haar.h"
#include "bases.h"

using complex = std::complex<double>;
using hyper_wavelet::Profiler;
using hyper_wavelet::Interval;
using hyper_wavelet::Distance;

using namespace std::complex_literals;

namespace hyper_wavelet_2d {

void PrepareSupports1D(std::vector<Interval>& intervals, int n) {
    intervals.resize(n);
    intervals[0] = {0., 1.};
    intervals[1] = {0., 1.};

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

std::array<Rectangle, 4> Bisection(const Rectangle& A) {
    std::array<Rectangle, 4> result;
    result[0] = Rectangle(A.a, (A.a + A.b) / 2, A.center, (A.a + A.d) / 2);
    result[1] = Rectangle((A.a + A.b) / 2, A.b, (A.b + A.c) / 2, A.center);
    result[2] = Rectangle(A.center, (A.b + A.c) / 2, A.c, (A.c + A.d) / 2);
    result[3] = Rectangle((A.a + A.d) / 2, A.center, (A.c + A.d) / 2, A.d);
    return result;
}

std::vector<Rectangle> RefineRectangle(const Rectangle& rectangle, int numLevels) {
    std::vector<Rectangle> result {rectangle};
    std::vector<Rectangle> helper;
    for (int level = 0; level < numLevels; ++level) {
        helper.resize(0);
        for (const auto& rect: result) {
            auto refined = Bisection(rect);
            helper.insert(helper.end(), refined.begin(), refined.end());
        }
        result = std::move(helper);
    }
    return result;
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

void RectangleMesh::_PrepareSpheres() {
    int nrows = _data.size();
    _wmatrix.spheres.resize(nrows);
    for (int row = 0; row < nrows; ++row) {
        Eigen::Vector3d center;
        center.fill(0.);
        int n = _wmatrix.ends[row] - _wmatrix.starts[row];
        for (int i = _wmatrix.starts[row]; i < _wmatrix.ends[row]; ++i) {
            center += _data[i].center / n;
        }
        double radious = 0.;
        for (int i = _wmatrix.starts[row]; i < _wmatrix.ends[row]; ++i) {
            double new_radious = (_data[i].center - center).norm() + _data[i].diameter;
            if (new_radious > radious) {
                radious = new_radious;
            }
        }
        radious *= r;
        _wmatrix.spheres[row] = {radious, center};
    }
}

const std::function<Eigen::Vector3d(double, double)> _unitMap = [](double x, double y) {
    Eigen::Vector3d result;
    result[0] = x;
    result[1] = y;
    result[2] = 0.;
    return result;
};

inline double RectangleSurfaceSolver::_Smooth(double r) const {
    return r < _eps ? 3*r*r*r/_eps/_eps/_eps - 2*r*r/_eps/_eps : 1.;
}

Eigen::Vector3cd RectangleSurfaceSolver::
_MainKernelPart(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& x) {
    
    Eigen::Vector3cd AM = (x - a).cast<complex>();
    Eigen::Vector3cd BM = (x - b).cast<complex>();

    return (AM / AM.norm() + BM / BM.norm()) * (b - a).norm() / (AM.norm() * BM.norm() + AM.dot(BM));
}

inline Eigen::Vector3cd K1(const Eigen::Vector3d& j, const Eigen::Vector3d& x, const Eigen::Vector3d& y, double k) {
    const double R = (x - y).norm();
    const Eigen::Vector3d& r = (x - y) / R;
    std::complex<double> i = {0, 1};
    return (1. - std::exp(i*k*R) + i*k*R * std::exp(i*k*R)) / R / R / R * 
           (j.cast<complex>() - 3. * r.dot(j) * r.cast<complex>()) +
            k * k * std::exp(i*k*R) / R * (j.cast<complex>() - r.dot(j) * r.cast<complex>());
}

Eigen::Vector3cd RectangleSurfaceSolver::
_RegularKernelPart(const Eigen::Vector3d& J, const Rectangle& X, const Rectangle& X0) {
    const Eigen::Vector3d& x0 = X0.center;
    int levels = (x0 - X.center).norm() / std::sqrt(std::min(X.area, X0.area)) < _adaptation ? _refineLevels : 1;
    Eigen::Vector3cd result;
    result.fill({0., 0.});
    const auto& rectangles = RefineRectangle(X, levels);
    for (const auto& s: rectangles) {
        result += _Smooth((s.center - x0).norm()) * s.area * K1(J, x0, s.center, _k);
    }
    return result;
}

Eigen::Matrix2cd RectangleSurfaceSolver::_LocalMatrix(const Rectangle& X, const Rectangle& X0) {
    Eigen::Matrix2cd a;

    const Eigen::Vector3cd I_ab = _MainKernelPart(X.a, X.b, X0.center);
    const Eigen::Vector3cd I_bc = _MainKernelPart(X.b, X.c, X0.center);
    const Eigen::Vector3cd I_cd = _MainKernelPart(X.c, X.d, X0.center);
    const Eigen::Vector3cd I_da = _MainKernelPart(X.d, X.a, X0.center);

    double scale;
    Eigen::Vector3d t_ab = (X.b - X.a); scale = t_ab.norm();
    t_ab /= (scale > 0.) ? scale : 1;

    Eigen::Vector3d t_bc = (X.c - X.b); scale = t_bc.norm();
    t_bc /= (scale > 0.) ? scale : 1; 

    Eigen::Vector3d t_cd = (X.d - X.c); scale = t_cd.norm(); 
    t_cd /= (scale > 0.) ? scale : 1; 

    Eigen::Vector3d t_da = (X.a - X.d); scale = t_da.norm();
    t_da /= (scale > 0.) ? scale : 1;    

    Eigen::Vector3cd Ke1 = X.e1.cross(X.normal).dot(t_ab) * I_ab;
    Ke1 += X.e1.cross(X.normal).dot(t_bc) * I_bc;
    Ke1 += X.e1.cross(X.normal).dot(t_cd) * I_cd;
    Ke1 += X.e1.cross(X.normal).dot(t_da) * I_da;

    Eigen::Vector3cd Ke2 = X.e2.cross(X.normal).dot(t_ab) * I_ab;
    Ke2 += X.e2.cross(X.normal).dot(t_bc) * I_bc;
    Ke2 += X.e2.cross(X.normal).dot(t_cd) * I_cd;
    Ke2 += X.e2.cross(X.normal).dot(t_da) * I_da;

    Ke1 += _RegularKernelPart(X.e1, X, X0);
    Ke2 += _RegularKernelPart(X.e2, X, X0);

    a(0, 0) =  X0.normal.cast<complex>().cross(Ke1).dot(X0.e2.cast<complex>());
    a(1, 0) = -X0.normal.cast<complex>().cross(Ke1).dot(X0.e1.cast<complex>());
    a(0, 1) =  X0.normal.cast<complex>().cross(Ke2).dot(X0.e2.cast<complex>());
    a(1, 1) = -X0.normal.cast<complex>().cross(Ke2).dot(X0.e1.cast<complex>());

    return a;
}

void RectangleSurfaceSolver::_formBlockCol(Eigen::MatrixXcd& blockCol, int j) {
    blockCol.resize(_dim, 2);
    const auto& rectangles = _mesh.Data();
    #pragma omp parallel for
    for (int i = 0; i < _dim / 2; i++) {
        blockCol.block<2, 2>(2*i, 0) = _LocalMatrix(rectangles[j], rectangles[i]);
    }

    auto V0 = blockCol.col(0);
    auto V1 = blockCol.col(1);

    Subvector2D V0_x(V0, _dim / 2, 0);
    Subvector2D V0_y(V0, _dim / 2, 1);
    Subvector2D V1_x(V1, _dim / 2, 0);
    Subvector2D V1_y(V1, _dim / 2, 1);

    Haar2D(V0_x, _ny, _nx);
    Haar2D(V0_y, _ny, _nx);
    Haar2D(V1_x, _ny, _nx);
    Haar2D(V1_y, _ny, _nx);
}

RectangleSurfaceSolver::RectangleSurfaceSolver(int nx, int ny, double k, 
    const std::function<Eigen::Vector3d(double, double)>& surfaceMap
): _mesh(nx, ny, surfaceMap), _unitMesh(nx, ny, _unitMap), 
   _k(k), _nx(nx), _ny(ny), _dim(2*nx*ny), _eps(1./std::sqrt(nx*ny)/4),
   _adaptation(std::log2(1.*nx*ny)) {

}

RectangleSurfaceSolver::RectangleSurfaceSolver(double k): 
    _k(k), _nx(-1), _ny(-1), _eps(1e-3), _adaptation(1e-2) {}

void RectangleSurfaceSolver::FormFullMatrix() {
    std::cout << "Forming full matrix" << std::endl;
    std::cout << "Matrix size: " << _dim << " x " << _dim << std::endl;
    Profiler profiler;
    _fullMatrix.resize(_dim, _dim);
    const auto& rectangles = _mesh.Data();
    const int n = rectangles.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            _fullMatrix.block<2, 2>(2*i, 2*j) = _LocalMatrix(rectangles[j], rectangles[i]) / std::sqrt(rectangles[j].area);
        }
    }
    std::cout << "Matrix is formed" << std::endl;
    std::cout << "Time for forming matrix: " << profiler.Toc() << " s." << std::endl << std::endl; 
}

void RectangleSurfaceSolver::FormTruncatedMatrix(double threshold, bool print) {
    RectangleMesh haarMesh = _unitMesh;
    haarMesh.HaarTransform();
    std::cout << "Forming truncated matrix\n";
    Profiler profiler;
    const auto& rectangles = haarMesh.Data();
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
    std::cout << "Proportion of nonzeros: " << 1. * triplets.size() / triplets.size() << "\n";

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

void MakeHaarMatrix1D(int n, Eigen::SparseMatrix<double>& H) {
    H.resize(n, n);
    H.makeCompressed();
    std::vector<Eigen::Triplet<double>> triplets;
    double scale = std::sqrt(1./ n);
    for (int i = 0; i < n; i++) {
        triplets.push_back({0, i, scale});
    }
    int row = 1;
    for (int nnz = n; nnz > 1; nnz /= 2) {
        int nrows = n / nnz;
        scale = std::sqrt(1./ nnz);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < nnz / 2; j++) {
                triplets.push_back({row, i*nnz + j, scale});
            }
            for (int j = nnz / 2; j < nnz; j++) {
                triplets.push_back({row, i*nnz + j, -scale});
            }
            row++;
        }
    }
    H.setFromTriplets(triplets.begin(), triplets.end());
}

void RectangleSurfaceSolver::FormMatrixCompressed(double threshold, bool print) {
    auto haarMesh = _unitMesh;
    haarMesh.HaarTransform();
    std::cout << "Forming truncated matrix\n";
    Profiler profiler;

    const auto& rectangles = haarMesh.Data();
    const int N = rectangles.size();
    std::vector<Eigen::Triplet<complex>> triplets;

    _truncMatrix.resize(_dim, _dim);
    _truncMatrix.makeCompressed();
    std::vector<size_t> rowStarts;
    rowStarts.push_back(0);

    size_t nnz = 0;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (PlaneParRectDist(rectangles[j], rectangles[i]) < threshold) {
                triplets.push_back({2*i, 2*j, complex(0.)});
                triplets.push_back({2*i+1, 2*j, complex(0.)});
                triplets.push_back({2*i, 2*j+1, complex(0.)});
                triplets.push_back({2*i+1, 2*j+1, complex(0.)});
                nnz += 4;
            }
        }
        rowStarts.push_back(nnz);
    }

    Eigen::SparseMatrix<double> haarX, haarY;
    MakeHaarMatrix1D(_nx, haarX);
    MakeHaarMatrix1D(_ny, haarY);
    
    for (int k = 0; k < N; k++) {
        const int ky = k / _nx, kx = k % _nx;
        Eigen::MatrixXcd blockB;
        _formBlockCol(blockB, k);
        #pragma omp parallel for
        for (int jy = haarY.outerIndexPtr()[ky]; jy < haarY.outerIndexPtr()[ky+1]; jy++) {
            for (int jx = haarX.outerIndexPtr()[kx]; jx < haarX.outerIndexPtr()[kx+1]; jx++) {
                const int j = _nx * haarY.innerIndexPtr()[jy] + haarX.innerIndexPtr()[jx]; 
                const double haar = haarX.valuePtr()[jx] * haarY.valuePtr()[jy];
                for (size_t tr = rowStarts[j]; tr < rowStarts[j+1]; tr += 4) {
                    const int i = triplets[tr].row() / 2;

                    complex& C_0_0 = const_cast<complex&>(triplets[tr].value());
                    complex& C_1_0 = const_cast<complex&>(triplets[tr+1].value());
                    complex& C_0_1 = const_cast<complex&>(triplets[tr+2].value());
                    complex& C_1_1 = const_cast<complex&>(triplets[tr+3].value());

                    const auto B = blockB.block<2, 2>(2*i, 0);

                    C_0_0 += haar * B(0, 0);
                    C_1_0 += haar * B(1, 0);
                    C_0_1 += haar * B(0, 1);
                    C_1_1 += haar * B(1, 1);
                }
            }
        }
    }

    _truncMatrix.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Time for forming truncated matrix: " << profiler.Toc() << " s.\n"; 
    std::cout << "Proportion of nonzeros: " << 1. * triplets.size() / _dim / _dim << "\n";

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

void RectangleSurfaceSolver::FormRhs(const std::function<Eigen::Vector3cd(const Eigen::Vector3d&)>& f) {
    _rhs.resize(_dim);
    const auto& rectangles = _mesh.Data();
    for (int i = 0; i < rectangles.size(); i++) {
        const Eigen::Vector3cd b = f(rectangles[i].center);
        const auto& e1 = rectangles[i].e1.cast<complex>();
        const auto& e2 = rectangles[i].e2.cast<complex>();
        const auto& n =  rectangles[i].normal.cast<complex>();
        _rhs(2 * i    ) = -n.cross(b).dot(e2);
        _rhs(2 * i + 1) =  n.cross(b).dot(e1);
    }
}

void RectangleSurfaceSolver::PlotSolutionMap(Eigen::VectorXcd x) const {
    std::cout << "Applying inverse Haar transfrom\n";
    Subvector2D E0(x, _dim / 2, 0);
    HaarInverse2D(E0, _ny, _nx);
    Subvector2D E1(x, _dim / 2, 1);
    HaarInverse2D(E1, _ny, _nx);
    std::cout << "Printing solution\n";
    std::ofstream fout("solution.txt", std::ios::out);
    for (int i = 0; i < _dim; i++) {
        fout << std::abs(x(i)) << '\n';
    }
    fout.close();
}

void RectangleSurfaceSolver::HaarTransform() {
    if (_fullMatrix.size()) {
        for (int i = 0; i < _dim; i++) {
            auto col = _fullMatrix.col(i);
            Subvector2D E0(col, _dim / 2, 0);
            Haar2D(E0, _ny, _nx);
            Subvector2D E1(col, _dim / 2, 1);
            Haar2D(E1, _ny, _nx);
        }
        for (int i = 0; i < _dim; i++) {
            auto row = _fullMatrix.row(i);
            Subvector2D E0(row, _dim / 2, 0);
            Haar2D(E0, _ny, _nx);
            Subvector2D E1(row, _dim / 2, 1);
            Haar2D(E1, _ny, _nx);
        }
    }
    if (_rhs.size()) {
        Subvector2D f0(_rhs, _dim / 2, 0);
        Haar2D(f0, _ny, _nx);
        Subvector2D f1(_rhs, _dim / 2, 1);
        Haar2D(f1, _ny, _nx);
    }   
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

void RectangleSurfaceSolver::PrintSolutionVtk(Eigen::VectorXcd x) const {
    std::cout << "Applying inverse Haar transfrom\n";
    Subvector2D E0(x, _dim / 2, 0);
    HaarInverse2D(E0, _ny, _nx);
    Subvector2D E1(x, _dim / 2, 1);
    HaarInverse2D(E1, _ny, _nx);
    _printVtk(x);
}

void RectangleSurfaceSolver::PrintEsa(const Eigen::VectorXcd& x) const {
    const int N = 360;
    Eigen::VectorXcd y = x;
    Subvector2D E0(y, _dim / 2, 0);
    HaarInverse2D(E0, _ny, _nx);
    Subvector2D E1(y, _dim / 2, 1);
    HaarInverse2D(E1, _ny, _nx);
    std::ofstream fout("esa.txt", std::ios::out);
    for (int i = 0; i < N; ++i) {
        fout << _CalcEsa(y, 2 * M_PI * i / N) << '\n';
    }
    fout.close();
}

void RectangleSurfaceSolver::PrintEsaInverse(const Eigen::MatrixXcd& x) const {
    const int N = 181;
    std::ofstream fout("esa.txt", std::ios::out);
    for (int i = 0; i < N; ++i) {
        Eigen::VectorXcd y = x.col(i);
        Subvector2D E0(y, _dim / 2, 0);
        HaarInverse2D(E0, _ny, _nx);
        Subvector2D E1(y, _dim / 2, 1);
        HaarInverse2D(E1, _ny, _nx);
        fout << _CalcEsa(y, 2 * M_PI * 2 * i / 360) << '\n';
    }
    fout.close();
}

void RectangleSurfaceSolver::_printVtk(const Eigen::VectorXcd& x) const {
    std::ofstream fout("solution.vtk", std::ios::out);
    fout << "# vtk DataFile Version 3.0\n";
    fout << "Surface electric current\n";
    fout << "ASCII\n";
    fout << "DATASET POLYDATA\n";
    const int npoints = _mesh.Data().size() * 4;
    const int ncells  = _mesh.Data().size();
    fout << "POINTS " << npoints << " double\n";
    for (const auto& rectangle: _mesh.Data()) {
        fout << rectangle.a << '\n' << rectangle.b << '\n' << rectangle.c << '\n' << rectangle.d << '\n';
    }
    int i = 0;
    fout << "POLYGONS " << ncells << ' ' << 5 * ncells << '\n';
    for (const auto& rectangle: _mesh.Data()) {
        fout << "4 " << i << ' ' << i+1 << ' ' << i+2 << ' ' << i+3 << '\n';
        i += 4; 
    }
    fout << "CELL_DATA " << ncells << '\n';
    fout << "VECTORS J_REAL double\n";
    i = 0;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        J /= std::sqrt(rectangle.area);
        fout << J.real() << '\n';
        i++;
    }
    fout << "VECTORS J_IMAG double\n";
    i = 0;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        J /= std::sqrt(rectangle.area);
        fout << J.imag() << '\n';
        i++;
    }
    fout.close();
}

double RectangleSurfaceSolver::_CalcEsa(const Eigen::VectorXcd& x, double phi) const {
    Eigen::Vector3d tau;
    tau << 0, -std::sin(phi), -std::cos(phi);
    int i = 0;
    Eigen::Vector3cd sigma;
    sigma << 0., 0., 0.;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        const auto& y = rectangle.center;
        const double ds = rectangle.area;
        sigma += std::exp(-1i*_k*tau.dot(y)) * _k * _k * ds * 
            (J - J.dot(tau.cast<complex>()) * tau.cast<complex>());
        i++;
    } 
    return 10. * std::log10(4 * M_PI * sigma.norm() * sigma.norm());
}

inline double PlaneParRectDist(const Rectangle& A, const Rectangle& B) {
    const double dx = Distance({A.a[0], A.b[0]}, {B.a[0], B.b[0]});
    const double dy = Distance({A.b[1], A.c[1]}, {B.b[1], B.c[1]});
    return std::sqrt(dx*dx + dy*dy);
}

SurfaceSolver::SurfaceSolver(
    const double k, 
    const double alpha, 
    const double lambda, 
    const double r
): RectangleSurfaceSolver(k), _alpha(alpha), _lambda(lambda) 
{
    _mesh = RectangleMesh(r);
    _dim = 2 * _mesh.Data().size();
    _mesh.FormWaveletMatrix(); 
}

void SurfaceSolver::WaveletTransform() {
    const auto& wmatrix = _mesh.GetWaveletMatrix();
    if (_fullMatrix.size()) {
        for (int i = 0; i < _dim; i++) {
            auto col = _fullMatrix.col(i);
            Subvector2D E0(col, _dim / 2, 0);
            SurphaseWavelet(E0, wmatrix);
            Subvector2D E1(col, _dim / 2, 1);
            SurphaseWavelet(E1, wmatrix);
        }
        for (int i = 0; i < _dim; i++) {
            auto row = _fullMatrix.row(i);
            Subvector2D E0(row, _dim / 2, 0);
            SurphaseWavelet(E0, wmatrix);
            Subvector2D E1(row, _dim / 2, 1);
            SurphaseWavelet(E1, wmatrix);
        }
    }
    if (_rhs.size()) {
        Subvector2D f0(_rhs, _dim / 2, 0);
        SurphaseWavelet(f0, wmatrix);
        Subvector2D f1(_rhs, _dim / 2, 1);
        SurphaseWavelet(f1, wmatrix);
    } 
}

void SurfaceSolver::WaveletTransformInverse(Eigen::VectorXcd& x) const {
    const auto& wmatrix = _mesh.GetWaveletMatrix();
    Subvector2D E0(x, _dim / 2, 0);
    SurphaseWaveletInverse(E0, wmatrix);
    Subvector2D E1(x, _dim / 2, 1);
    SurphaseWaveletInverse(E1, wmatrix);
}

void SurfaceSolver::FormMatrixCompressed(bool print) {
    std::cout << "Forming truncated matrix" << std::endl;
    Profiler profiler;

    const auto& rectangles = _mesh.Data();
    const auto& wmatrix = _mesh.GetWaveletMatrix();
    const int N = rectangles.size();
    std::vector<Eigen::Triplet<complex>> triplets;

    _truncMatrix.resize(_dim, _dim);
    _truncMatrix.makeCompressed();
    std::vector<size_t> rowStarts;
    rowStarts.push_back(0);

    size_t nnz = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (SphereDistance(wmatrix.spheres[j], wmatrix.spheres[i]) < _epsilon(wmatrix, i, j)) {
                triplets.push_back({2*i, 2*j, complex(0.)});
                triplets.push_back({2*i+1, 2*j, complex(0.)});
                triplets.push_back({2*i, 2*j+1, complex(0.)});
                triplets.push_back({2*i+1, 2*j+1, complex(0.)});
                nnz += 4;
            }
        }
        rowStarts.push_back(nnz);
    }

    for (int k = 0; k < N; k++) {
        Eigen::MatrixXcd blockB;
        _formBlockRow(blockB, k);
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (k < wmatrix.starts[i] || k >= wmatrix.ends[i]) {
                continue;
            }

            double N1 = wmatrix.medians[i] - wmatrix.starts[i];
            double N2 = wmatrix.ends[i] - wmatrix.medians[i];
            double left  = (i > 0) ?  1. * N2 / std::sqrt(N1*N1*N2 + N2*N2*N1) : 1. / sqrt(N1);
            double right = (i > 0) ? -1. * N1 / std::sqrt(N1*N1*N2 + N2*N2*N1) : 0.;
            double wavelet = (k < wmatrix.medians[i]) ? left : right;

            for (size_t tr = rowStarts[i]; tr < rowStarts[i+1]; tr += 4) {
                const int j = triplets[tr].col() / 2;

                complex& C_0_0 = const_cast<complex&>(triplets[tr].value());
                complex& C_1_0 = const_cast<complex&>(triplets[tr+1].value());
                complex& C_0_1 = const_cast<complex&>(triplets[tr+2].value());
                complex& C_1_1 = const_cast<complex&>(triplets[tr+3].value());

                const auto B = blockB.block<2, 2>(0, 2*j);

                C_0_0 += wavelet * B(0, 0);
                C_1_0 += wavelet * B(1, 0);
                C_0_1 += wavelet * B(0, 1);
                C_1_1 += wavelet * B(1, 1);
            }
        }
    }

    _truncMatrix.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Time for truncated matrix forming: " << profiler.Toc() << " s.\n"; 
    std::cout << "Proportion of nonzeros: " << 1. * triplets.size() / _dim / _dim << "\n";

    if (print) {
        std::ofstream fout("trunc_mat.txt", std::ios::out);
        std::cout << "Printing truncated matrix\n";
        for (const auto& triplet: triplets) {
            fout << triplet.col() << ' ' << triplet.row()
                 << ' ' << std::abs(triplet.value()) << '\n';
        }
        fout.close();    
        std::cout << "Done\n";
    }
    std::cout << '\n';
}

void SurfaceSolver::PrintEsa(const Eigen::VectorXcd& x, const std::string& fname) const {
    const int N = 360;
    std::ofstream fout(fname, std::ios::out);
    for (int i = 0; i < N; ++i) {
        fout << _CalcEsa(x, 2 * M_PI * i / N) << '\n';
    }
    fout.close();
}

void SurfaceSolver::PrintEsaInverse(const Eigen::MatrixXcd& x, const std::string& fname) const {
    const int N = 181;
    std::ofstream fout(fname, std::ios::out);
    const auto& wmatrix = _mesh.GetWaveletMatrix();
    for (int i = 0; i < N; ++i) {
        Eigen::VectorXcd y = x.col(i);
        Subvector2D E0(y, _dim / 2, 0);
        SurphaseWaveletInverse(E0, wmatrix);
        Subvector2D E1(y, _dim / 2, 1);
        SurphaseWaveletInverse(E1, wmatrix);
        fout << _CalcEsa(y, 2 * M_PI * 2 * i / 360) << '\n';
    }
    fout.close();
}

double SurfaceSolver::_CalcEsa(const Eigen::VectorXcd& x, double phi) const {
    Eigen::Vector3d tau;
    tau << -std::cos(phi), -std::sin(phi), 0.;
    int i = 0;
    Eigen::Vector3cd sigma;
    sigma << 0., 0., 0.;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        J /= std::sqrt(rectangle.area);
        const auto& y = rectangle.center;
        const double ds = rectangle.area;
        sigma += std::exp(-1i*_k*tau.dot(y)) * _k * _k * ds * 
            (J - J.dot(tau.cast<complex>()) * tau.cast<complex>());
        ++i;
    }
    return 10. * std::log10(4 * M_PI * sigma.norm() * sigma.norm());
}

void SurfaceSolver::_formBlockRow(Eigen::MatrixXcd& blockRow, int k) {
    blockRow.resize(2, _dim);
    const auto& rectangles = _mesh.Data();
    #pragma omp parallel for
    for (int j = 0; j < _dim / 2; j++) {
        blockRow.block<2, 2>(0, 2*j) = _LocalMatrix(rectangles[j], rectangles[k]) / std::sqrt(rectangles[j].area);
    }

    auto V0 = blockRow.row(0);
    auto V1 = blockRow.row(1);

    Subvector2D V0_x(V0, _dim / 2, 0);
    Subvector2D V0_y(V0, _dim / 2, 1);
    Subvector2D V1_x(V1, _dim / 2, 0);
    Subvector2D V1_y(V1, _dim / 2, 1);

    const auto& wmatrix = _mesh.GetWaveletMatrix();

    SurphaseWavelet(V0_x, wmatrix);
    SurphaseWavelet(V0_y, wmatrix);
    SurphaseWavelet(V1_x, wmatrix);
    SurphaseWavelet(V1_y, wmatrix);
}

double SurfaceSolver::_SuperDistance(int i, int j) const {
    const auto& rectangles = _mesh.Data();
    const auto& wmatrix = _mesh.GetWaveletMatrix();
    double distance = std::numeric_limits<double>::infinity();
    for (int n = wmatrix.starts[i]; n < wmatrix.ends[i]; ++n) {
        for (int m = wmatrix.starts[j]; m < wmatrix.ends[j]; ++m) {
            double new_distance = (rectangles[m].center - rectangles[n].center).norm() - 
                    rectangles[n].diameter - rectangles[m].diameter;
            new_distance = std::max(new_distance, 0.);
            distance = std::min(new_distance, distance);
        }
    }
    return distance;
}

double SurfaceSolver::_epsilon(const WaveletMatrix& wmatrix, int i, int j) {
    int lI = wmatrix.rowLevels[i];
    int lJ = wmatrix.rowLevels[j];
    return _mesh.Area() * std::pow(2., _lambda * _mesh.Levels() / 3 - _alpha / 3 * (lI + lJ));
}

void SurfaceSolver::EstimateErrors(const Eigen::VectorXcd& exact, const Eigen::VectorXcd& approx) {
    const Eigen::VectorXcd err = exact - approx;
    double normL1 = 0., normL2 = 0., errNormL1 = 0., errNormL2 = 0.;
    const auto& rectangles = _mesh.Data();
    Eigen::Vector3cd average;
    average.fill(0.);
    for (int i = 0; i < rectangles.size(); ++i) {
        normL2 += (std::norm(exact[2*i]) + std::norm(exact[2*i+1])) * std::sqrt(rectangles[i].area);
        errNormL2 += (std::norm(err[2*i]) + std::norm(err[2*i+1])) * std::sqrt(rectangles[i].area);
    }
    for (int i = 0; i < rectangles.size(); ++i) {
        Eigen::Vector3cd j = exact[2*i] * rectangles[i].e1.cast<complex>() 
            + exact[2*i+1] * rectangles[i].e2.cast<complex>();
        Eigen::Vector3cd j_err = err[2*i] * rectangles[i].e1.cast<complex>() 
            + err[2*i+1] * rectangles[i].e2.cast<complex>();
        normL1 += j.norm() * std::sqrt(rectangles[i].area);
        errNormL1 += j_err.norm() * std::sqrt(rectangles[i].area);
        average += std::sqrt(rectangles[i].area) * j_err;
    }
    std::cout << '\n';
    std::cout << "L1-error: " << errNormL1 / normL1 << std::endl;
    std::cout << "L2-error: " << std::sqrt(errNormL2 / normL2) << std::endl;
    std::cout << "Average error: " << average.norm() << std::endl;
}

inline double SphereDistance(const Sphere& s1, const Sphere& s2) {
    double distance = (s1.center - s2.center).norm() - s1.radious - s2.radious;
    return distance > 0. ? distance : 0.;
}

}