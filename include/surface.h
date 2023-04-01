#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace hyper_wavelet_2d {

enum LinearSolverType {EigenLU, EigenGMRES, PETScGMRES};

struct Rectangle {
    Eigen::Vector3d a, b, c, d;
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Vector3d e1, e2;
    double area;

    Rectangle() = default;

    Rectangle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, 
              const Eigen::Vector3d& c, const Eigen::Vector3d& d)
    : a(a), b(b), c(c), d(d), center((a + b + c + d) / 4.) {
        
        e1 = a - center;
        e2 = b - center;

        e1 /= e1.norm();
        e2 /= e2.norm();

        normal  = e1.cross(e2);
        normal /= normal.norm();

        area  = (b - a).cross(d - a).norm() / 2;
        area += (b - c).cross(d - c).norm() / 2;
    }
};

class RectangleMesh {
public:
    RectangleMesh(
        int nx, int ny, 
        const std::function<Eigen::Vector3d(double, double)>& surfaceMap);

    RectangleMesh() = default;

    RectangleMesh(const std::string& fileName); 

    void HaarTransform();

    const std::vector<Rectangle>& Data() const {return _data;};

private:
    std::vector<Rectangle> _data;

    // for rectangle surface solver 
    std::function<Eigen::Vector3d(double, double)> surfaceMap;
    int _nx;
    int _ny;

    // for general case
    std::vector<std::pair<int, int>> _graphEdges;
}; 

class RectangleSurfaceSolver {
public:
    RectangleSurfaceSolver(int nx, int ny, double k, 
        const std::function<Eigen::Vector3d(double, double)>& surfaceMap);

    RectangleSurfaceSolver(double k);

    void FormFullMatrix();
    void FormTruncatedMatrix(double threshold, bool print = true);
    void FormMatrixCompressed(double threshold, bool print = true);
    void FormRhs(const std::function<Eigen::Vector3cd(const Eigen::Vector3d&)>& f);

    void HaarTransform();

    const Eigen::MatrixXcd& GetFullMatrix() const {return _fullMatrix;}

    const Eigen::SparseMatrix<std::complex<double>>& GetTruncatedMatrix() const {return _truncMatrix;};

    const Eigen::VectorXcd& GetRhs() const {return _rhs;}

    void PlotSolutionMap(Eigen::VectorXcd x) const;
    void PrintFullMatrix(const std::string& file) const;
    void PrintSolutionVtk(Eigen::VectorXcd x) const;

protected:
    const double _k;
    int _dim;
    RectangleMesh _mesh;
    Eigen::MatrixXcd _fullMatrix;
    Eigen::SparseMatrix<std::complex<double>> _truncMatrix;
    Eigen::VectorXcd _rhs;

    const int _refineLevels = 3;
    const double _eps;
    const double _adaptation;

    inline double _Smooth(double r) const;
    void _printVtk(const Eigen::VectorXcd& x) const;
    Eigen::Vector3cd _RegularKernelPart(const Eigen::Vector3d& j, const Rectangle& X, const Rectangle& X0);
    Eigen::Vector3cd _MainKernelPart(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& x);
    Eigen::Matrix2cd _LocalMatrix(const Rectangle& X, const Rectangle& X0);
    Eigen::Matrix2cd _RegularPart(const Rectangle& X, const Rectangle& X0);

private:
    const int _nx, _ny;
    RectangleMesh _unitMesh;

    void _formBlockCol(Eigen::MatrixXcd& col, int j);
};

inline double PlaneParRectDist(const Rectangle& A, const Rectangle& B);

class SurfaceSolver: RectangleSurfaceSolver {
public:
    using RectangleSurfaceSolver::FormFullMatrix;
    using RectangleSurfaceSolver::FormRhs;
    using RectangleSurfaceSolver::GetFullMatrix;
    using RectangleSurfaceSolver::GetTruncatedMatrix;
    using RectangleSurfaceSolver::GetRhs;
    using RectangleSurfaceSolver::PrintFullMatrix;

    SurfaceSolver(double k, const std::string& meshFile);
    void PrintSolutionVtk(Eigen::VectorXcd x) { _printVtk(x); }
    void PrintEsa(const Eigen::VectorXcd& x) const;

private:
    double CalcEsa(const Eigen::VectorXcd& x, double phi) const;
};

}