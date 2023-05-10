#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "metis.h"

namespace hyper_wavelet_2d {

enum LinearSolverType {EigenLU, EigenGMRES, PETScGMRES};

struct Rectangle {
    Eigen::Vector3d a, b, c, d;
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Vector3d e1, e2;
    double area;
    double diameter;

    Rectangle() = default;

    Rectangle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, 
              const Eigen::Vector3d& c, const Eigen::Vector3d& d)
    : a(a), b(b), c(c), d(d), center((a + b + c + d) / 4.), 
      diameter(std::max((a - c).norm(), (b - d).norm())) {
        
        e1 = center - (b + c) / 2;
        e2 = center - (d + c) / 2;

        e1 /= e1.norm();
        e2 /= e2.norm();

        normal  = e1.cross(e2);
        normal /= normal.norm();

        area  = (b - a).cross(d - a).norm() / 2;
        area += (b - c).cross(d - c).norm() / 2;
    }
};

struct Sphere {
    double radious; 
    Eigen::Vector3d center;
};

struct WaveletMatrix {
    std::vector<int> starts;
    std::vector<int> medians;
    std::vector<int> ends;
};

struct WaveletTransformation {
    Eigen::SparseMatrix<double, Eigen::RowMajor> haar1D;
    std::vector<WaveletMatrix> wmatrices;
    std::vector<int> offsets;
    int size;
};

class RectangleMesh {
public:
    RectangleMesh(
        int nx, int ny, 
        const std::function<Eigen::Vector3d(double, double)>& surfaceMap);

    RectangleMesh() = default;

    RectangleMesh(const std::string& meshFile, const std::string& graphFile = ""); 

    void HaarTransform();

    void FormWaveletMatrix();

    const std::vector<Rectangle>& Data() const {return _data;};
    const WaveletMatrix& GetWaveletMatrix() const {return _wmatrix;};
    const WaveletTransformation& GetWaveletTransform() const {return _waveletTransform;}

private:
    std::vector<Rectangle> _data;

    // for rectangle surface solver 
    std::function<Eigen::Vector3d(double, double)> surfaceMap;
    int _nx;
    int _ny;

    // for mesh graph in general case
    std::vector<std::pair<int, int>> _graphEdges;
    WaveletMatrix _wmatrix; 

    WaveletTransformation _waveletTransform;

    void _PrepareSpheres();
    void _FormSubMatrix(int nvertices, Rectangle* data, 
            std::vector<std::pair<int, int>>& edges,
            std::vector<std::vector<std::pair<int, int>>>& subgraphs_edges,
            std::vector<int>& barriers, WaveletMatrix* wmatrix, 
            int min_diam);
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
    void PrintEsa(const Eigen::VectorXcd& x) const;

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
    double _CalcEsa(const Eigen::VectorXcd& x, double phi) const;
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

    SurfaceSolver(double k, const std::string& meshFile, const std::string& graphFile = "");
    void WaveletTransform();
    void WaveletTransformInverse(Eigen::VectorXcd& x) const;
    void FormMatrixCompressed(double threshold, bool print=true);
    void FormTruncatedMatrix(double threshold, bool print=true);
    void PrintSolutionVtk(const Eigen::VectorXcd& x) const { _printVtk(x); }
    void PrintEsa(const Eigen::VectorXcd& x) const;
    void EstimateErrors(const Eigen::VectorXcd& exact, const Eigen::VectorXcd& approx);

private:
    double _CalcEsa(const Eigen::VectorXcd& x, double phi) const;
    double _SuperDistance(int i, int j) const;
    void _formBlockRow(Eigen::MatrixXcd& blockRow, int k);
};

inline double SphereDistance(const Sphere& s1, const Sphere& s2);

void MakeHaarMatrix1D(int n, Eigen::SparseMatrix<double, 1>& H);

}