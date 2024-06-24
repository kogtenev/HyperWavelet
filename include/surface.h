#pragma once

#ifdef USE_MKL_PARDISO
    #define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "mesh.h"

namespace hyper_wavelet_2d {

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

    int GetDimension() const {return _dim;}

    void PlotSolutionMap(Eigen::VectorXcd x) const;
    void PrintFullMatrix(const std::string& file) const;
    void PrintSolutionVtk(Eigen::VectorXcd x) const;
    void PrintEsa(const Eigen::VectorXcd& x) const;
    void PrintEsaInverse(const Eigen::MatrixXcd& x) const;

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
    using RectangleSurfaceSolver::GetDimension;

    SurfaceSolver(const double k, const double alpha, const double lambda, const double r);
    
    void WaveletTransform();
    void WaveletTransformInverse(Eigen::VectorXcd& x) const;
    void FormMatrixCompressed(bool print=false);
    void PrintSolutionVtk(const Eigen::VectorXcd x) const { _printVtk(x); }
    void PrintEsa(const Eigen::VectorXcd& x, const std::string& fname) const;
    void PrintEsaInverse(const Eigen::MatrixXcd& x, const std::string& fnamePrefix, int attempt) const;
    void EstimateErrors(const Eigen::VectorXcd& exact, const Eigen::VectorXcd& approx);

private:
    double _CalcEsa(const Eigen::VectorXcd& x, double phi) const;
    double _SuperDistance(int i, int j) const;
    double _epsilon(const WaveletMatrix& wmatrix, int i, int j);
    void _formBlockRow(Eigen::MatrixXcd& blockRow, int k);
    double _alpha = 3, _lambda = 3, _r = 2;
    double _nnzProp;
};

inline double SphereDistance(const Sphere& s1, const Sphere& s2);

}