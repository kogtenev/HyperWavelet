#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace hyper_wavelet_2d {

struct Rectangle {
    Eigen::Vector3d a, b, c, d;
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Vector3d e1, e2;
    double area;

    Rectangle() = default;

    Rectangle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, 
        const Eigen::Vector3d& c, const Eigen::Vector3d& d
    ): a(a), b(b), c(c), d(d), 
       center((a + b + c + d) / 4.),  
       area((b - a).norm() * (c - b).norm()), e1(b - a), e2(c - b) {

        e1 /= e1.norm();
        e2 /= e2.norm();
        normal  = e1.cross(e2);
        normal /= normal.norm();
    }
};

class RectangleMesh {
public:
    RectangleMesh(
        int nx, int ny, 
        const std::function<Eigen::Vector3d(double, double)>& surfaceMap);

    void HaarTransform();

    const std::vector<Rectangle>& Data() const {return _data;};

private:
    const std::function<Eigen::Vector3d(double, double)> surfaceMap;
    std::vector<Rectangle> _data;
    int _nx;
    int _ny;
}; 



class RectangleSurfaceSolver {
public:
    RectangleSurfaceSolver(
        int nx, int ny, double k,
        const std::function<Eigen::Vector3d(double, double)>& surfaceMap
    ): _mesh(nx, ny, surfaceMap), _k(k), _nx(nx), _dim(2*nx*ny), _smootherEpsilon(0.125 / nx / ny) {}

    void FormFullMatrix();
    void FormTruncatedMatrix(double threshold, bool print = true);
    void FormMatrixCompressed(double threshold, bool print = true);
    void FormRhs(const std::function<Eigen::Vector3cd(const Eigen::Vector3d&)>& f);

    void HaarTransform();

    const Eigen::MatrixXcd& GetFullMatrix() const {return _fullMatrix;}

    const Eigen::SparseMatrix<std::complex<double>>& GetTruncatedMatrix() const {return _truncMatrix;};

    const Eigen::VectorXcd& GetRhs() const {return _rhs;}

    void PlotSolutionMap(Eigen::VectorXcd& x) const;
    void PrintFullMatrix(const std::string& file) const;

private:
    const double _k;
    const int _dim;
    const int _nx;
    RectangleMesh _mesh;
    Eigen::MatrixXcd _fullMatrix;
    Eigen::SparseMatrix<std::complex<double>> _truncMatrix;
    Eigen::VectorXcd _rhs;

    int _integralPoints = 8;
    double _smootherEpsilon;

    inline double _Smooth(double r) const;
    void _printTruncMatrix();
    void _formBlockCol(Eigen::MatrixXcd& col, int j);
    inline Eigen::Vector3cd _RegularKernelPart(const Eigen::Vector3d& j, const Rectangle& X, const Eigen::Vector3d& x0);
    inline Eigen::Vector3cd _MainKernelPart(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& x);
    inline Eigen::Matrix2cd _LocalMatrix(const Rectangle& X, const Rectangle& X0);
    inline Eigen::Matrix2cd _RegularPart(const Rectangle& X, const Rectangle& X0);
};

inline double PlaneParRectDist(const Rectangle& A, const Rectangle& B);

}