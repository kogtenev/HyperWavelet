#pragma once

#include <Eigen/Dense>

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
    }
};

class RectangleMesh {
public:
    RectangleMesh(
        int nx, int ny, 
        const std::function<Eigen::Vector3d(double, double)>& surfaceMap);

    const std::vector<Rectangle>& Data() const {return _data;};

private:
    std::vector<Rectangle> _data;
};  

class RectangleSurfaceSolver {
public:
    RectangleSurfaceSolver(
        int nx, int ny, double k,
        const std::function<Eigen::Vector3d(double, double)>& surfaceMap
    ): _mesh(nx, ny, surfaceMap), _k(k), _dim(2*nx*ny), _smootherEpsilon(0.125 / nx / ny) {}

    void FormFullMatrix();
    void FormRhs(const std::function<Eigen::Vector3cd(const Eigen::Vector3d&)>& f);

    const Eigen::MatrixXcd& GetFullMatrix() const {return _fullMatrix;}
    const Eigen::VectorXcd& GetRhs() const {return _rhs;}

    void PlotSolutionMap(const Eigen::VectorXcd& x) const;

private:
    const double _k;
    const int _dim;
    const RectangleMesh _mesh;
    Eigen::MatrixXcd _fullMatrix;
    Eigen::VectorXcd _rhs;

    int _integralPoints = 4;
    double _smootherEpsilon;

    double _Smooth(double r) const;
    Eigen::Vector3cd _RegularKernelPart(const Eigen::Vector3d& j, const Rectangle& X, const Eigen::Vector3d& x0);
    Eigen::Vector3cd _MainKernelPart(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& x);
    Eigen::Matrix2cd _LocalMatrix(const Rectangle& X, const Rectangle& X0);
    Eigen::Matrix2cd _RegularPart(const Rectangle& X, const Rectangle& X0);
};

}