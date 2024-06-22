#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "metis.h"

namespace hyper_wavelet_2d {

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
    std::vector<int> rowLevels;
    std::vector<int> modules;
    std::vector<Sphere> spheres;
};

class RectangleMesh {
public:
    RectangleMesh(int nx, int ny, const std::function<Eigen::Vector3d(double, double)>& surfaceMap);

    RectangleMesh() = default;

    RectangleMesh(const double r); 

    void HaarTransform();

    void FormWaveletMatrix();
    void PrintLocalBases() const;

    const std::vector<Rectangle>& Data() const {return _data;};
    const WaveletMatrix& GetWaveletMatrix() const {return _wmatrix;};
    
    double Area() const {return _fullArea;}
    const std::vector<int>& Levels() const {return _levels;}

private:
    std::vector<Rectangle> _data;
    std::vector<double> _moduleAreas;
    double _fullArea;

    // for rectangle surface solver 
    std::function<Eigen::Vector3d(double, double)> surfaceMap;
    int _nx;
    int _ny;

    // for mesh graph in general case
    std::vector<std::vector<std::pair<int, int>>> _graphEdges;
    std::vector<int> _offsets;
    WaveletMatrix _wmatrix;
    std::vector<int> _levels; 
    double r;

    void _PrepareSpheres();
    void _ReorientLocalBases(const int begin, const int end);
    void _ReadData(const int module);
    void _FormWaveletSubmatrix(const int module);
}; 

}