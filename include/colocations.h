#pragma once

#include <vector>
#include <Eigen/Dense>

#include "bases.h"

namespace hyper_wavelet {

class ColocationsMethod {
public:
    ColocationsMethod(int numLevels, double a, double b);
    void FormFullMatrix();
    void FormRhs(const std::function<double(double)>& f);
    const Eigen::MatrixXd& GetFullMatrix() const {return _mat;}
    const Eigen::VectorXd& GetRhs() const {return _rhs;}
    double GetDimension() const {return _dim;}
    void PrintSolution(const Eigen::VectorXd& x) const;

private:
    // dimension of linear system
    const int _dim;
    // number of refinment levels
    const int _numLevels;
    // borders of the interval
    const double _a, _b;
    // colocation points
    std::vector<double> _x0;
    // points of the interval refinment 
    std::vector<double> _x;
    Eigen::MatrixXd _mat;
    Eigen::VectorXd _rhs;
};

}