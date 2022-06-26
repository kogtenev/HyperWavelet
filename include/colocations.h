#pragma once

#include <vector>
#include <Eigen/Dense>

#include "bases.h"

namespace hyper_wavelet {

class ColocationMethod {
public:
    ColocationMethod(int numLevels, double a, double b);
    void FormFullMatrix();
    void FormRhs(const std::function<double(double)>& f);
    const Eigen::MatrixXd& GetFullMatrix() const {return _mat;}
    const Eigen::VectorXd& GetRhs() const {return _rhs;}
    double GetDimension() const {return _dim;}
    void PrintSolution(const Eigen::VectorXd& x) const;

private:
    const int _dim;
    const int _numLevels;
    const double _a, _b;
    std::vector<double> _x0;
    std::vector<double> _x;
    Eigen::MatrixXd _mat;
    Eigen::VectorXd _rhs;
};

}