#pragma once

#include <vector>
#include <Eigen/Dense>

#include "bases.h"

namespace hyper_wavelet {

class ColocationMethod {
public:
    ColocationMethod(int numLevels);
    void FormFullMatrix();
    void FormRhs(const std::function<double(double)>& f);
    const Eigen::MatrixXd& GetFullMatrix() const {return _mat;}
    const Eigen::VectorXd& GetRhs() const {return _rhs;}
    void PrintSolution(const Eigen::VectorXd& x) const;

private:
    std::vector<double> _points;
    UnitIntervalBasis _basis;
    Eigen::MatrixXd _mat;
    Eigen::VectorXd _rhs;
};

}