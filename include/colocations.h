#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "bases.h"

namespace hyper_wavelet {

class ConjugateSpace {
public:
    ConjugateSpace(double a, double b, int numLevels);
    const std::vector<LinearFunctional>& Data() const {
        return _data;
    }
    
private:
    double _a, _b;
    int _numLevels;
    int _dim;
    std::vector<LinearFunctional> _data;

};


class ColocationsMethod {
public:
    ColocationsMethod(int numLevels, double a, double b);

    void FormFullMatrix();
    void FormTruncatedMatrix(double threshold);
    void FormRhs(const std::function<double(double)>& f);

    const Eigen::MatrixXd& GetFullMatrix() const {return _mat;}
    const Eigen::SparseMatrix<double>& GetTruncatedMatrix() const;
    const Eigen::VectorXd& GetRhs() const {return _rhs;}
    int GetDimension() const {return _dim;}

    void PrintSolution(const Eigen::VectorXd& x) const;

private:
    // dimension of linear system
    const int _dim;
    // number of refinment levels
    const int _numLevels;
    // borders of the interval
    const double _a, _b;
    
    Basis _basis;
    ConjugateSpace _conjugateSpace;

    Eigen::MatrixXd _mat;
    Eigen::SparseMatrix<double> _truncMat;
    Eigen::VectorXd _rhs;
};

}