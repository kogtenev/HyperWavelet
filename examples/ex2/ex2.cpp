#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>

#include "Haar.h"
#include "helpers.h"
#include "bases.h"
#include "petsc.h"

using namespace std;
using namespace hyper_wavelet;


class ColocationsMethodHaar {
public:
    ColocationsMethodHaar(int numLevels, double a, double b): 
        _a(a), _b(b), _numLevels(numLevels), _dim(IntPow(2, numLevels)) {

        cout << "Colocation method on interval: (" << a << ", " << b << ")" << endl;
        cout << "Number of refinment levels: " << numLevels << endl; 
        cout << "Dimension of linear system: " << _dim << endl << endl;

        _t0.resize(_dim + 2);
        _t.resize(_dim + 2);
        const double h = (_b - _a) / (_dim + 1);
        for (int i = 0; i <= _dim + 1; i++) {
            _t0[i] = _a + i * h;
        }
        for (int i = 1; i <= _dim + 1; i++) {
            _t[i] = _a + (1. * i - 0.5) * h;
        }

        _supports.reserve(_dim);
        _supports.push_back({_a + h/2, _b - h/2});
        _supports.push_back({_a + h/2, _b - h/2});

        int numOfSupports = 1;
        double step = (_b - _a - h);

        for (int level = 2; level <= numLevels; level++) {
            numOfSupports *= 2;
            step /= 2;
            for (int i = 0; i < numOfSupports; i++) {
                _supports.push_back({_a + h/2 + step * i, _a + h/2 + step * (i + 1)});
            }
        }

        _levelForIndex.push_back(0);
        _levelForIndex.push_back(1);
        int numOfFunctions = 1;
        for (int level = 2; level <= numLevels; level++) {
            numOfFunctions *= 2;
            for (int i = 0; i < numOfFunctions; ++i) {
                _levelForIndex.push_back(level);
            }
        }

        if (_dim != _levelForIndex.size()) {
            throw std::runtime_error("Something wrong with refinment levels!");
        }
    }

    void FormFullMatrixUseQuads(const std::function<double(double, double)>& K) {
        cout << "Forming dense system matrix" << endl;
        _mat.resize(_dim, _dim);
        const double h = (_b - _a) / (_dim + 1);
        for (int i = 1; i <= _dim; i++) {
            for (int j = 1; j <= _dim; j++) {
                _mat(i-1, j-1) = -1.0 / (_t0[i] - _t[j]) + 1.0 / (_t0[i] - _t[j+1]) + K(_t0[i], _t0[j]) * h;
            }
        }
        for (int j = 0; j < _dim; j++) {
            auto col = _mat.col(j);
            Haar(col);
        }
        for (int i = 0; i < _dim; i++) {
            auto row = _mat.row(i);
            HaarUnweighted(row);
        }
        _fullMatrixIsFormed = true;
    }

    void FormFullMatrixExact(const std::function<double(double, double)>& intK) {
        cout << "Forming dense system matrix" << endl;
        _mat.resize(_dim, _dim);
        for (int i = 1; i <= _dim; i++) {
            for (int j = 1; j <= _dim; j++) {
                _mat(i-1, j-1) = -1.0 / (_t0[i] - _t[j]) + 1.0 / (_t0[i] - _t[j+1]) + intK(_t0[i], _t[j+1]) - intK(_t0[i], _t[j]);
            }
        }
        for (int j = 0; j < _dim; j++) {
            auto col = _mat.col(j);
            Haar(col);
        }
        for (int i = 0; i < _dim; i++) {
            auto row = _mat.row(i);
            HaarUnweighted(row);
        }
        _fullMatrixIsFormed = true;
    }

    void FormRhs(const std::function<double(double)>& f) {
        _rhs.resize(_dim);
        for (int i = 1; i <= _dim; i++) {
            _rhs(i-1) = f(_t0[i]);
        }
        Haar(_rhs);
    }

    void FormTruncatedMatrix(bool printMatrix) {
        cout << "Forming truncated system matrix" << endl;

        _truncMat.resize(_dim, _dim);
        std::vector<Eigen::Triplet<double>> triplets;

        for (int i = 0; i < _dim; i++) {
            for (int j = 0; j < _dim; j++) {
                if (Distance(_supports[i], _supports[j]) <= _espilon(i, j)) {
                    triplets.push_back({i, j, _mat(i, j)});
                }
            }
        }

        _truncMat.setFromTriplets(triplets.begin(), triplets.end());

        cout << "Truncated system matrix is formed\n"
             << "Proportion of nonzeros: " 
             << 1. * _truncMat.nonZeros() / _truncMat.size()
             << endl; 

        // TODO: Learn about Eigen's sparse matrix output,
        // refactor truncated matrix output.
        if (printMatrix) {
            ofstream fout("trunc_mat.txt", ios::out);
            cout << "Printing truncated matrix" << endl;
            for (const auto& triplet: triplets) {
                fout << triplet.col() << ' ' << triplet.row()
                     << ' ' << triplet.value() << '\n';
            }
            fout.close();
        }

        cout << endl;
    }

    const Eigen::MatrixXd& GetFullMatrix() const {
        return _mat;
    }

    const Eigen::VectorXd& GetRhs() const {
        return _rhs;
    }

    const Eigen::SparseMatrix<double>& GetTruncatedMatrix() const {
        return _truncMat;
    }

    void PrintSolution(const Eigen::VectorXd& x, const string& fileName) const {
        Eigen::VectorXd sol = x;
        HaarInverse(sol);
        std::ofstream fout(fileName, std::ios::out);
        fout << sol;
        fout.close();
    }

private:
    double _a, _b;
    int _numLevels;
    int _dim;
    bool _fullMatrixIsFormed = false;
    std::vector<Interval> _supports;
    std::vector<double> _t0, _t;
    Eigen::MatrixXd _mat;
    Eigen::VectorXd _rhs;
    Eigen::SparseMatrix<double> _truncMat;
    std::vector<int> _levelForIndex;

    double alpha = 2.5, lambda = 10. / 3;
    double _espilon(int i, int j) {
        int levelI = _levelForIndex[i];
        int levelJ = _levelForIndex[j];
        return std::pow(2., lambda / 3 * _numLevels - alpha * (levelI + levelJ) / 3);
    }
};

function<double(double)> f = [](double x) { return -2 * M_PI; };

function<double(double, double)> K = [](double s0, double s) {
    return abs(s0 - s) > 1e-7 ? -cos(s0 - s) / (2. - 2. * cos(s0 - s)) +
        2 * sin(s0 - s) * sin(s0 - s) / (2. - 2. * cos(s0 - s)) / (2. - 2. * cos(s0 - s)) -
        1 / (s - s0) / (s - s0) : 0.; 
};

function<double(double, double)> intK = [](double s0, double s) {
    return abs(s0 - s) > 1e-7 ? sin(s0 - s) / (2. - 2. * cos(s0 - s)) - 1. / (s0 - s) : 0.;
};

static char petsc_magic[] = "Appends to an ASCII file.\n\n";

int main(int argc, char* argv[]) {
    //PetscInitialize(&argc, &argv, (char*)0, petsc_magic);

    int numLevels = stoi(argv[1]);
    double a = stod(argv[2]);
    double b = stod(argv[3]);
    bool printFull = stoi(argv[4]);
    bool printTrunctated = stoi(argv[5]);

    ColocationsMethodHaar method(numLevels, a, b);
    //method.FormFullMatrixUseQuads(K);
    method.FormFullMatrixExact(intK);
    method.FormRhs(f);

    const Eigen::MatrixXd& A = method.GetFullMatrix();
    const Eigen::VectorXd& rhs = method.GetRhs();
    
    if (printFull) {
        cout << "Printing full matrix" << endl;
        ofstream fout("mat.txt", ios::out);
        fout << A;
        fout.close();
    }

    cout << "Solving full linear system" << endl;
    Profiler profiler;
    Eigen::VectorXd x = A.lu().solve(rhs);
    cout << "Time for solution: " << profiler.Toc() << " s." << endl;

    method.FormTruncatedMatrix(printTrunctated);
    const auto& sparseMatrix = method.GetTruncatedMatrix();

    cout << "Solving system with truncated matrix" << endl;
    profiler.Tic();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(sparseMatrix);
    Eigen::VectorXd _x = lu.solve(rhs);
    /*Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> gmres(sparseMatrix);
    Eigen::VectorXd _x = gmres.solve(rhs);
    cout << "Iterations: " << gmres.iterations() << endl;*/

    //PETSC::PGMRES<double> gmres(sparseMatrix);
    //Eigen::VectorXd _x = gmres.Solve(rhs);
    //cout << "Time for solution: " << profiler.Toc() << " s." << endl;

    method.PrintSolution(x, "sol.txt");
    method.PrintSolution(_x, "sol_trunc.txt");

    HaarInverse(x);
    HaarInverse(_x);

    Eigen::VectorXd err = x - _x;
    cout << "Relative error(l2): " << err.norm() / x.norm() << endl;
    cout << "Relative error(c): " << err.lpNorm<Eigen::Infinity>() / x.lpNorm<Eigen::Infinity>() << endl;

    return 0;
}