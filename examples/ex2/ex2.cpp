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

        _supports.reserve(_dim);
        _supports.push_back({_a, _b});
        _supports.push_back({_a, _b});

        int numOfSupports = 1;
        double step = (_b - _a);

        for (int level = 2; level <= numLevels; level++) {
            numOfSupports *= 2;
            step /= 2;
            for (int i = 0; i < numOfSupports; i++) {
                _supports.push_back({_a + step * i, _a + step * (i + 1)});
            }
        }

        _t0.resize(_dim + 1);
        _t.resize(_dim);
        const double h = (_b - _a) / _dim;
        for (int i = 0; i < _dim + 1; i++) {
            _t[i] = _a + i * h;
        }
        for (int i = 0; i < _dim; i++) {
            _t0[i] = (_t[i] + _t[i+1]) / 2;
        }
    }

    void FormFullMatrix(const std::function<double(double, double)>& K) {
        cout << "Forming dense system matrix" << endl;
        _mat.resize(_dim, _dim);
        const double h = (_b - _a) / _dim;
        for (int i = 0; i < _dim; i++) {
            for (int j = 0; j < _dim; j++) {
                _mat(i, j) = 1.0 / (_t0[i] - _t[j]) - 1.0 / (_t0[i] - _t[j+1]) + K(_t0[i], _t[j]) * h;
            }
        }
        for (int j = 0; j < _dim; j++) {
            auto col = _mat.col(j);
            Haar(col);
        }
        for (int i = 0; i < _dim; i++) {
            auto row = _mat.row(i);
            Haar(row);
        }
        _fullMatrixIsFormed = true;
    }

    void FormRhs(const std::function<double(double)>& f) {
        _rhs.resize(_dim);
        for (int i = 0; i < _dim; i++) {
            _rhs(i) = f(_t0[i]);
        }
        Haar(_rhs);
    }

    void FormTruncatedMatrix(double threshold, double reg, bool printMatrix) {
        cout << "Forming truncated system matrix" << endl;

        _truncMat.resize(_dim, _dim);
        std::vector<Eigen::Triplet<double>> triplets;
        threshold *= (_b - _a);

        for (int i = 0; i < _dim; i++) {
            for (int j = 0; j < _dim; j++) {
                if (Distance(_supports[i], _supports[j]) <= threshold) {
                    if (i == j) {
                        triplets.push_back({i, j, _mat(i, j) + reg});
                    } else {
                        triplets.push_back({i, j, _mat(i, j)});
                    }
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
};

function<double(double)> f = [](double x) {return sin(10 * M_PI * x);}; 

function<double(double, double)> K = [](double x, double y) {
    return sqrt(1. - x*x) + sqrt(1. - y*y);
    //return 0.; 
};

static char petsc_magic[] = "Appends to an ASCII file.\n\n";

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, (char*)0, petsc_magic);

    int numLevels = stoi(argv[1]);
    double a = stod(argv[2]);
    double b = stod(argv[3]);
    double threshold = stod(argv[4]);
    double reg = stod(argv[5]);
    bool printFull = static_cast<bool>(stoi(argv[6]));
    bool printTrunctated = static_cast<bool>(stoi(argv[7]));

    ColocationsMethodHaar method(numLevels, a, b);
    method.FormFullMatrix(K);
    method.FormRhs(f);

    const Eigen::MatrixXd& A = method.GetFullMatrix();
    const Eigen::VectorXd& rhs = method.GetRhs();

    cout << "\nAnilyzing system matrix" << endl;
    double normA;
    PrintSparsityTable(A);
    {
        cout << "Computing SVD" << endl;
        Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU);
        const auto& sigma = svd.singularValues();
        reg *= sigma(0);
        normA = sigma(0);
        const Eigen::MatrixXd& U = svd.matrixU();
        size_t dim = sigma.size();
        cout << "Maximal singular value: " << sigma(0) << endl;
        cout << "Minimal singular value: " << sigma(dim - 1) << endl;
        cout << "Condition number: " << sigma(0) / sigma(dim - 1) << endl;

        cout << "\nAnilysing rhs in singular basis" << endl;
        const Eigen::VectorXd& rhsSingular = U.adjoint() * rhs;
        PrintSparsityTable(rhsSingular);
        ofstream fout("rhs_sing.txt", ios::out);
        fout << rhsSingular;
        fout.close();
    } 

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
    cout << "\nAnilyzing solution vector" << endl;
    PrintSparsityTable(x);

    method.FormTruncatedMatrix(threshold, reg, printTrunctated);
    const auto& sparseMatrix = method.GetTruncatedMatrix();

    /*Eigen::MatrixXd errA = A;
    errA -= sparseMatrix;
    Eigen::BDCSVD<Eigen::MatrixXd> svd(errA);
    const auto& sigma = svd.singularValues();
    cout << "Relative error for sparse matrix: " << sigma(0) / normA << endl << endl;*/

    cout << "Solving system with truncated matrix" << endl;
    profiler.Tic();
    //Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(sparseMatrix);
    //Eigen::VectorXd _x = lu.solve(rhs);
    /*Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> gmres(sparseMatrix);
    Eigen::VectorXd _x = gmres.solve(rhs);
    cout << "Iterations: " << gmres.iterations() << endl;*/

    PETSC::PGMRES<double> gmres(sparseMatrix);
    Eigen::VectorXd _x = gmres.Solve(rhs);
    cout << "Time for solution: " << profiler.Toc() << " s." << endl;

    Eigen::VectorXd err = x - _x;
    cout << "Relative error: " << err.norm() / x.norm() << endl;

    method.PrintSolution(x, "sol.txt");
    method.PrintSolution(_x, "sol_trunc.txt");

    return 0;
}