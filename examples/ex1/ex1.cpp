#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

#ifdef USE_MKL_PARDISO
    #define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "colocations.h"
#include "helpers.h"

using namespace std;
using namespace hyper_wavelet;

function<double(double)> f = [](double x) {return sin(M_PI * x);};

function<double(double, double)> K = [](double x, double y) {
    //return x * x + y * y + sin(M_PI * x * y);
    return 0.; 
};

int main(int argc, char* argv[]) {
    int numLevels = stoi(argv[1]);
    double a = stod(argv[2]);
    double b = stod(argv[3]);
    double threshold = stod(argv[4]);
    double alpha = stod(argv[5]);
    bool printFull = static_cast<bool>(stoi(argv[6]));
    bool printTrunctated = static_cast<bool>(stoi(argv[7]));

    ColocationsMethod method(numLevels, a, b);
    method.FormFullMatrix(K);
    method.FormRhs(f);

    const Eigen::MatrixXd& A = method.GetFullMatrix();
    const Eigen::VectorXd& rhs = method.GetRhs();

    cout << "\nAnilyzing system matrix" << endl;
    PrintSparsityTable(A);
    {
        cout << "Computing SVD" << endl;
        Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU);
        const auto& sigma = svd.singularValues();
        const Eigen::MatrixXd& U = svd.matrixU();
        size_t dim = sigma.size();
        cout << "Maximal singular value: " << sigma(0) << endl;
        cout << "Minimal singular value: " << sigma(dim - 1) << endl;
        cout << "Condition number: " << sigma(0) / sigma(dim - 1) << endl;

        alpha = sqrt(sigma(dim - 1));

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

    method.FormTruncatedMatrix(K, threshold, alpha, printTrunctated);
    const auto& sparseMatrix = method.GetTruncatedMatrix();

    cout << "Solving system with truncated matrix" << endl;
    profiler.Tic();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(sparseMatrix);
    Eigen::VectorXd _x = lu.solve(rhs);
    cout << "Time for solution: " << profiler.Toc() << " s." << endl;
    Eigen::VectorXd err = x - _x;

    cout << "Relative error: " << err.norm() / x.norm() << endl;

    method.PrintSolution(x, "sol.txt");
    method.PrintSolution(_x, "sol_trunc.txt");

    return 0;
}