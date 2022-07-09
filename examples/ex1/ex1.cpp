#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "colocations.h"
#include "helpers.h"

using namespace std;
using namespace hyper_wavelet;

function<double(double)> f = [](double x) {return sin(M_PI * x);};

int main(int argc, char* argv[]) {
    int numLevels = stoi(argv[1]);
    double a = stod(argv[2]);
    double b = stod(argv[3]);
    double threshold = stod(argv[4]);

    ColocationsMethod method(numLevels, a, b);
    method.FormFullMatrix();
    method.FormRhs(f);

    const Eigen::MatrixXd& A = method.GetFullMatrix();
    const Eigen::VectorXd& rhs = method.GetRhs();

    cout << "\nAnilyzing system matrix" << endl;
    PrintSparsityTable(A); 

    cout << "Printing matrix" << endl;
    ofstream fout("mat.txt", ios::out);
    fout << A << endl;
    fout.close();

    cout << "Solving full linear system" << endl;
    Profiler profiler;
    Eigen::VectorXd x = A.lu().solve(rhs);
    cout << "Time for solution: " << profiler.Toc() << " s." << endl;
    cout << "\nAnilyzing solution vector" << endl;
    PrintSparsityTable(x);

    method.FormTruncatedMatrix(threshold);
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