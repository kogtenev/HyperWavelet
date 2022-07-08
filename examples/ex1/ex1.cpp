#include <iostream>
#include <fstream>
#include <cmath>
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
    Eigen::VectorXd x = A.fullPivLu().solve(rhs);
    cout << "\nAnilyzing solution vector" << endl;
    PrintSparsityTable(x);

    method.FormTruncatedMatrix(threshold);
    const auto& sparseMatrix = method.GetTruncatedMatrix();

    cout << "Solving system with truncated matrix" << endl;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(sparseMatrix);
    Eigen::VectorXd _x = lu.solve(rhs);
    Eigen::VectorXd err = x - _x;

    cout << "System with truncated matrix is solved" << endl;
    cout << "Relative error: " << err.norm() / x.norm() << endl;

    method.PrintSolution(x);

    return 0;
}