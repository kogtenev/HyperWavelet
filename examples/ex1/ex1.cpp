#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>

#include "colocations.h"

using namespace std;
using namespace hyper_wavelet;

function<double(double)> f = [](double x) {return sin(M_PI * x);};

int main(int argc, char* argv[]) {
    int numLevels = stoi(argv[1]);
    double a = stod(argv[2]);
    double b = stod(argv[3]);

    ColocationsMethod method(numLevels, a, b);
    method.FormFullMatrix();
    method.FormRhs(f);

    const Eigen::MatrixXd& A = method.GetFullMatrix();
    const Eigen::VectorXd& rhs = method.GetRhs();

    cout << "Printing matrix" << endl;
    ofstream fout("mat.txt", ios::out);
    fout << A << endl;
    fout.close();

    cout << "Solving linear system" << endl;
    Eigen::VectorXd x = A.fullPivLu().solve(rhs);

    method.PrintSolution(x);

    return 0;
}