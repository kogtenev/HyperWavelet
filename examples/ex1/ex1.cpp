#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "colocations.h"

using namespace std;
using namespace hyper_wavelet;

function<double(double)> f = [](double x) {return 1.;};

int main(int argc, char* argv[]) {
    int numLevels = stoi(argv[1]);
    double a = stod(argv[2]);
    double b = stod(argv[3]);

    ColocationMethod method(numLevels, a, b);
    method.FormFullMatrix();
    method.FormRhs(f);

    const Eigen::MatrixXd& A = method.GetFullMatrix();
    const Eigen::VectorXd& rhs = method.GetRhs();

    ofstream fout("mat.txt", ios::out);
    fout << A << endl;
    fout.close();

    Eigen::VectorXd x = A.fullPivLu().solve(rhs);
    fout.open("x.txt", ios::out);
    fout << x << endl;
    fout.close();

    method.PrintSolution(x);

    return 0;
}