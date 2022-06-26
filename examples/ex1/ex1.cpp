#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "colocations.h"

using namespace std;
using namespace hyper_wavelet;

function<double(double)> f = [](double x) {return 1.;};

int main(int argc, char* argv[]) {
    int numLevels = stoi(argv[1]);

    ColocationMethod method(numLevels);
    method.FormFullMatrix();
    method.FormRhs(f);

    const Eigen::MatrixXd& A = method.GetFullMatrix();
    const Eigen::VectorXd& rhs = method.GetRhs();
    const Eigen::VectorXd& x = A.fullPivLu().solve(rhs);

    ofstream fout("mat.txt", ios::out);
    fout << A << endl;
    fout.close();
    fout.open("rhs.txt", ios::out);
    fout << rhs << endl;

    method.PrintSolution(x);

    return 0;
}