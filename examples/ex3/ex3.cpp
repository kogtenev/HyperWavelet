#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>

#include "surface.h"

using namespace std;
using namespace hyper_wavelet_2d;

function<Eigen::Vector3d(double, double)> surfaceMap = [](double x, double y) {
    Eigen::Vector3d result;
    result[0] = 0.1 * x;
    result[1] = 0.1 * y;
    result[2] = 0.;
    return result;
};

function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [](const Eigen::Vector3d& r) {
    const complex<double> zero = {0., 0.};
    const complex<double> one = {1., 0.};
    const complex<double> i = {0., 1.};
    Eigen::Vector3cd k;
    Eigen::Vector3cd E0;
    k << zero, zero, one;
    E0 << zero, one, zero;
    return exp(i * k.dot(r)) * E0; 
};

void ChecForNanAndInf(const Eigen::MatrixXcd& A) {
    cout << "Checking system matrix\n";
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            if (isnan(A(i, j).real()) or isnan(A(i, j).imag())) {
                throw runtime_error("NAAAAAAAAAAAAAAAAN!");
            }
            if (isinf(A(i, j).real()) or isinf(A(i, j).imag())) {
                throw runtime_error("INF!");
            }
        }
    }
    cout << "No nans and infs\n";
}

int main(int argc, char* argv[]) {
    double k = stod(argv[1]);
    int nx = stoi(argv[2]);
    int ny = stoi(argv[3]);

    cout << "Colocation method for scattering problem on unit rectangle" << endl;
    cout << "k = "  << k << endl;
    cout << "nx = " << nx << endl;
    cout << "ny = " << ny << endl << endl;

    RectangleSurfaceSolver solver(nx, ny, k, surfaceMap);
    solver.FormFullMatrix();
    solver.FormRhs(f);

    cout << "Solving linear system" << endl;
    const Eigen::MatrixXcd A = solver.GetFullMatrix();
    //ChecForNanAndInf(A);
    const Eigen::VectorXcd rhs = solver.GetRhs();
    //cout << rhs << endl;
    ///cout << A << endl;
    const auto Lu = A.lu();
    //cout << "Rank of the matrix: " << Lu.rank() << endl;
    const Eigen::VectorXcd x = Lu.solve(rhs);
    cout << "System is solved" << endl << endl;

    cout << "Printing solution" << endl;
    solver.PlotSolutionMap(x);
    cout << "Done" << endl;
    
    return 0;
}