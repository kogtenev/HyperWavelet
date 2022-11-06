#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>

#include "surface.h"
#include "helpers.h"

using namespace std;
using namespace hyper_wavelet;
using namespace hyper_wavelet_2d;

using hyper_wavelet::Profiler;

function<Eigen::Vector3d(double, double)> surfaceMap = [](double x, double y) {
    Eigen::Vector3d result;
    result[0] = 0.01 * x;
    result[1] = 0.01 * y;
    result[2] = 0.;
    return result;
};

const function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [](const Eigen::Vector3d& r) {
    const complex<double> zero = {0., 0.};
    const complex<double> one = {1., 0.};
    const complex<double> i = {0., 1.};

    Eigen::Vector3cd k;
    Eigen::Vector3cd E0;

    k[0] = zero;
    k[1] = zero;
    k[2] = one;

    E0[0] = zero;
    E0[1] = one;
    E0[2] = zero;

    E0 *= exp(i * k.dot(r.cast<complex<double>>()));

    return E0; 
};

void CheckForNanAndInf(const Eigen::MatrixXcd& A) {
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
    const double k = stod(argv[1]);
    const int nx = stoi(argv[2]);
    const int ny = stoi(argv[3]);
    const double threshold = stod(argv[4]);

    cout << "Colocation method for scattering problem on unit rectangle" << endl;
    cout << "k = "  << k << endl;
    cout << "nx = " << nx << endl;
    cout << "ny = " << ny << endl << endl;

    RectangleSurfaceSolver solver(nx, ny, k, surfaceMap);
    solver.FormFullMatrix();
    solver.FormRhs(f);
    
    cout << "Applying Haar transformation" << endl;
    solver.HaarTransform();
    solver.PrintFullMatrix("mat.txt");

    const Eigen::MatrixXcd& A = solver.GetFullMatrix();
    const Eigen::VectorXcd& rhs = solver.GetRhs();
    const double normA = A.lpNorm<Eigen::Infinity>();

    cout << "Analyzing matrix" << endl;
    PrintSparsityTable(A);

    Profiler profiler;
    cout << "Solving linear system" << endl;
    Eigen::VectorXcd x = A.lu().solve(rhs);
    cout << "System is solved" << endl;
    cout << "Time for solution: " << profiler.Toc() << endl << endl;

    solver.FormMatrixCompressed(threshold);
    const auto& truncA = solver.GetTruncatedMatrix();
    Eigen::MatrixXcd& dA = const_cast<Eigen::MatrixXcd&>(A);
    dA -= truncA;
    const double errNorm = dA.lpNorm<Eigen::Infinity>();
    cout << "Relative error for matrices: " << errNorm / normA << endl << endl; 

    cout << "Solving truncated system" << endl;
    profiler.Tic();
    Eigen::SparseLU<Eigen::SparseMatrix<complex<double>>> lu(truncA);
    Eigen::VectorXcd _x = lu.solve(rhs);
    cout << "Time for solution: " << profiler.Toc() << " s." << endl;
    cout << "Relative error: " << (_x - x).norm() / x.norm() << endl << endl;

    solver.PlotSolutionMap(_x);
    cout << "Done" << endl;
    
    return 0;
}
