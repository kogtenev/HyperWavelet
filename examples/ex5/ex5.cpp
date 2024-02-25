#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>

#include "surface.h"
#include "helpers.h"

using namespace std;
using namespace hyper_wavelet;
using namespace hyper_wavelet_2d;

using SparseMatrix = Eigen::SparseMatrix<complex<double>>;

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cout << "Usage: ./ex5 k lambda alpha r mesh_file mesh_graph_file basis_orient (random / polar_min_global)" << std::endl;
        return 0;
    }

    const double k = stod(argv[1]);
    const double lambda = stod(argv[2]);
    const double alpha = stod(argv[3]);
    const double r = stod(argv[4]);
    const string meshFile(argv[5]);
    const string graphFile(argv[6]);
    const string basisOrientation(argv[7]);

    cout << "Wave number: " << k << endl;
    cout << "alpha  = " << alpha << endl;
    cout << "lambda = " << alpha << endl;
    cout << "r = " << r << endl;
    cout << "Mesh file: " << meshFile << endl;
    cout << "Graph file: " << graphFile << endl;
    cout << "Basis orientation: " << basisOrientation << endl;

    SurfaceSolver solver(k, alpha, lambda, r, meshFile, graphFile, basisOrientation);
    Eigen::MatrixXcd fullRhs(solver.GetDimension(), 181);    

    for (int n = 0; n < 181; ++n) {
        const double phi = 2 * M_PI * 2 * n / 360;
        const function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [&k, &phi](const Eigen::Vector3d& r) {
            const complex<double> i = {0., 1.};
            Eigen::Vector3cd _k, E0;
            _k << k * cos(phi), k * sin(phi), 0.;
            E0 << 0, 0, 1.;
            E0 *= exp(i * _k.dot(r.cast<complex<double>>()));
            return E0; 
        };
        solver.FormRhs(f);
        solver.WaveletTransform();
        fullRhs.col(n) = solver.GetRhs();
    }

    solver.FormFullMatrix();
    const auto& A = solver.GetFullMatrix();

    cout << "Applying wavelet transform" << endl;
    solver.WaveletTransform();

    Profiler profiler;
    cout << "Solving full linear system" << endl; 
    Eigen::MatrixXcd x(fullRhs.rows(), fullRhs.cols());
    {
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
        x = lu.solve(fullRhs);
    }
    cout << "Time for solution: " << profiler.Toc() << endl;

    solver.FormMatrixCompressed();
    const auto& truncA = solver.GetTruncatedMatrix();

    Eigen::MatrixXcd& dA = const_cast<Eigen::MatrixXcd&>(A);
    const double normA = dA.norm();
    dA -= truncA;
    const double errNorm = dA.norm();
    cout << "Relative error for matrices: " << errNorm / normA << '\n' << endl;

    cout << "Solving truncated system" << endl;
    profiler.Tic();
    Eigen::SparseLU<Eigen::SparseMatrix<complex<double>>> lu(truncA);
    auto _x = lu.solve(fullRhs);

    cout << "Time for solution: " << profiler.Toc() << endl;
    cout << "\nPrinting solution" << endl;
    cout << "Rel. error: " << (x - _x).norm() / x.norm() << endl;

    solver.PrintEsaInverse(x, "esa.txt");
    solver.PrintEsaInverse(_x, "esa_sparse.txt");
    cout << "Done" << endl;

    return 0;
}
