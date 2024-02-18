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
using SparseMatrix = Eigen::SparseMatrix<complex<double>>;

int main(int argc, char* argv[]) {
    const double k = stod(argv[1]);
    const double threshold = stod(argv[2]);
    const double reg = stod(argv[3]);
    const string meshFile(argv[4]);

    cout << "Wave number: " << k << endl;
    cout << "threshold: " << threshold << endl;
    cout << "Mesh file: " << meshFile << endl;

    string graphFile;
    if (argc > 4) {
        graphFile = string(argv[5]);
        cout << "Graph file: " << graphFile << endl;
    }
    cout << endl;

    SurfaceSolver solver(k, meshFile, graphFile);
    Eigen::MatrixXcd fullRhs(solver.GetDimension(), 181);    

    for (int n = 0; n < 181; ++n) {
        const double phi = 2 * M_PI * 2 * n / 360;
        const function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [&k, &phi](const Eigen::Vector3d& r) {
            const complex<double> i = {0., 1.};
            Eigen::Vector3cd _k, E0;
            _k << k * cos(phi), k * sin(phi), 0.;
            E0 << -sin(phi), cos(phi), 1.;
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

    solver.FormMatrixCompressed(threshold, reg);
    const auto& truncA = solver.GetTruncatedMatrix();

    Eigen::MatrixXcd& dA = const_cast<Eigen::MatrixXcd&>(A);
    const double normA = dA.norm();
    dA -= truncA;
    const double errNorm = dA.norm();
    cout << "Relative error for matrices: " << errNorm / normA << endl << endl;

    cout << "Solving truncated system" << endl;
    profiler.Tic();
    Eigen::SparseLU<Eigen::SparseMatrix<complex<double>>> lu(truncA);
    auto _x = lu.solve(fullRhs);
    cout << "Time for solution: " << profiler.Toc() << endl;
    cout << "\nPrinting solution" << endl;
    cout << "Rel. error: " << (x - _x).norm() / x.norm() << endl;

    /*solver.WaveletTransformInverse(_x);
    solver.WaveletTransformInverse(x);
    solver.EstimateErrors(x, _x);
    solver.PrintSolutionVtk(_x);
    solver.PrintEsa(x, "esa.txt");
    solver.PrintEsa(_x, "esa_sparse.txt");*/

    solver.PrintEsaInverse(x, "esa.txt");
    solver.PrintEsaInverse(_x, "esa_sparse.txt");
    cout << "Done" << endl;

    return 0;
}
