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
    const string meshFile(argv[3]);

    cout << "Wave number: " << k << endl;
    cout << "threshold: " << threshold << endl;
    cout << "Mesh file: " << meshFile << endl;

    string graphFile;
    if (argc > 3) {
        graphFile = string(argv[4]);
        cout << "Graph file: " << graphFile << endl;
    }
    cout << endl;

    const function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [&k](const Eigen::Vector3d& r) {
        const complex<double> i = {0., 1.};
        Eigen::Vector3cd _k, E0;
        _k << k,  0., 0.;
        E0 << 0., 1., 0.;
        E0 *= exp(i * _k.dot(r.cast<complex<double>>()));
        return E0; 
    };

    SurfaceSolver solver(k, meshFile, graphFile);
    solver.FormFullMatrix();
    solver.FormRhs(f);

    /*{
        cout << "Computing svd" << endl;
        Eigen::BDCSVD<Eigen::MatrixXcd> svd(A);
        const auto& sigma = svd.singularValues();
        cout << "Condition number: " << sigma[0] / sigma[sigma.size()-1] << endl; 
    }*/
    
    cout << "Applying wavelet transform" << endl;
    solver.WaveletTransform();
    solver.PrintFullMatrix("matrix.txt");

    const auto& A = solver.GetFullMatrix();
    const auto& rhs = solver.GetRhs();

    Profiler profiler;
    cout << "Solving full linear system" << endl; 
    Eigen::VectorXcd x;
    {
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
        x = lu.solve(rhs);
    }
    cout << "Time for solution: " << profiler.Toc() << endl;

    solver.FormTruncatedMatrix(threshold);
    const auto& truncA = solver.GetTruncatedMatrix();

    Eigen::MatrixXcd& dA = const_cast<Eigen::MatrixXcd&>(A);
    const double normA = dA.lpNorm<Eigen::Infinity>();
    dA -= truncA;
    const double errNorm = dA.lpNorm<Eigen::Infinity>();
    cout << "Relative error for matrices: " << errNorm / normA << endl << endl;

    cout << "Solving truncated system" << endl;
    profiler.Tic();
    Eigen::SparseLU<Eigen::SparseMatrix<complex<double>>> lu(truncA);
    Eigen::VectorXcd _x = lu.solve(rhs);
    cout << "Time for solution: " << profiler.Toc() << endl;
    cout << "\nPrinting solution" << endl;
    cout << "Rel. error: " << (x - _x).norm() / x.norm() << endl;

    /*solver.WaveletTransformInverse(_x);
    solver.PrintSolutionVtk(_x);
    solver.PrintEsa(_x);*/
    cout << "Done" << endl;

    return 0;
}
