#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <filesystem>

#include "surface.h"
#include "helpers.h"

using namespace std;
using namespace std::complex_literals;
using namespace hyper_wavelet;
using namespace hyper_wavelet_2d;

class AttemptCounter {
public:
    AttemptCounter() {
        const string fname = "attempt.txt";
        if (filesystem::exists(fname)) {
            ifstream fin(fname, ios::in);
            int attempt = -1;
            fin >> attempt;
            if (attempt >= 0) {
                _attempt = attempt;
            }
        }
        ofstream fout(fname, ios::out);
        fout << _attempt + 1 << '\n';
    }
    int Attempt() const { return _attempt; }
private:
    int _attempt = 0;
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: ./ex5 k lambda alpha r" << std::endl;
        return 0;
    }

    const double k = stod(argv[1]);
    const double lambda = stod(argv[2]);
    const double alpha = stod(argv[3]);
    const double r = stod(argv[4]);

    cout << "Wave number: " << k << endl;
    cout << "alpha  = " << alpha << endl;
    cout << "lambda = " << lambda << endl;
    cout << "r = " << r << endl;

    SurfaceSolver solver(k, alpha, lambda, r);
    Eigen::MatrixXcd fullRhs(solver.GetDimension(), 181);    

    for (int n = 0; n < 181; ++n) {
        const double phi = 2 * M_PI * 2 * n / 360;
        const function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [&k, &phi](const Eigen::Vector3d& r) {
            Eigen::Vector3cd _k, E0;
            _k << k * cos(phi), k * sin(phi), 0.;
            E0 << 0, 0, 1.;
            E0 *= exp(1i * _k.dot(r.cast<complex<double>>()));
            return E0; 
        };
        solver.FormRhs(f);
        const auto& rhs = solver.GetRhs();
        solver.WaveletTransform();
        fullRhs.col(n) = solver.GetRhs();
    }

    solver.FormMatrixCompressed();
    const auto& truncA = solver.GetTruncatedMatrix();

    AttemptCounter counter;
    Profiler profiler;
    
    Eigen::MatrixXcd _x(fullRhs.rows(), fullRhs.cols());
    {   
        cout << "Solving truncated system" << endl;
        Eigen::SparseLU<Eigen::SparseMatrix<complex<double>>> lu(truncA);
        _x = lu.solve(fullRhs);
        cout << "Time for solution: " << profiler.Toc() << '\n' << endl;
    }
    solver.PrintEsaInverse(_x, "esa_sparse", counter.Attempt());

    solver.FormFullMatrix();
    const auto& A = solver.GetFullMatrix();

    cout << "Applying wavelet transform" << endl;
    solver.WaveletTransform();

    cout << "Solving full linear system" << endl; 
    profiler.Tic();
    Eigen::MatrixXcd x(fullRhs.rows(), fullRhs.cols());
    {
        #ifdef USE_MKL_PARDISO
            Eigen::PardisoLU<Eigen::MatrixXcd> lu(A);
            x = lu.solve(fullRhs);
        #else
            Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
            x = lu.solve(fullRhs);
        #endif
    }
    cout << "Time for solution: " << profiler.Toc() << endl;

    Eigen::MatrixXcd& dA = const_cast<Eigen::MatrixXcd&>(A);
    const double normA = dA.norm();
    dA -= truncA;
    const double errNorm = dA.norm();
    cout << "Relative error for matrices: " << errNorm / normA << '\n' << endl;

    cout << "Rel. error: " << (x - _x).norm() / x.norm() << endl;

    solver.PrintEsaInverse(x, "esa", counter.Attempt());
    cout << "Done" << endl;

    return 0;
}
