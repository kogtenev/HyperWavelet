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

const function<Eigen::Vector3d(double, double)> unitMap = [](double x, double y) {
    Eigen::Vector3d result;
    result[0] = x;
    result[1] = y;
    result[2] = 0.;
    return result;
};

const function<Eigen::Vector3d(double, double)> sphereMap = [](double phi, double theta) {
    Eigen::Vector3d result;
    result[0] = sin(M_PI * theta) * cos(2 * M_PI * phi);
    result[1] = sin(M_PI * theta) * sin(2 * M_PI * phi);
    result[2] = cos(M_PI * theta);
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

int main(int argc, char* argv[]) {
    if (argc < 6) {
        cout << "Usage: ./ex3 k nx ny threshold surface_type" << endl;
        return 0;
    }

    const double k = stod(argv[1]);
    const int nx = stoi(argv[2]);
    const int ny = stoi(argv[3]);
    const double threshold = stod(argv[4]);
    const int surfaceType = stoi(argv[5]);

    function<Eigen::Vector3d(double, double)> surfaceMap;
    string surfaceName;

    switch (surfaceType) {
        case 0: 
            surfaceMap = unitMap; 
            surfaceName = "square";
            break;
        case 1: 
            surfaceMap = sphereMap;
            surfaceName = "sphere"; 
            break;
        default:
            cout << "Wrong surface type input!" << endl
                 << "0 - square, 1 - sphere" << endl;
            return -1;
    }

    cout << "Colocation method for scattering problem" << endl;
    cout << "k = "  << k << endl;
    cout << "nx = " << nx << endl;
    cout << "ny = " << ny << endl;
    cout << "threshold = " << threshold << endl;
    cout << "Surface: " << surfaceName << endl << endl;

    RectangleSurfaceSolver solver(nx, ny, k, surfaceMap);
    solver.FormFullMatrix();
    solver.FormRhs(f);
    
    cout << "Applying Haar transformation" << endl;
    solver.HaarTransform();
    solver.PrintFullMatrix("mat.txt");

    const Eigen::MatrixXcd& A = solver.GetFullMatrix();
    const Eigen::VectorXcd& rhs = solver.GetRhs();
    const double normA = A.lpNorm<Eigen::Infinity>();

    /*const auto& eig = A.eigenvalues();
    std::ofstream eigFile("eig.txt", ios::out);
    for (int i = 0; i < eig.size(); i++) {
        eigFile << eig[i].real() << ' ' << eig[i].imag() << '\n';
    }
    eigFile.close();*/

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
    solver.PrintSolutionVtk(x);
    cout << "Done" << endl;
    
    return 0;
}
