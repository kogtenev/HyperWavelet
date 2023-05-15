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

const function<Eigen::Vector3d(double, double)> cylinderMap = [](double phi, double l) {
    Eigen::Vector3d result;
    result[0] = -0.1 + 0.2*l;
    result[1] =  0.1 * cos(2 * M_PI * phi);
    result[1] =  0.1 * sin(2 * M_PI * phi);
    return result;
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
            surfaceMap = cylinderMap;
            surfaceName = "cylinder"; 
            break;
        default:
            cout << "Wrong surface type input!" << endl
                 << "0 - square, 1 - cylinder" << endl;
            return -1;
    }

    cout << "Colocation method for scattering problem" << endl;
    cout << "k = "  << k << endl;
    cout << "nx = " << nx << endl;
    cout << "ny = " << ny << endl;
    cout << "threshold = " << threshold << endl;
    cout << "Surface: " << surfaceName << endl << endl;

    RectangleSurfaceSolver solver(nx, ny, k, surfaceMap);
    Eigen::MatrixXcd fullRhs(solver.GetDimension(), 181);    

    for (int n = 0; n < 181; ++n) {
        const double phi = 2 * M_PI * 2 * n / 360;
        const function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [&k, &phi](const Eigen::Vector3d& r) {
            const complex<double> i = {0., 1.};
            Eigen::Vector3cd _k, E0;
            _k << 0., k * sin(phi), k * cos(phi);
            E0 << 0., cos(phi), -sin(phi);
            E0 *= exp(i * _k.dot(r.cast<complex<double>>()));
            return E0; 
        };
        solver.FormRhs(f);
        solver.HaarTransform();
        fullRhs.col(n) = solver.GetRhs();
    }

    solver.FormFullMatrix();
    
    cout << "Applying Haar transformation" << endl;
    solver.HaarTransform();
    const auto& A = solver.GetFullMatrix();
    //solver.PrintFullMatrix("mat.txt");

    /*const auto& eig = A.eigenvalues();
    std::ofstream eigFile("eig.txt", ios::out);
    for (int i = 0; i < eig.size(); i++) {
        eigFile << eig[i].real() << ' ' << eig[i].imag() << '\n';
    }
    eigFile.close();*/

    cout << "Analyzing matrix" << endl;
    //PrintSparsityTable(A);

    Profiler profiler;
    cout << "Solving linear system" << endl;
    //Eigen::VectorXcd x = A.lu().solve(rhs);
    Eigen::MatrixXcd x;
    {
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
        x = lu.solve(fullRhs);
    }
    const double normA = A.lpNorm<Eigen::Infinity>();
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
    //Eigen::VectorXcd _x = lu.solve(rhs);
    Eigen::MatrixXcd _x(fullRhs.rows(), fullRhs.cols());
    for (int i = 0; i < _x.cols(); ++i) {
        _x.col(i) = lu.solve(fullRhs.col(i));
    }
    cout << "Time for solution: " << profiler.Toc() << " s." << endl;
    cout << "Relative error: " << (_x - x).norm() / x.norm() << endl << endl;

    //solver.PlotSolutionMap(_x);
    //solver.PrintSolutionVtk(_x);
    solver.PrintEsaInverse(x);
    cout << "Done" << endl;
    
    return 0;
}
