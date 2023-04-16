#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>

#include <unsupported/Eigen/IterativeSolvers>
#include "petscsys.h"  

#include "surface.h"
#include "helpers.h"
#include "petsc.h"

using namespace std;
using namespace hyper_wavelet;
using namespace hyper_wavelet_2d;

using hyper_wavelet::Profiler;
using SparseMatrix = Eigen::SparseMatrix<complex<double>>;

function<Eigen::Vector3d(double, double)> surfaceMap = [](double x, double y) {
    Eigen::Vector3d result;
    result[0] = x;
    result[1] = y;
    result[2] = 0.;
    return result;
};

static char petsc_magic[] = "Appends to an ASCII file.\n\n";

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, (char*)0, petsc_magic);

    const double k = stod(argv[1]);
    const int nx = stoi(argv[2]);
    const int ny = stoi(argv[3]);
    const double threshold = stod(argv[4]);

    cout << "Colocation method for scattering problem on unit rectangle" << endl;
    cout << "k = "  << k << endl;
    cout << "nx = " << nx << endl;
    cout << "ny = " << ny << endl << endl;

    const function<Eigen::Vector3cd(const Eigen::Vector3d&)> f = [&k](const Eigen::Vector3d& r) {
        const complex<double> i = {0., 1.};
        Eigen::Vector3cd _k, E0;
        _k << 0., 0., k;
        E0 << 0., 1., 0.;
        E0 *= exp(i * _k.dot(r.cast<complex<double>>()));
        return E0; 
    };

    RectangleSurfaceSolver solver(nx, ny, k, surfaceMap);
    solver.FormMatrixCompressed(threshold);
    solver.FormRhs(f);
    
    cout << "Applying Haar transformation for rhs" << endl;
    solver.HaarTransform();
    
    const auto& truncA = solver.GetTruncatedMatrix();
    const auto& rhs = solver.GetRhs();

    cout << "Solving truncated system" << endl;
    Profiler profiler;
    //Eigen::SparseLU<Eigen::SparseMatrix<complex<double>>> lu(truncA);
    //Eigen::VectorXcd x = lu.solve(rhs);
    //Eigen::GMRES<SparseMatrix, Eigen::IncompleteLUT<complex<double>>> gmres;
    //gmres.preconditioner().setFillfactor(3);
    //gmres.preconditioner().setDroptol(1e-8);
    /*gmres.setTolerance(1e-8);
    gmres.set_restart(80);
    gmres.setMaxIterations(500);*/
    //gmres.compute(truncA);
    PETSC::PGMRES gmres(truncA);
    Eigen::VectorXcd x = gmres.Solve(rhs);
    Eigen::VectorXcd res = (truncA * x - rhs);
    //cout << "iterations: " << gmres.iterations() << endl;
    cout << "Relative residual: " << res.norm() / rhs.norm() << endl;
    cout << "Time for solution: " << profiler.Toc() << " s." << endl;

    //solver.PlotSolutionMap(x);
    solver.PrintSolutionVtk(x);
    cout << "Done" << endl;
    
    //PetscFinalize();

    return 0;
}