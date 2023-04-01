#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "petscksp.h" 

namespace PETSC {

#define EigenVector Eigen::Matrix<DataType, Eigen::Dynamic, 1>

template<typename DataType> 
EigenVector PetscVectorToEigen(Vec b, PetscInt dim);

template<>
Eigen::VectorXd PetscVectorToEigen<double>(Vec b, PetscInt dim) {
    Eigen::VectorXd result(dim);
    for (PetscInt i = 0; i < dim; ++i) {
        PetscScalar value;
        VecGetValues(b, 1, &i, &value);
        result[i] = std::real(value);
    }
    return result;
}

template<>
Eigen::VectorXcd PetscVectorToEigen<std::complex<double>>(Vec b, PetscInt dim) {
    Eigen::VectorXcd result(dim);
    for (PetscInt i = 0; i < dim; ++i) {
        PetscScalar value;
        VecGetValues(b, 1, &i, &value);
        result[i] = value;
    }
    return result;
}

template<typename DataType>
class PGMRES {
private:
    const PetscInt nnz, dim;
    PetscInt* nnz_per_row;

    Vec x, b, bcopy, diag;
    Mat mat, pmat;

    KSP solver;
    PC prec;

    KSPConvergedReason reason;
    PetscReal rtol = 1e-6, atol = 1e-6;
    PetscInt maxits = 500;

public:
    PGMRES(const Eigen::SparseMatrix<DataType>& A): nnz(A.nonZeros()), dim(A.rows()) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("Non-square matrix in GMRES!");
        }

        nnz_per_row = new PetscInt[dim];

        const auto* innerIndexPtr = A.innerIndexPtr();
        const auto* outerIndexPtr = A.outerIndexPtr();
        const auto* valuePtr = A.valuePtr();

        for (auto i = 0; i < dim; ++i) {
            nnz_per_row[i] = outerIndexPtr[i+1] - outerIndexPtr[i];
        }

        MatCreateSeqAIJ(PETSC_COMM_SELF, dim, dim, PETSC_DEFAULT, nnz_per_row, &mat);

        for (auto i = 0; i < dim; ++i) {
            for (auto j = outerIndexPtr[i]; j < outerIndexPtr[i+1]; ++j) {
                PetscInt idx = i;
                PetscInt idy = innerIndexPtr[j];
                PetscScalar value = valuePtr[j];
                MatSetValues(mat, 1, &idx, 1, &idy, &value, INSERT_VALUES);
            }
        }

        MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

        KSPCreate(PETSC_COMM_SELF, &solver);
        KSPSetOperators(solver, mat, mat);
        KSPSetType(solver, KSPGMRES);
        KSPSetInitialGuessNonzero(solver, PETSC_FALSE);
        KSPGMRESSetRestart(solver, 60);

        KSPSetNormType(solver, KSP_NORM_UNPRECONDITIONED);
        KSPSetTolerances(solver, atol, rtol, PETSC_DEFAULT, maxits);

        KSPGetPC(solver, &prec);
        PCSetType(prec, PCILU);
        //PCFactorSetDropTolerance(prec, 1e-5, 0.01, 800);
        //PCFactorSetFill(prec, 10.);
        PCFactorSetLevels(prec, 2);
        PCFactorSetMatOrderingType(prec, MATORDERINGQMD);


        KSPSetFromOptions(solver);
        KSPSetUp(solver);

        PCFactorGetMatrix(prec, &pmat);
    }

    EigenVector Solve(const EigenVector& rhs) {
        if (rhs.size() != dim) {
            throw std::invalid_argument("Wrong input rhs size!");
        }

        VecCreateSeq(PETSC_COMM_SELF, dim, &b);
        VecDuplicate(b, &x);
        VecDuplicate(b, &bcopy);

        for (PetscInt i = 0; i < dim; ++i) {
            PetscScalar value = rhs[i];
            VecSetValues(b, 1, &i, &value, INSERT_VALUES);
        }
        VecCopy(b, bcopy);

        VecAssemblyBegin(b);
        VecAssemblyEnd(b);

        VecDuplicate(b, &diag);
        MatGetDiagonal(pmat, diag);

        KSPSolve(solver, b, x);

        KSPGetConvergedReason(solver, &reason);
        if (reason == KSP_CONVERGED_RTOL_NORMAL) {
            std::cout << "Converged by rel.tolerance" << std::endl;
        } else {
            std::cout << "Converged by unknown reason" << std::endl;
        }

        PetscInt iterations;
        PetscReal residual;
        PetscReal rhsnorm;

        MatInfo info;

        KSPGetIterationNumber(solver, &iterations);
        KSPGetResidualNorm(solver, &residual);
        MatGetInfo(pmat, MAT_LOCAL, &info);
        VecNorm(bcopy, NORM_2, &rhsnorm);

        std::cout << "Fill ratio for ilu: " << info.fill_ratio_needed << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Relative residual: " << residual / rhsnorm << std::endl << std::endl;

        return PetscVectorToEigen<DataType>(x, dim);
    }

    ~PGMRES() {
        VecDestroy(&x);
        VecDestroy(&b);
        VecDestroy(&bcopy);
        VecDestroy(&diag);
        MatDestroy(&mat);
        KSPDestroy(&solver);
        delete[] nnz_per_row;
        PetscFinalize();
    }
};

}