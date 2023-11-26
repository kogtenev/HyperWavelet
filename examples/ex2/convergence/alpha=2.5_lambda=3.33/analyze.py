import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def analyze_convergence(prefix):
    exact_data = np.loadtxt('sol_n=16.txt')
    err_c  = np.inf
    err_l2 = np.inf
    max_dim = 16384
    duplicats = 256
    dim = 256
    print('N err(c) order(c) err(l2) order(l2)')
    for n in range(8, 14, 1):
        sol = np.loadtxt(prefix + str(n) + '.txt')
        exact_data_reshaped = exact_data.reshape((-1,duplicats), order='C')
        exact_sol = exact_data_reshaped[:, duplicats-1]
        new_err_c = np.max(np.abs(sol - exact_sol)) / np.max(np.abs(exact_sol))
        new_err_l2 = np.linalg.norm(sol - exact_sol) / np.linalg.norm(sol)
        duplicats = int(duplicats / 2)
        order_c = np.log2(err_c / new_err_c)
        order_l2 = np.log2(err_l2 / new_err_l2)
        err_c = new_err_c
        err_l2 = new_err_l2
        print(dim, round(err_c, 5), round(order_c, 5), round(err_l2, 5), round(order_l2, 5))
        dim = int(dim * 2)

def analyze_nonzeros():
    print('N  nnz  nnz(prop)  order')
    data = np.loadtxt('nnz.txt')
    nnz = 10000000000000
    for i in range(6):
        dim = data[i, 2]
        nnz_proportion = data[i, 3]
        new_nnz = dim * dim * nnz_proportion
        order = np.log2(new_nnz / nnz)
        nnz = new_nnz
        print(int(dim), int(nnz), nnz_proportion, round(order, 3))

def plot_solutions():
    exact_data = np.loadtxt('sol_n=16.txt')
    dense_data = np.loadtxt('sol_n=9.txt')
    sparse_data = np.loadtxt('sol_trunc_n=9.txt')
    exact_sol = np.hstack((np.zeros(1), exact_data[127::128], np.zeros(1)))
    dense_sol = np.hstack((np.zeros(1), dense_data, np.zeros(1)))
    sparse_sol = np.hstack((np.zeros(1), sparse_data, np.zeros(1)))
    dx = 1 / 513
    x = np.arange(dx/2, 1 - dx/2, dx)
    x = x * np.pi
    x = np.hstack(([0], x, [np.pi]))
    plt.plot(x, exact_sol)
    plt.plot(x, dense_sol)
    plt.plot(x, sparse_sol)
    plt.xlabel(r'$s_0$')
    plt.ylabel(r'$g(s_0)$')
    plt.legend(['a', 'b', 'c'])
    plt.grid()
    plt.savefig('solutions.png', dpi=199)
    
def plot_residuals():
    residuals = []
    data = open('gmres_convergence.txt', 'r')
    for line in data:
        _, res = line.split(sep='residual:')
        residuals.append(float(res))
    fig, ax = plt.subplots()
    ax.clear()
    plt.plot(residuals)
    plt.xlabel('Число итераций')
    plt.ylabel('Отн. невязка')
    plt.yscale('log')
    plt.xticks(range(len(residuals)))
    plt.grid()
    plt.savefig('residuals.png', dpi=199)


print('Analyze convergence with dense matrix')
analyze_convergence('sol_n=')

print('\nAnalyze convergence with sparse matrix')
analyze_convergence('sol_trunc_n=')

print('\nAnalyze nonzeros')
analyze_nonzeros()

plot_solutions()

plot_residuals()
