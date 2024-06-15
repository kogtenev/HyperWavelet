import numpy as np
from numpy import linalg
import pymesh
import matplotlib.pyplot as plt

dense_file  = 'esa.txt'
sparse_file = 'esa_sparse.txt'

true_esa = np.loadtxt(dense_file)
alpha = np.linspace(0, 360, 181)
legend_members = ['Плотн. матр.']
plt.plot(alpha, true_esa)

true_esa_sparse = np.loadtxt(sparse_file)
err = np.abs(true_esa - true_esa_sparse)
print('max. error: ', np.max(err))
print('av. error: ', np.average(err), '\n')
plt.plot(alpha, true_esa_sparse)
legend_members.append('Разреж. матр.')
#plt.ylim(-46, -16)
    

plt.legend(legend_members)
plt.xlabel(r'$\varphi$, градусы')
plt.ylabel(r'$\sigma$, dB')
plt.grid()
plt.show()