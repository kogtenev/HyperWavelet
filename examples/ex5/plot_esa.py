import sys
import numpy as np
from numpy import linalg
import pymesh
import matplotlib.pyplot as plt

attempt = str(sys.argv[1])

dense_file  = 'esa' + attempt + '.txt'
sparse_file = 'esa_sparse' + attempt + '.txt'

data = np.loadtxt(dense_file)

N = int(data[0])
k = data[1]
nnz = int(100 * data[2])

true_esa = data[6:]
alpha = np.linspace(0, 360, 181)
legend_members = ['Плотн. матр.']
plt.plot(alpha, true_esa)

data = np.loadtxt(sparse_file)
true_esa_sparse = data[6:]
err = np.abs(true_esa - true_esa_sparse)
print('max. error: ', np.max(err))
print('av. error: ', np.average(err), '\n')
plt.plot(alpha, true_esa_sparse)
legend_members = ['Плотн. матр.', 'Разреж. матр']
#plt.ylim(-46, -16)
    

plt.legend(legend_members)
plt.title('N = ' + str(N) + ', k = ' + str(k) + ', ' + str(nnz) + '%' )
plt.xlabel(r'$\varphi$, градусы')
plt.ylabel(r'$\sigma$, dB')
plt.grid()
plt.savefig('esa' + attempt + '.png')