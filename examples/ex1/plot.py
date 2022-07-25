import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as scp
from matplotlib.colors import LogNorm

file = open('sol.txt', 'r')

a = -1.
b = 1.

data = np.loadtxt(file)

file = open('sol_trunc.txt', 'r')
data_trunc = np.loadtxt(file)

dim = len(data)
step = (b - a) / dim
x = np.arange(a, b, step)

fig, ax = plt.subplots()
ax.plot(x, data)
ax.plot(x, data_trunc)
ax.legend(['Full system', 'Truncated system'], loc = 'upper left')
plt.savefig('solution.png')

#A = np.loadtxt('mat.txt')
#A = np.abs(A)
#a_max = np.max(A)
#a_min = np.min(A)
#A /= a_max
#a_min /= a_max
#p = ax.imshow(A, cmap = 'plasma', norm = LogNorm(vmin=a_min, vmax=1.))
#fig.colorbar(p)
#ax.legend([],[], frameon=False)
#plt.savefig('matrix.png', dpi = 199)

#sparse_data = np.loadtxt('trunc_mat.txt')
#i = sparse_data[:, 0].astype(int)
#j = sparse_data[:, 1].astype(int)
#vals = sparse_data[:, 2]
#A = scp.coo_matrix((vals, (i, j)))
#ax.clear()
#ax.legend([],[], frameon=False)
#fig.clear()
#plt.spy(A, markersize = 1)
#plt.savefig('truncated_matrix.png', dpi = 199)

fig, ax = plt.subplots()
file = open('rhs_sing.txt', 'r')
rhs = np.loadtxt(file)
ax.plot(rhs)
plt.yscale('log')
plt.savefig('rhs_sing.png')
