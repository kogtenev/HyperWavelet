import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.sparse as scp

E = np.loadtxt('solution.txt')

n = int(len(E) / 2)
E = E.reshape((n, 2), order = 'C')

E_max = np.max(E)
E_x = E[:, 0]
E_y = E[:, 1]

n = int(np.sqrt(1.0 * n))
E_x = E_x.reshape(n, n)
E_y = E_y.reshape(n, n)

extent = 0., 1., 0., 1.

fig, ax = plt.subplots()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$|j_x|$')
p = ax.imshow(E_x, cmap = 'plasma', clim=(0., E_max), extent=extent)
fig.colorbar(p)
ax.legend([],[], frameon=False)
plt.savefig('E_x.png', dpi = 199) 

fig, ax = plt.subplots()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'|$j_y$|')
p = ax.imshow(E_y, cmap = 'plasma', clim=(0., E_max), extent=extent)
fig.colorbar(p)
ax.legend([],[], frameon=False)
plt.savefig('E_y.png', dpi = 199) 

fig, ax = plt.subplots()
A = np.loadtxt('mat.txt')
A = np.abs(A)
a_max = np.max(A)
A += (1e-8 / a_max)
a_min = np.min(A)
A /= a_max
a_min /= a_max
p = ax.imshow(A, cmap = 'plasma', norm = LogNorm(vmin=a_min, vmax=1.))
fig.colorbar(p)
ax.legend([],[], frameon=False)
plt.savefig('matrix.png', dpi = 199)

sparse_data = np.loadtxt('trunc_mat.txt')
i = sparse_data[:, 0].astype(int)
j = sparse_data[:, 1].astype(int)
vals = sparse_data[:, 2]
A = scp.coo_matrix((vals, (i, j)))
ax.clear()
ax.legend([],[], frameon=False)
fig.clear()
plt.spy(A, markersize = 1)
plt.savefig('truncated_matrix.png', dpi = 199)
