import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

E = np.loadtxt('solution.txt')

n = int(len(E) / 2)
E = E.reshape((n, 2), order = 'C')

E_max = np.max(E)
E_x = E[:, 0]
E_y = E[:, 1]

n = int(np.sqrt(1.0 * n))
E_x = E_x.reshape(n, n)
E_y = E_y.reshape(n, n)

fig, ax = plt.subplots()
p = ax.imshow(E_x, cmap = 'plasma', clim=(0., E_max))
fig.colorbar(p)
ax.legend([],[], frameon=False)
plt.savefig('E_x.png', dpi = 199) 

fig, ax = plt.subplots()
p = ax.imshow(E_y, cmap = 'plasma', clim=(0., E_max))
fig.colorbar(p)
ax.legend([],[], frameon=False)
plt.savefig('E_y.png', dpi = 199) 
