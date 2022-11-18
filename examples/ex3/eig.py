import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

eig = np.loadtxt('eig.txt')

plt.grid()
plt.scatter(eig[:, 0], eig[:, 1], s=6)
plt.xlabel(r'$Re\: \lambda$')
plt.ylabel(r'$Im\; \lambda$')

plt.savefig('eig.png', dpi=199)