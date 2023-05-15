import numpy as np
from numpy import linalg
import pymesh
import matplotlib.pyplot as plt

true_esa = np.loadtxt('esa.txt')
alpha = np.linspace(0, 360, 181)
plt.plot(alpha, true_esa)
plt.xlabel(r'$\alpha$' + ', degrees')
plt.ylabel(r'$\sigma$' + ', dB')
plt.grid()
plt.ylim(-50, np.max(true_esa)+3)
plt.savefig('esa.png')