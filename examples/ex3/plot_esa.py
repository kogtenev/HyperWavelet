import numpy as np
from numpy import linalg
import pymesh
import matplotlib.pyplot as plt

true_esa = np.loadtxt('esa.txt')
plt.plot(true_esa)
plt.xlabel(r'$\alpha$' + ', degrees')
plt.ylabel(r'$\sigma$' + ', dB')
plt.savefig('esa.png')