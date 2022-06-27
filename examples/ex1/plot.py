import numpy as np
import matplotlib.pyplot as plt

file = open('sol.txt', 'r')

a = -1.
b = 1.

data = np.loadtxt(file)

dim = len(data)
step = (b - a) / dim
h = step / 2
a += h
b -= h
step = (b - a) / dim
x = np.arange(a, b, step)

fig, ax = plt.subplots()
ax.plot(x, data)
plt.savefig('solution.png')
