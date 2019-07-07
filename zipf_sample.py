import numpy as np
import matplotlib.pyplot as plt
from scipy import special


a = 2. # parameter
s = np.random.zipf(a, 1000)

# Hist
count, bins, ignored = plt.hist(s[s<50], 50, normed=True)
x = np.arange(1., 1000.)
y = x**(-a) / special.zetac(a)
plt.plot(x, y/max(y), linewidth=2, color='r')
plt.show()

# Log Log
plt.xscale('log')
plt.yscale('log')
plt.scatter(x, y, color='r')
plt.savefig('logscale.png')
