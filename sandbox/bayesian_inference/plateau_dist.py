import numpy as np
from scipy import special
from scipy.stats import gennorm

import matplotlib.pyplot as plt

# plateau shaped distribution (following https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution)
# x = np.arange(-8, 8, 0.1)
# a = 5
# sigma = 0.5
#
# def f(x, a, sigma):
#     y = 1 / (4 * a) * (special.erf((x + a) / (sigma * np.sqrt(2))) - special.erf((x - a) / (sigma * np.sqrt(2))))
#     return y
#
# y_orig = f(x, a, sigma)
# y_orig = y_orig / sum(y_orig)
#
# y = y_orig
#
# n_iter = 50
# for _ in range(n_iter):
#     # y = f(y, a, sigma)
#     y = y * y_orig
#     y = y / sum(y)
#
# plt.plot(x, y_orig)
# plt.plot(x, y)
# plt.legend(['orig', 'posterior'])
# plt.show()

# plateau shaped distribution (using scipy gennorm)
x = np.arange(-15, 15, 0.1)
beta = 2
sigma = 3
loc = 5
new_loc = -5
new_new_loc = -10

y_orig = gennorm.pdf(x, beta, loc, sigma)
y_orig = y_orig / sum(y_orig)

y_target = gennorm.pdf(x, beta, new_loc, sigma)
y_target = y_target / sum(y_target)

y_target2 = gennorm.pdf(x, beta, new_new_loc, sigma)
y_target2 = y_target2 / sum(y_target2)

y = y_orig

# n_iter = 10
# for _ in range(n_iter):
#     y = gennorm.pdf(x, beta, new_loc, sigma)
#     y = y * y_orig
#     y = y / sum(y)

y = y * y_target
y = y / sum(y)
#
# y = y * y_target2
# y = y / sum(y)


plt.plot(x, y_orig)
plt.plot(x, y_target)
plt.plot(x, y)
# plt.plot(x, y_target2)
plt.legend(['likelihood1', 'likelihood2', 'posterior'])

# plt.plot(x, y_orig)
plt.show()