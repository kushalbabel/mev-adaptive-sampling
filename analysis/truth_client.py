import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import sqrt
from numpy import minimum
from simulate_client import simulate
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support


template = ''

def parallel(x,y):
    global template
    counter = 0
    processors = 24
    mev = np.empty_like(x)
    batch = []
    positions = []
    for iy, ix in np.ndindex(x.shape):
        concrete = []
        for line in template:
            concrete.append(line.replace('alpha1', str(x[iy, ix])).replace('alpha2', str(y[iy, ix])).strip())
        batch.append((concrete, (counter % processors)))
        positions.append((iy, ix))
        counter += 1
        print(counter)
        # print(counter)
        if (len(batch) == processors) or x.size == counter:
            with Pool() as pool:
                # print(batch)
                result = pool.starmap(simulate, batch)
                # print(result)
                for idx in range(len(positions)):
                    mev[positions[idx][0], positions[idx][1]] = result[idx]
            batch = []
            positions = []
    return mev

def f(x, y):
    # print("X", x)
    # print("Y", y)
    global template
    counter = 0
    mev = np.empty_like(x)
    
    for iy, ix in np.ndindex(x.shape):
        # mev[iy, ix] = x[iy, ix] + y[iy, ix]
        # print(counter)
        concrete = []
        for line in template:
            concrete.append(line.replace('alpha1', str(x[iy, ix])).replace('alpha2', str(y[iy, ix])).strip())
        # print('\n'.join(concrete))
        mev[iy,ix] = simulate(concrete, -1)    
        counter += 1
    return mev

template = open('manualtests/template').readlines()
# template = open('eth_token_tests/0x397ff1542f962076d0bfe58ea045ffa2d347aca0/13076406/amm_reduced').readlines()

# Make data.
X = np.linspace(0, 1000, 100)
Y = np.linspace(0, 3170000000000, 100)
# Y = np.linspace(0, 1000, 100)
X, Y = np.meshgrid(X, Y)
Z = parallel(X, Y)

# Plot the surface.

# print(X)
# print(Y)
# print(Z)

fig = plt.figure()
ax = plt.axes(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.contour3D(X, Y, Z, cmap='binary')


# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
ax.set_xlabel('alpha1')
ax.set_ylabel('alpha2')
ax.set_zlabel('mev');
plt.show()
# plt.savefig('temp.png')

# x = np.linspace(0, 1000, 690)
# y = np.linspace(0, 5000000, 690)

# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()