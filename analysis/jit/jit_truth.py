from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import sqrt
from numpy import minimum

def f(x, y):
    eps = 0.00001
    a = 378370
    b = 100
    delta0 = 218292735.245136
    beta0 = 57691.989596691725
    l = sqrt(delta0*beta0) - eps
    delta1 = delta0 + y  
    beta1 = beta0 + x 
    delta2 = (delta1*beta1)/(beta1 + 0.997 * b)
    beta2 = beta1 + b 
    delta3 = delta2 + a
    beta3 = (delta2*beta2)/(delta2 + 0.997*a)
    delta4 = (delta3*beta3)/(beta3 + 0.997 * b)
    beta4 = beta3 + b 
    delta5 = delta4 + a
    beta5 = (delta4*beta4)/(delta4 + 0.997*a)
    
    neta = minimum((y*l)/delta0, (x*l)/beta0)
    # neta = (y*l)/delta0
    betaret = neta * beta5 / (neta + l) - x
    deltaret = neta * delta5 / (neta + l) - y
    print()
    #     return (betaret ) + (beta5*delta5)/(delta5 + 0.997*deltaret)
    # else:
    #     pass
    temp1 = beta5 - (beta5*delta5)/(delta5 + 0.997*deltaret)
    temp2 = (1000*deltaret*beta5) / (1 + 997*(delta5+deltaret))
    sign = np.sign(deltaret)
    return betaret + (sign+1)*(sign) * temp1 / 2 + (sign-1) * sign * temp2/2
    
    # return betaret
    # return (betaret ) + (beta5*delta5)/(delta5 + 0.997*deltaret)
    # return x*y

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 1000, 2)
Y = np.arange(0, 5000000, 10000)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

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