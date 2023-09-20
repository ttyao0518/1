import numpy as np

from scipy.special import factorial, genlaguerre, sph_harm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a0 = 1

def hydrogen_wave_function(n, l, m):
    def R(r):
        factor = np.sqrt((2. / (n * a0)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        rho = 2 * r / (n * a0)
        return factor * (rho ** 1) * np.exp(-rho / 2) * genlaguerre(n - l - 1, 2 * l + 1)(rho)
    def Y(theta, phi):
        return sph_harm(m, l, phi, theta)

    return lambda r, theta, phi: R(r) * Y(theta, phi)

n, l, m = 2, 1, 0
psi = hydrogen_wave_function(n, l, m)

#print(f"{psi = }")
#print(f"psi(1, np.pi / 4, np.pi / 3){psi(1, np.pi / 4, np.pi / 3)}")

limit = 10
n_points = 50
vec = np.linspace(-limit, limit, n_points)
#print(f"vec = {vec}")
#print(f"len(vec) = {len(vec)}")

X, Y, Z = np.meshgrid(vec, vec, vec)
#print(f"X.shape = {X.shape}", f"Y.shape = {Y.shape}", f"Z.shape = {Z.shape}")

coords_xyz = np.vstack(list(map(np.ravel, (X, Y, Z)))).T
#print(f"len(coords_xyz) = {len(coords_xyz)}")
#print(coords_xyz)

R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
THETA = np.arccos(Z / R)
PHI = np.arctan2(Y, X)
#print(f"R.shape = {R.shape}", f"THETA.shape = {THETA.shape}", f"PHI.shape = {PHI.shape}")

coords_rtf = np.vstack(list(map(np.ravel, (R, THETA, PHI)))).T
#print(f"len(coords_rtf) = {len(coords_rtf)}")
#print(coords_rtf)

psi_values = psi(R, THETA, PHI)
#print(f"psi_values.shape = {psi_values.shape}")
#print(f"psi_values[0][0][0] = {psi_values[0][0][0]}")

prob_dens = np.abs(psi_values) ** 2

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(psi_values[:, 0, 0], psi_values[0, :, 0], psi_values[0, 0, :])
plt.show()