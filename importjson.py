import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the heightmap JSON
with open('hmap.json') as f:
    data = json.load(f)
xs = np.array(data['xs'])
zs = np.array(data['zs'])
H = np.array(data['H'])  # shape (nx, nz)

# Create meshgrid for plotting
X, Z = np.meshgrid(xs, zs, indexing='ij')  # X, Z both shape (nx, nz)
Y = H  # Height values

# Plot 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
ax.set_xlabel('X')
ax.set_ylabel('Height (Y)')
ax.set_zlabel('Z')
plt.show()