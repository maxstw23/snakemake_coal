import matplotlib.pyplot as plt
import numpy as np

# create a figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# heres how you can remove grid and panes
ax.grid(False)
ax.set_axis_off()
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
bound = 1.5
X, Y = np.meshgrid(np.linspace(-bound, bound, 10),
                   np.linspace(-bound, bound, 10))
Z = np.zeros_like(X)
# create a wireframe grid in xz-plane
ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.1)

# example of how you cna do the arrows
ax.quiver([0], [0], [0], [0], [2], [0], color='black', linewidth=1.5)
ax.quiver([0], [0], [0], [2], [0], [0], color='black', linewidth=1.5)
ax.quiver([0], [0], [0], [0], [0], [2], color='black', linewidth=1.5)

# then your 3D plot code would go here
plt.show()