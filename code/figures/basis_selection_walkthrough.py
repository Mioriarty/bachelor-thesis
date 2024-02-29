import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import scipy.optimize

POINTS = np.array([
    [6, 4],
    [0, 2],
    [5, 5],
    [2, 6],
    [0, 1],
    [2, 3]
])

B = np.array([6, 11])

plt.rcParams['text.usetex'] = True
_, ax = plt.subplots()

# Do the convex hull magic
points = np.vstack((np.array([0, 0]), POINTS))
hull = scipy.spatial.ConvexHull(points=points, qhull_options='QG0')

for i, visible_facet in enumerate(hull.simplices):
    if hull.good[i]:
        plt.plot(hull.points[visible_facet, 0], hull.points[visible_facet, 1], color='green', lw=2)
    else:
        plt.plot(hull.points[visible_facet, 0], hull.points[visible_facet, 1], color='black', lw=1)

c = np.array([0] * POINTS.shape[0] + [1])
b_eq = np.array([0] * POINTS.shape[1] + [1])
A = np.column_stack((POINTS.T, -B))
A = np.row_stack((A, np.array([1] * POINTS.shape[0] + [0])))
res = scipy.optimize.linprog(c=c, A_eq=A, b_eq=b_eq)
p = B * res.x[-1]
selected = [i for i in range(POINTS.shape[0]) if not np.isclose(res.x[i], 0)]

plt.plot([0, B[0] * 10], [0, B[1] * 10], linestyle=(0, (5, 7)), lw=0.7, color="red")

# Draw the selected highlight
plt.scatter(POINTS[selected, 0], POINTS[selected, 1], color='red', lw=3, zorder=2)

plt.scatter(POINTS[:,0], POINTS[:,1], zorder=10)
plt.scatter(B[0], B[1], color='red', label='B')
plt.scatter(p[0], p[1], zorder=10, lw=0.1)

for i, (x, y) in enumerate(POINTS):
    plt.text(x - 0.1, y + 0.1, fr'$\vec a_{{{i+1}}}$', verticalalignment='bottom', horizontalalignment='right', fontsize=14)

plt.text(B[0] - 0.1, B[1] + 0.1, fr'$\vec b$', verticalalignment='bottom', horizontalalignment='right', fontsize=14)
plt.text(p[0] - 0.05, p[1] + 0.2, fr'$\vec p$', verticalalignment='bottom', horizontalalignment='right', fontsize=12)


# Do axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-0.5, 6.5)
plt.ylim(-0.5, 11.5)

plt.savefig("walkthrough_basis_selection.png")
plt.show()
