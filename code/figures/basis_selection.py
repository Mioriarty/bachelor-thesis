import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import scipy.optimize

POINTS = np.array([
    [3, 1.6],
    [1, 6],
    [1.5, 4],
    [7, 5],
    [2, 9],
    [4.5, 5.5],
    [4, 3.5],
    [6, 1],
    [6.5, 7.5]
])

B = np.array([10, 3])

axis_max = max(np.max(POINTS), np.max(B)) + 1

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

# Calculate the intersection point
# ray = B / np.linalg.norm(B)
# good_equations = hull.equations[hull.good]
# normals, offsets = good_equations[:, :-1], good_equations[:, -1]
# gammas = [-offsets[i] / np.dot(normals[i], ray) for i in range(offsets.shape[0])]
# gamma = np.max(gammas)
# p = ray * gamma

# Alternative way to compute intersection point
# Solve LP:
# Find (x, γ) such that
# Minimize γ
# Ax - γ*b = 0
# x^T * vec(1) = 1
# (x, γ) > 0

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
    plt.text(x - 0.1, y + 0.1, fr'$\vec a_{i+1}$', verticalalignment='bottom', horizontalalignment='right', fontsize=14)

plt.text(B[0] - 0.1, B[1] + 0.1, fr'$\vec b$', verticalalignment='bottom', horizontalalignment='right', fontsize=14)
plt.text(p[0] - 0.05, p[1] + 0.2, fr'$\vec p$', verticalalignment='bottom', horizontalalignment='right', fontsize=12)


# Do axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,  labelbottom=False, labelleft=False, labeltop=False, labelright=False)

ax.annotate('', xy=(11, 0), xytext=(0, 0),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(0, 11), xytext=(0, 0),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.text(10.7, -0.7, '$x$', fontsize=14, ha='center')
ax.text(-0.5, 10.7, '$y$', fontsize=14, va='center')
ax.text(-0.1, -0.1, '$0$', ha='right', va='top', fontsize=12)


plt.xlim(0, axis_max)
plt.ylim(0, axis_max)

plt.show()