import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.optimize

def k(A, b, mu):
    return np.dot(b, mu) / np.min(A.T @ mu)

def choice_of_basis(matrix):
    # Perform LU decomposition
    _, _, U = scipy.linalg.lu(matrix)
    U[np.isclose(U, 0)] = 0 # Fixing close to zero values

    pivot_columns = [np.flatnonzero(U[i, :])[0] for i in range(U.shape[0]) if len(np.flatnonzero(U[i, :])) > 0]
    
    # Select only the pivot columns from the original matrix
    basis_matrix = matrix[:, pivot_columns]
    
    return basis_matrix

def null_space(matrix):
    return np.array(scipy.linalg.null_space(matrix))

def basis_selection_by_convex_hull(matrix, b):
    zero_vec = np.zeros((matrix.shape[0], 1))
    points = np.concatenate((zero_vec, matrix), axis=1).T
    hull = scipy.spatial.ConvexHull(points=points, qhull_options="QG0")

    ray = b / np.linalg.norm(b)

    good_equations = hull.equations[hull.good]
    normals, offsets = good_equations[:, :-1], good_equations[:, -1]
    gammas = [-offsets[i] / np.dot(normals[i], ray) for i in range(offsets.shape[0])]

    facet_index = np.argmax(gammas)
    facet = hull.simplices[hull.good][facet_index]
    column_indecies = np.array([i-1 for i in facet])
    return matrix[:, column_indecies]

def delta(matrix):
    return matrix[:, :-1].T - matrix[:, 1:].T

def algorithm(matrix, b):
    # Step 1/2: Convex hull and basis selection
    matrix_prime = basis_selection_by_convex_hull(matrix, b)

    # Step 3: Compute ∆ of that matrix and its null space
    null_space_delta = null_space(delta(matrix_prime))
    print(null_space_delta)

    # Step 4: Null Space of tranpose
    null_space_transpose = null_space(matrix.T)
    print(null_space_transpose)

    # Step 5: Find correct µ
    concatinated_matrix = np.concatenate((null_space_transpose, null_space_delta), axis=1)
    print(concatinated_matrix)
    mu = choice_of_basis(concatinated_matrix)[:, -1]
    print(mu)

    # Step 6: Make sure d > 0
    d = np.dot(mu, matrix_prime[:,0])

    print(matrix_prime.T @ mu)
    return mu if d > 0 else -mu

def construct_gauss_steps(matrix, mu):
    # Contruct D
    ds = np.diag([1 if np.isclose(m, 0) else m for m in mu])
    es = np.diag([-1 if np.isclose(m, 0) else 0 for m in mu])
    es = np.vstack((es[1:,:], es[0,:]))
    D = ds + es

    matrix_tilde = D @ matrix
    c = np.max(np.abs(matrix_tilde))
    C_prime = np.identity(mu.shape[0]) + np.ones((mu.shape[0], mu.shape[0])) * c

    return C_prime @ D


def get_random_ilp(min_rows=2, max_rows=25, min_cols=2, max_cols=25, max_entry=50):
    # Define the number of rows and columns randomly
    num_rows = np.random.randint(min_rows, max_rows + 1)
    num_cols = np.random.randint(min_cols, max_cols + 1)

    # Generate a random matrix with integer entries up to max_entry
    matrix = np.random.randint(0, max_entry + 1, size=(num_rows, num_cols))

    # Generate a solution
    # solution = np.random.randint(0, high=10, size=matrix.shape[1])
    solution = np.ones(matrix.shape[1])

    # Get rhs
    b = matrix @ solution

    return matrix, b

def numerical_minimum(A, b):
    result = scipy.optimize.minimize(lambda mu, A=A, b=b: k(A, b, mu), np.ones(A.shape[0]), method='nelder-mead')
    return result.x

def complete_ilp(matrix, b):
    matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1))))
    column_sums = np.sum(matrix, axis=0)
    alpha = np.max(column_sums)
    new_row = alpha - column_sums
    matrix = np.vstack((matrix, new_row))

    s = np.min(column_sums[:-1])
    k = np.ceil(1/s * np.sum(b))
    b = np.append(b, alpha * k - np.sum(b))

    return matrix, b


def solution_component_sum(matrix, b):
    # In matrix all columns must have the same positive component sum
    return np.sum(b) / np.sum(matrix[:,0])


matrix, b = get_random_ilp()
print("=== INPUT ===")
print(matrix)
print(b)

print("=== ALGORITHM ===")
my_res = algorithm(matrix, b)
num_res = numerical_minimum(matrix, b)

print("=== OUTPUT ===")
print(matrix.T @ my_res)
print(matrix.T @ num_res)
print(k(matrix, b, my_res), k(matrix, b, num_res))

print("=== Create C from mu ===")
my_res = (my_res * 10000).astype(int) # maybe with log_10 estimate the size of the result
C = construct_gauss_steps(matrix, my_res)

completed_mat, completed_b = complete_ilp(matrix, b)
completed_optimized_mat, completed_optimized_b = complete_ilp(C @ matrix, C @ b)
print(solution_component_sum(completed_mat, completed_b))
print(solution_component_sum(completed_optimized_mat, completed_optimized_b))

