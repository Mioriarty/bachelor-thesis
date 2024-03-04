import numpy as np
import scipy.optimize, scipy.linalg


def find_mu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find the μ that minimizes k(A, b; μ).
    It does that according to chapter 3.3.3.

    Args:
        A (np.ndarray): Input A
        b (np.ndarray): Input b

    Returns:
        np.ndarray: The μ that minimizes k(A, b; μ)
    """
    # Step 1: Select basis with linear program
    ## Run the linear program
    x = run_LP_for_basis_selection(A, b)

    ## Extrect all columns who's coefficiant if non-zero
    selected_columns = [i for i in range(A.shape[1]) if not np.isclose(x[i], 0)]
    A_prime = A[:, selected_columns]


    # Step 2: Find a mu in the correct solution space
    ## Compute the two null space basis seperately
    null_space_delta = null_space(delta(A_prime))
    null_space_transpose = null_space(A_prime.T)

    ## Concatinate the two basis and retrieve last pivot column
    concatinated_matrix = np.concatenate((null_space_transpose, null_space_delta), axis=1)
    mu = pivot_columns(concatinated_matrix)[:, -1]

    ## Make sure that the dot-product is positive
    d = np.dot(mu, A_prime[:,0])
    return mu if d >= 0 else -mu


def run_LP_for_basis_selection(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Runs the linear program described in chapter 3.3.4.1 to select an apropriate basis.

    Args:
        A (np.ndarray): Input A
        b (np.ndarray): Input b

    Returns:
        np.ndarray: Solution vector x. The first n components are the coefficiants and the last component is γ.
        See chapter 3.3.4.1 for details.
    """
    # Construct omega, A_tilde, b_tilde
    omega = np.array([0] * A.shape[1] + [1])

    b_tilde = np.array([0] * A.shape[0] + [1])

    A_tilde = np.column_stack((A, -b))
    A_tilde = np.row_stack((A_tilde, np.array([1] * A.shape[1] + [0])))

    # Run linear program
    return scipy.optimize.linprog(c=omega, A_eq=A_tilde, b_eq=b_tilde).x


def delta(matrix: np.ndarray) -> np.ndarray:
    """Compute Δ() of some matrix according to definition 3.2

    Args:
        matrix (np.ndarray): The input matrix

    Returns:
        np.ndarray: Δ of the input matrix
    """
    return matrix[:, :-1].T - matrix[:, 1:].T

def null_space(matrix: np.ndarray) -> np.ndarray:
    """Compute a basis of the null space of a matrix.

    Args:
        matrix (np.ndarray): Input matrix

    Returns:
        np.ndarray: A matrix who's columns are a basis of the null space of that input matrix
    """
    return np.array(scipy.linalg.null_space(matrix))

def pivot_columns(matrix: np.ndarray) -> np.ndarray:
    """Computes the pivot columns of a matrix.

    Args:
        matrix (np.ndarray): The input matrix

    Returns:
        np.ndarray: All pivot columns of that matrix smushed into one output matrix.
    """
    # Perform LU decomposition
    _, _, U = scipy.linalg.lu(matrix)
    U[np.isclose(U, 0)] = 0 # Fixing close to zero values

    pivot_columns = [np.flatnonzero(U[i, :])[0] for i in range(U.shape[0]) if len(np.flatnonzero(U[i, :])) > 0]
    
    # Select only the pivot columns from the original matrix
    basis_matrix = matrix[:, pivot_columns]
    
    return basis_matrix