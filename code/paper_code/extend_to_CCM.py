import numpy as np

def extend_to_CCM(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extends any given system of linear equations to a CCM system of linear
    equations according to theorem 3.1

    Args:
        A (np.ndarray): Matrix representation of the system of linear equations
        b (np.ndarray): Inhomogenity of the system of equations

    Returns:
        tuple[np.ndarray, np.ndarray]: New CCM system of linear equations.
        First the matrix and second the inhomoginity.
    """
    # append column of zeros to A
    A = np.hstack((A, np.zeros((A.shape[0], 1))))

    # Compute all column sums in A
    column_sums = np.sum(A, axis=0)

    # Compute v_i and append that column to A
    alpha = np.max(column_sums)
    new_row = alpha - column_sums
    A = np.vstack((A, new_row))

    # Compute beta and append that to b
    s = np.min(column_sums[:-1])
    k = np.ceil(1/s * np.sum(b))
    beta = alpha * k - np.sum(b)
    b = np.append(b, beta)

    return A, b