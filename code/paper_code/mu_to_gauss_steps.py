import numpy as np


def mu_to_gauss_steps(mu: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Construct gauss steps i.e. the invertable matrix C from some vector μ according to lemma 3.2 1).

    Args:
        mu (np.ndarray): Vector μ that should be turned into gauss steps
        A (np.ndarray): Input matrix A

    Returns:
        np.ndarray: Invertable matrix C that represents the gauss steps dictadet by μ.
    """
    # Contruct D
    # Constructing matrices that only contains d entries and e entrie sfirst and add them together
    ds = np.diag([1 if np.isclose(m, 0) else m for m in mu])
    es = np.diag([-1 if np.isclose(m, 0) else 0 for m in mu])
    es = np.vstack((es[1:,:], es[0,:]))
    D = ds + es

    # Construct C_prime
    A_tilde = D @ A
    c = np.max(np.abs(A_tilde))
    C_prime = np.identity(mu.shape[0]) + np.ones((mu.shape[0], mu.shape[0])) * c

    # return C = C_prime * D
    return C_prime @ D