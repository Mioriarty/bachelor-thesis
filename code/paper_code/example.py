import numpy as np
from find_mu import find_mu
from mu_to_gauss_steps import mu_to_gauss_steps
from extend_to_CCM import extend_to_CCM

def run_example() -> None:
    # Define values from chapter 3.3.5
    A = np.array([[6, 0, 5, 2, 0, 2],
                  [4, 2, 5, 6, 1, 3]])
    b = np.array([6, 11])

    print("k-bound with optimisation:", k_bound(A, b))

    # Run optimisation
    mu = find_mu(A, b)
    C = mu_to_gauss_steps(mu, A)

    A_tilde = C @ A
    b_tilde = C @ b
    print("k-bound with optimisation:", k_bound(A_tilde, b_tilde))

    # Extend optimized system of linear equations
    A_ext, b_ext = extend_to_CCM(A_tilde, b_tilde)
    

    




def k_bound(A, b):
    vec1 = np.ones(shape=A.shape[0])
    return np.dot(b, vec1) / np.min(A.T @ vec1)

if __name__ == "__main__":
    run_example()