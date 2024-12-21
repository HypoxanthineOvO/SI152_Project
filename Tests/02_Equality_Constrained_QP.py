import numpy as np
import os, sys

# Constants
alpha = 1e-2
nonzero_ratio = 0.4
FILE = "../EQU_Constr_QP.txt"

# Variables
n = 2
random_seed = 0


if __name__ == "__main__":
    # Read n and random_seed from the command line
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        random_seed = int(sys.argv[2])
    if len(sys.argv) > 3:
        FILE = os.path.join("..", sys.argv[3])
    
    np.random.seed(random_seed)
    
    # Generate variables
    m = np.floor(n / 2).astype(int)
    I_n = np.identity(n)
    
    # Generate P, q, A, b
    ## P = MM^T+alpha * I
    ### M: 15% Non-zero elements
    M = np.zeros((n, n))
    num_nonzero_NN = round(n * n * nonzero_ratio)
    if num_nonzero_NN < 1:
        num_nonzero_NN = 1
    M[np.random.randint(0, n, num_nonzero_NN), np.random.randint(0, n, num_nonzero_NN)] = np.random.randn(num_nonzero_NN)
    print(f"M: {M}")
    P = M @ M.T + alpha * I_n
    print(f"P: {P}")
    ## q: Random vector
    q = np.random.randn(n)
    
    ## A: Random matrix
    num_nonzero_MN = int(n * m * nonzero_ratio)
    A = np.zeros((m, n))
    A[np.random.randint(0, m, num_nonzero_MN), np.random.randint(0, n, num_nonzero_MN)] = np.random.randn(num_nonzero_MN)
    ## b: Random vector
    b = np.random.randn(m)
    # Ax = b & Ax + b = 0
    
    # Save the variables as code to a file
    with open(FILE, "w") as f:
        f.write(f"n: {n}\n")
        f.write(f"m: {m}\n")
        # To flatten the matrix, we use the .tolist() method
        f.write(f"H: {P.tolist()}\n")
        f.write(f"g: {q.tolist()}\n")
        f.write(f"AI: None\n")
        f.write(f"bI: None\n")
        f.write(f"AE: {A.tolist()}\n")
        f.write(f"bE: {(-b).tolist()}\n")
    
    print("========== Equality Constrained QP Generated ==========")
    print(f"n: {n}, m: {m}")
    
    
    # This QP can be explicitly solved
    left_mat = np.zeros((n+m, n+m))
    left_mat[:n, :n] = P
    left_mat[n:, :n] = A
    left_mat[:n, n:] = A.T
    right_vec = np.zeros(n+m)
    right_vec[:n] = q
    right_vec[n:] = b
    x_ref = np.linalg.solve(left_mat, right_vec)[:n]
    print(f"x_ref: {-x_ref}")
    opt_val = 0.5 * x_ref.T @ P @ x_ref + q.T @ x_ref
    print(f"Optimal Value: {opt_val}")