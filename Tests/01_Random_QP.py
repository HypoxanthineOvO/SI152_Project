import numpy as np
import os, sys

# Constants
alpha = 1e-2
nonzero_ratio = 0.8
FILE = "../01_RANDOM_QP.txt"

# Variables
n = 5
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
    m = 1 * n
    I_n = np.identity(n)
    
    # Generate P, q, A, b
    ## P = MM^T+alpha * I
    ### M: 15% Non-zero elements
    M = np.zeros((n, n))
    num_nonzero_NN = int(n * n * nonzero_ratio)
    if num_nonzero_NN < 1:
        num_nonzero_NN = 1
    M[np.random.randint(0, n, num_nonzero_NN), np.random.randint(0, n, num_nonzero_NN)] = np.random.randn(num_nonzero_NN)
    P = M @ M.T + alpha * I_n
    ## q: Random vector
    q = np.random.randn(n)
    
    ## A: Random matrix
    A = np.zeros((m, n))
    num_nonzero_MN = int(n * m * nonzero_ratio)
    if num_nonzero_MN < 1:
        num_nonzero_MN = 1
    A[np.random.randint(0, m, num_nonzero_MN), np.random.randint(0, n, num_nonzero_MN)] = np.random.randn(num_nonzero_MN)
    ## l <= Ax <= u
    left_bound = np.random.uniform(-1, 0, (m))
    right_bound = np.random.uniform(0, 1, (m))
    
    Final_A = np.zeros((2*m, n))
    Final_A[:m] = A
    Final_A[m:] = -A
    Final_b = np.zeros(2*m)
    Final_b[:m] = right_bound # Ax <= u == Ax + (-u) <= 0
    Final_b[m:] = -left_bound # -Ax <= -l == -Ax + l <= 0
    
    # To align with the Ax + b <= 0 format 
    Final_b = -Final_b
    
    # Save the variables as code to a file
    with open(FILE, "w") as f:
        f.write(f"n: {n}\n")
        f.write(f"m: {2 * m}\n")
        # To flatten the matrix, we use the .tolist() method
        f.write(f"H: {P.tolist()}\n")
        f.write(f"g: {q.tolist()}\n")
        f.write(f"AI: {Final_A.tolist()}\n")
        f.write(f"bI: {Final_b.tolist()}\n")
        f.write(f"AE: None\n")
        f.write(f"bE: None\n")
    
    print("========== Random QP Generated ==========")
    print(f"n: {n}, m: {2 * m}")