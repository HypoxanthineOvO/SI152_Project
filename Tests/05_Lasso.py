import numpy as np
import os, sys

# Constants
alpha = 1e-2
nonzero_ratio = 0.15
FILE = "../LASSO.txt"

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
    m = 100 * n
    A = np.zeros((m, n))
    num_nonzero_MN = int(n * m * nonzero_ratio)
    if num_nonzero_MN < 1:
        num_nonzero_MN = 1
    
    v = np.zeros(n)
    v[np.random.randint(0, n, n//2)] = np.random.normal(0, 1 / n, n//2)
    epsilon = np.random.randn(m)
    b = A @ v + epsilon

    raise NotImplementedError("LASSO Problem not implemented yet")
    # Save the variables as code to a file
    # with open(FILE, "w") as f:
    #     f.write(f"n: {n}\n")
    #     f.write(f"m: {2 * m}\n")
    #     # To flatten the matrix, we use the .tolist() method
    #     f.write(f"H: {P.tolist()}\n")
    #     f.write(f"g: {q.tolist()}\n")
    #     f.write(f"AI: {Final_A.tolist()}\n")
    #     f.write(f"bI: {Final_b.tolist()}\n")
    #     f.write(f"AE: None\n")
    #     f.write(f"bE: None\n")
    
    
    print("========== LASSO Problem Generated ==========")
    print(f"n: {n}, m: {2 * m}")