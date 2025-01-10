import numpy as np
import os, sys

# Constants
alpha = 1e-2
nonzero_ratio = 0.3
FILE = "../03_MIXED_QP.txt"

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
        FILE = os.path.join(".", sys.argv[3])
    
    np.random.seed(random_seed)
    
    # Generate variables
    m_INEQ = 4 * n
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
    A = np.zeros((m_INEQ, n))
    num_nonzero_MN = int(n * m_INEQ * nonzero_ratio)
    if num_nonzero_MN < 1:
        num_nonzero_MN = 1
    A[np.random.randint(0, m_INEQ, num_nonzero_MN), np.random.randint(0, n, num_nonzero_MN)] = np.random.randn(num_nonzero_MN)
    ## l <= Ax <= u
    left_bound = np.random.uniform(-1, 0, (m_INEQ))
    right_bound = np.random.uniform(0, 1, (m_INEQ))
    
    Final_A = np.zeros((2*m_INEQ, n))
    Final_A[:m_INEQ] = A
    Final_A[m_INEQ:] = -A
    Final_b = np.zeros(2*m_INEQ)
    Final_b[:m_INEQ] = right_bound # Ax <= u
    Final_b[m_INEQ:] = -left_bound # -Ax <= -l
    
    # 1 Equality constraints
    m_EQ = n // 8
    if m_EQ == 0:
        m_EQ = 1
    
    num_nonzero_MN_EQ = int(n * m_EQ * nonzero_ratio)
    
    mean_A = np.random.randint(1, 10)
    std_A = np.random.randint(1, 10)
    A_EQ = np.random.normal(mean_A, std_A, (m_EQ, n))
    
    # Generate b: Random vector
    mean_b = np.random.randint(-100, 100)
    std_b = np.random.randint(1, 100)
    b_EQ = np.random.normal(mean_b, std_b, m_EQ)
    
    
    
    
    
    
    A_EQ = np.random.randn(m_EQ, n)
    ## b: Random vector
    # b_EQ = np.random.randn(m_EQ)
    b_EQ = np.random.uniform(0, 1, m_EQ)
    
    ## To align with the Ax + b <= 0 format
    Final_b = -Final_b
    
    # Save the variables as code to a file
    with open(FILE, "w") as f:
        f.write(f"n: {n}\n")
        f.write(f"m: {2 * m_INEQ + 1}\n")
        # To flatten the matrix, we use the .tolist() method
        f.write(f"H: {P.tolist()}\n")
        f.write(f"g: {q.tolist()}\n")
        f.write(f"AI: {Final_A.tolist()}\n")
        f.write(f"bI: {Final_b.tolist()}\n")
        f.write(f"AE: {A_EQ.tolist()}\n")
        f.write(f"bE: {(-b_EQ).tolist()}\n")
    
    print("========== Mixed QP Generated ==========")
    print(f"n: {n}, m: {2 * m_INEQ + m_EQ}")