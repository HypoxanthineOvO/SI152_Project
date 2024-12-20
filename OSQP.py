import numpy as np
import scipy.sparse as sp

# Constants
MAX_ITER = 50000
EPS = 1e-5

# Denotions
"""
AE & bE: Equality constraints. AE x = bE
AI & bI: Inequality constraints. AI x <= bI
"""

# Util Functions
def printVec(x: np.ndarray, num_digits: int = 4):
    for i in range(x.shape[0]):
        rounded_x = round(x[i], num_digits)
        print("{:.4f}".format(rounded_x), end = " ")
    print()

def projection(pt: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    # Clip the values
    return np.clip(pt, l, u)

if __name__ == "__main__":

    # Initialize the problem's parameters

    with open("./RANDOM_QP.txt", "r") as f:
        exec(f.read())

    # Check Dimensions
    I_n = np.identity(n)
    I_m = np.identity(m)
    AI_len = AI.shape[0] if AI is not None else 0
    bI_len = bI.shape[0] if bI is not None else 0
    assert AI_len == bI_len, "Inequality constraints do not match the dimension of the problem"
    AE_len = AE.shape[0] if AE is not None else 0
    bE_len = bE.shape[0] if bE is not None else 0
    assert AE_len == bE_len, "Equality constraints do not match the dimension of the problem"
    assert AI_len + AE_len == m, "Inequality and equality constraints do not match the dimension of the problem"
    
    
    # Generate A and l,u from AI, bI, AE, bE
    A = np.zeros((m, n))
    l = np.zeros(m)
    u = np.zeros(m)
    
    if AI is not None:
        A[:AI_len] = AI
        l[:AI_len] = -np.inf
        u[:AI_len] = bI
    if AE is not None:
        A[AI_len:] = AE
        l[AI_len:] = bE
        u[AI_len:] = bE
    
    # Do OSQP
    ## Step 1: Initialize
    x = np.zeros(n)
    z = np.zeros(m)
    y = np.zeros(m)
    
    rho = 1
    sigma = 1
    alpha = 1.6
    
    
    ## Step 2: Run OSQP
    for iter in range(MAX_ITER):
        # Step 1: Solve the linear system subproblem
        ## Step 1.1: Compute the matrix and vector
        
        Left_Matrix = np.zeros((n + m, n + m))
        Left_Matrix[:n, :n] = H + rho * I_n # Left-Top
        Left_Matrix[n:, :n] = A # Left-Bottom
        Left_Matrix[:n, n:] = A.T # Right-Top
        Left_Matrix[n:, n:] = -(1/rho) * I_m # Right-Bottom
        
        Right_Vector = np.zeros(n + m)
        Right_Vector[:n] = sigma * x - g
        Right_Vector[n:] = z - (1/rho * y)
        
        sol_subproblem = np.linalg.solve(Left_Matrix, Right_Vector)
        
        x_aul_new = sol_subproblem[:n]
        nu_new = sol_subproblem[n:]
        
        ## Step 1.2: Update z
        z_aul_new = z + (1/rho) * (nu_new - y)
        ## Step 1.3: Update x
        x_new = alpha * x_aul_new + (1 - alpha) * x
        ## Step 1.4: Update z
        z_before_proj = alpha * z_aul_new + (1 - alpha) * z + (1/rho) * y
        z_old = z
        ### TODO: Projection
        #printVec(z_before_proj)
        z_new = projection(z_before_proj, l, u)
        #printVec(z_new)
        ## Step 1.5: Update y
        y_new = y + rho * (alpha * z_aul_new + (1 - alpha) * z_old - z_new)
        
        # Step 2: Check convergence
        # print("X: ", end = "")
        # printVec(x_new)
        # print("Ax: ", end = "")
        # printVec(A @ x_new)
        # print("Z: ", end = "")
        # printVec(z_new)
        r_primal = np.linalg.norm(A @ x_new - z_new)
        r_dual = np.linalg.norm(H @ x_new + g + A.T @ y_new)
        if iter % (MAX_ITER // 10) == 0:
            print(f"Iter: {iter}, r_primal: {r_primal}, r_dual: {r_dual}")
        if r_primal < EPS and r_dual < EPS:
            break
        # Step 3: Update rho and sigma
        x = x_new
        z = z_new
        y = y_new
    
    printVec(x)