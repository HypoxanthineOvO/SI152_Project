import numpy as np
import scipy.sparse as sp

# Constants
MAX_ITER = 1000
EPS = 1e-5

# Denotions
"""
AE & bE: Equality constraints. AE x + bE = 0
AI & bI: Inequality constraints. AI x + bI <= 0
"""

# Util Functions
def printVec(x: np.ndarray, num_digits: int = 4):
    for i in range(x.shape[0]):
        rounded_x = round(x[i], num_digits)
        print("{:.4f}".format(rounded_x), end = " ")
    print()


if __name__ == "__main__":
    # n = 2
    # m = 5
    
    # # Convert to scipy sparse matrix
    # g = np.array([-2, -5])
    # AI = np.array([
    #     [-1, 2],
    #     [1, 2],
    #     [1, -2],
    #     [-1, 0],
    #     [0, -1]
    # ])
    # bI = -np.array([2, 6, 2, 0, 0])
    # AE = None
    # bE = None

    n = 3
    m = 4
    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    H = M.T @ M  # this is a positive definite matrix
    g = np.array([3.0, 2.0, 3.0]) @ M
    AI = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
    bI = np.array([3.0, 2.0, -2.0])
    AE = np.array([
        [1.0, 1.0, 1.0]
    ])
    bE = np.array([1.0])


    # Check Dimensions
    I_n = np.identity(n)
    I_m = np.identity(m)
    
    H = 2 * I_n
    num_equ = 0 if AE is None else AE.shape[0]
    num_inequ = 0 if AI is None else AI.shape[0]
    assert num_equ+num_inequ == m, "AE and AI should have m rows"
    
    A = np.zeros((m, n))
    if AI is not None:
        A[:num_inequ, :] = AI
    if AE is not None:
        A[num_inequ:, :] = AE
    b = np.zeros(m)
    if bI is not None:
        b[:num_inequ] = bI
    if bE is not None:
        b[num_inequ:] = bE
    
    print("A: \n", A)
    print("B: \n", b)
    
    # Initialize
    x = np.zeros(n)
    p = np.zeros(m)
    nu = np.zeros((m))
    Const = 1.1
    
    
    # Main Loop
    print("\nIterating...")
    for iter in range(MAX_ITER):
        # Step 1
        ## Step 1.1: Minimize p. It can be explicitly solved.
        p_new = np.zeros_like(p)
        for i in range(m):
            si = A[i] @ x + b[i] + Const * nu[i]
            if (i < num_inequ): # Inequality Constraints
                Project_si = np.maximum(si, 0)
                dist = np.linalg.norm(si - Project_si)
            else: # Equality Constraints
                Project_si = 0
                dist = np.linalg.norm(si - Project_si)
            
            if (dist <= Const):
                p_new[i] = Project_si
            else:
                term = Const / dist * (si - Project_si)
                p_new[i] = si - term
        
        ## Step 1.2: Minimize x
        ### Gradient Descent
        x_new = x
        for i in range(256):
            grad_x = g + H @ x_new + 1/Const * (A.T @ A) @ x_new + 1/Const * A.T @ (b - p_new + Const * nu)
            x_new = x_new - 0.1 / (i+1) * grad_x
            if np.linalg.norm(grad_x) < 1e-5:
                break
        
        ## Step 1.3: Minimize nu
        nu_new = nu + 1/Const * (A @ x_new + b - p_new)
        
        # Step 3
        delta_1 = np.linalg.norm(x_new - x)
        delta_2 = 0
        for i in range(m):
            delta_2_i = np.abs(A[i] @ x_new + b[i] - p_new[i])
            delta_2 = max(delta_2, delta_2_i)
        if delta_1 < EPS and delta_2 < EPS:
            print("Converged")
            break
        # Update
        print(f"Iter {iter}: x: {x_new}, p: {p_new}")
        x = x_new
        p = p_new
        nu = nu_new
    
    print("Algorithm Finished.")
    print("x: ", end = "")
    printVec(x)
    print("p: ", end = "")
    printVec(p)
    print("Ax + b - p: ", end = "")
    printVec(A @ x + b - p)
    print("nu: ", end = "")
    printVec(nu)
    
    
    # Feasibility Check
    print("Feasibility Check")
    ## Inequality Constraints: AI, bI: print AIx+bI
    if AI is not None:
        
        print("AIx+bI <= 0: ", end="")
        printVec(AI @ x + bI)
        bool_val = ((AI @ x + bI) <= 0)
        for i in range(num_inequ):
            print(bool_val[i], end = " ")
        print()
    ## Equality Constraints: AE, bE: print AEx+bE
    if AE is not None:
        print("AEx+bE: ", np.abs((AE @ x + bE)) <= 1e-3)