import numpy as np
import sys
from utils import init_from_config, check_feasible
import scipy.sparse as sp
from tqdm import tqdm, trange
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
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "./Testcases/reference.txt"
    n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_file)

    
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
    
    # Initialize
    x = np.zeros(n)
    p = np.zeros(m)
    nu = np.zeros((m))
    Const = 2
    
    # Main Loop
    print("\nIterating...")
    for iter in trange(MAX_ITER):
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
            
            dist = dist * (iter / MAX_ITER * 1.5 + 1)
            if (dist <= Const):
                p_new[i] = Project_si
            else:
                term = Const / dist * (si - Project_si)
                p_new[i] = si - term
        
        ## Step 1.2: Minimize x
        ### Gradient Descent
        x_new = x
        for i in range(1000):
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
            #print("Converged")
            #if check_feasible(x_new, AI, bI, "inequ", printResult = False) and check_feasible(x_new, AE, bE, "equ", printResult = False):
            #    break
            #elif iter >= 0.8 * MAX_ITER:
            #    break
            break
        # Update
        #print(f"Iter {iter}: x: {x_new}, p: {p_new}")
        x = x_new
        p = p_new
        nu = nu_new
        
        optimize_process = round(iter / MAX_ITER * 100)
        
    
    print("Algorithm Finished.")
    print("x: ", end = "")
    printVec(x)
    print("Objective Value: ", end = "")
    print(round(0.5 * x.T @ H @ x + g.T @ x, 4))
    
    # Feasibility Check
    if (AI is not None) and (bI is not None):
        check_feasible(x, AI, -bI, "inequ")
    if (AE is not None) and (bE is not None):
        check_feasible(x, AE, -bE, "equ")