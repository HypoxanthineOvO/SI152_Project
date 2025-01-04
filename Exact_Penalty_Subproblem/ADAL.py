import numpy as np
import os, sys
from Transform_Test_To_Lingo import init_from_config

def eval_exact_penalty(
    H: np.ndarray, g: np.ndarray, 
    A: np.ndarray, b: np.ndarray,
    equal_cnt: int, ineq_cnt: int, 
    x: np.ndarray,):
    val = 0.5 * x.T @ H @ x + g @ x
    for i in range(equal_cnt):
        val += np.abs(A[i] @ x + b[i])
    for i in range(ineq_cnt):
        val += np.maximum(0, A[i + equal_cnt] @ x + b[i + equal_cnt])
    return val

def ADAL(
    H: np.ndarray, g: np.ndarray,
    A: np.ndarray, b: np.ndarray, eq_cnt: int, ineq_cnt: int,
    mu: float, sigma: float = 1e-3, sigmapp: float = 1e-3,
):
    n = H.shape[0]
    m = ineq_cnt + eq_cnt
    assert m == A.shape[0], "Inequality and equality constraints do not match the dimension of the problem"
    
    # Initialize
    x = np.zeros(n)
    u = np.zeros(m)
    p = np.zeros(m)
    
    for iter in range(1000):
        # Step 1: Solve the augmented Lagrangian subproblem for (x^{k+1}, p^{k+1})
        ## Step 1.1: optimize p
        s = A @ x + b + mu * u
        p_new = np.zeros(m)
        for i in range(m):
            if (i < eq_cnt): # i < eq_cnt, equality constraints
                dist = np.abs(s[i])
                ## Project s[i] -> 0
                if (dist <= mu):
                    p_new[i] = 0
                else:
                    p_new[i] = s[i] - mu / dist * (s[i] - 0)
            
            else: # i >= eq_cnt, inequality constraints
                dist = np.maximum(s[i], 0)
                if (dist <= mu):
                    p_new[i] = np.maximum(0, s[i])
                else:
                    p_new[i] = s[i] - mu / dist * (s[i] - np.maximum(0, s[i]))
        
        ## Step 1.2: optimize x
        x_new = x
        for x_opt_iter in range(1000):
            grad = H @ x_new + g + 1 / mu * (A.T @ A @ x_new + A.T @ (b - p_new + mu * u))
            x_new = x_new - grad * 0.01
            if np.linalg.norm(grad) < sigma:
                #print(f"Converged at iter {x_opt_iter}")
                break
        
        # Step 2: Optimize u
        u_new = u + 1 / mu * (A @ x_new + b - p_new)
        
        # Step 3: Check convergence
        loss_1 = np.linalg.norm(x_new - x)
        loss_2 = np.linalg.norm(A @ x_new + b - p_new)
        
        if loss_1 < sigma and loss_2 < sigmapp:
            print(f"Algorithm Converged at iter {iter}")
            break
        # Step 4: Update Variables
        x = x_new
        p = p_new
        u = u_new
        
        print(f"Iter {iter}, x = {x}")
    return x

if __name__ == "__main__":
    # n = 2
    # m = 3
    # H, g = np.array([
    #     [1, 0],
    #     [0, 1]
    # ]), np.array([1, 0])
    # A, b = np.zeros((m, n)), np.zeros(m)
    
    # # Equalities first!
    # equal_cnt = 1
    # inequal_cnt = 2
    # # 2x1+2x2 = 3 => Ax + (-3) = 0
    # A[0] = np.array([2, 2])
    # b[0] = -3
    # # x1 <= 4 => Ax + (-4) <= 0
    # A[1] = np.array([1, 0])
    # b[1] = -4
    # # x2 <= 2 => Ax + (-2) <= 0
    # A[2] = np.array([0, 1])
    # b[2] = -2
    FILE = "./Tests/00-Easy.txt"
    if (len(sys.argv) > 1):
        FILE = sys.argv[1]
    
    n, m, H, g, AI, bI, AE, bE = init_from_config(FILE)
    
    equal_cnt = AE.shape[0] if AE is not None else 0
    inequal_cnt = AI.shape[0] if AI is not None else 0
    
    A = np.zeros((m, n))
    b = np.zeros(m)
    
    if AE is not None:
        A[:equal_cnt] = AE
        b[:equal_cnt] = bE
    if AI is not None:
        A[equal_cnt:] = AI
        b[equal_cnt:] = bI
    
    x = ADAL(H, g, A, b, equal_cnt, inequal_cnt, 1)
    
    print(f"Solution:", end = " ")
    for i in range(n):
        print(f"{round(x[i], 4)}", end = " ")
    print()
    #print(f"Objective: {0.5 * x.T @ H @ x + g @ x}")
    print(f"Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, x)}")
    for i in range(m):
        if (i < equal_cnt):
            print(f"Equality {i}: {A[i] @ x + b[i]}")
        else:
            print(f"Inequality {i}: {A[i] @ x + b[i]}")
    
    ref_x = np.array([0, 97.7261, 0.103e-5, 0.888139])
    ref_obj = 0.5 * ref_x.T @ H @ ref_x + g @ ref_x
    print(f"Reference: {ref_x}")
    #print(f"Reference Objective: {ref_obj}")
    print(f"Reference Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, ref_x)}")
    for i in range(m):
        if (i < equal_cnt):
            print(f"Equality {i}: {A[i] @ ref_x + b[i]}")
        else:
            print(f"Inequality {i}: {A[i] @ ref_x + b[i]}")