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
    mu: float, sigma: float = 1e-5, sigmapp: float = 1e-5,
):
    n = H.shape[0]
    m = ineq_cnt + eq_cnt
    assert m == A.shape[0], "Inequality and equality constraints do not match the dimension of the problem"
    
    # Initialize
    x = np.zeros(n)
    u = np.zeros(m)
    p = np.zeros(m)
    
    for iter in range(10000):
        # Step 1: Solve the augmented Lagrangian subproblem for (x^{k+1}, p^{k+1})
        ## Step 1.1: optimize p
        s = A @ x + b + mu * u
        p_new = np.zeros(m)
        for i in range(m):
            if (i < eq_cnt): # i < eq_cnt, equality constraints
                proj_pt = 0
            else: # i >= eq_cnt, inequality constraints
                proj_pt = 0
                if s[i] < 0:
                    proj_pt = s[i]
            dist = np.linalg.norm(s[i] - proj_pt)
            if (dist <= mu):
                p_new[i] = proj_pt
            else:
                p_new[i] = s[i] - mu / dist * (s[i] - proj_pt)
        ## Step 1.2: optimize x
        x_new = x
        for x_opt_iter in range(10000):
            grad = H @ x_new + g + 1 / mu * (A.T @ A @ x_new + A.T @ (b - p_new + mu * u))
            x_new = x_new - grad * 0.01
            if np.linalg.norm(grad) < sigma:
                #print(f"Converged at iter {x_opt_iter}")
                break
        grad = H @ x_new + g + 1 / mu * (A.T @ A @ x_new + A.T @ (b - p_new + mu * u))
        if np.linalg.norm(grad) > sigma:
            raise Exception(f"Failed to converge at iter {iter}")
        
        # Step 2: Optimize u
        u_new = u + 1 / mu * (A @ x_new + b - p_new)
        
        # Step 3: Check convergence
        loss_1 = np.linalg.norm(x_new - x)
        loss_2 = np.linalg.norm(A @ x_new + b - p_new)
        
        if loss_1 < sigma and loss_2 < sigmapp:
            print(f"\nAlgorithm Converged at iter {iter}: loss 1 = {loss_1}, loss 2 = {loss_2}")
            break
        # Step 4: Update Variables
        x = x_new
        p = p_new
        u = u_new
        
        print(f"Iter {iter:5}, loss = {round(loss_1, 6):.6f} & {round(loss_2, 6):.6f}")
    return x

if __name__ == "__main__":
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
    
    print(A, b)
    x = ADAL(H, g, A, b, equal_cnt, inequal_cnt, 0.5)
    
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
    
    #ref_x = np.array([0.2323276, 0.000000 , 0.2177959E-08, 131.0391])
    # ref_x = np.array([0.25, 1.25])
    # print(f"Reference: {ref_x}")
    # #print(f"Reference Objective: {ref_obj}")
    # print(f"Reference Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, ref_x)}")
    # for i in range(m):
    #     if (i < equal_cnt):
    #         print(f"Equality {i}: {A[i] @ ref_x + b[i]}")
    #     else:
    #         print(f"Inequality {i}: {A[i] @ ref_x + b[i]}")