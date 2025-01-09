import numpy as np
import os, sys

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
    init_x: np.ndarray = None,
    show_log: bool = False
):
    n = H.shape[0]
    m = ineq_cnt + eq_cnt
    assert m == A.shape[0], "Inequality and equality constraints do not match the dimension of the problem"
    
    # Initialize
    if init_x is not None:
        x = init_x
    else:
        x = np.zeros(n)
    u = np.zeros(m)
    p = np.zeros(m)
    
    x_log = [x]
    if show_log:
        print(f"Initialize, Objective: {round(eval_exact_penalty(H, g, A, b, eq_cnt, ineq_cnt, x), 4)}, loss = 0.000000 & {round(np.linalg.norm(A @ x + b - p), 6)}")
    
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
                if s[i] <= 0:
                    proj_pt = s[i]
            dist = np.linalg.norm(s[i] - proj_pt)
            if (dist <= mu):
                p_new[i] = proj_pt
            else:
                p_new[i] = s[i] - mu / dist * (s[i] - proj_pt)
        ## Step 1.2: optimize x
        ### 操，正经人谁玩梯度下降啊
        left_Matrix = H + 1 / mu * A.T @ A
        right_Vector = g + 1 / mu * A.T @ (b - p_new + mu * u)
        x_new = np.linalg.solve(left_Matrix, -right_Vector)
        
        # Step 2: Optimize u
        u_new = u + 1 / mu * (A @ x_new + b - p_new)
        
        # Step 3: Check convergence
        loss_1 = np.linalg.norm(x_new - x)
        loss_2 = np.linalg.norm(A @ x_new + b - p_new)
        
        if loss_1 < sigma and loss_2 < sigmapp:
            if show_log:
                print(f"\nAlgorithm Converged at iter {iter}: loss 1 = {loss_1}, loss 2 = {loss_2}")
            break
        # Step 4: Update Variables
        x = x_new
        p = p_new
        u = u_new
        
        x_log.append(x)
        if show_log:
            print(f"Iter {iter:5}, Objective: {round(eval_exact_penalty(H, g, A, b, eq_cnt, ineq_cnt, x), 4)}, loss = {round(loss_1, 6):.6f} & {round(loss_2, 6):.6f}")
    return x, x_log

if __name__ == "__main__":
    from Transform_Test_To_Lingo import init_from_config
    FILE = "./Exact_Penalty_Test/00-Easy.txt"
    if (len(sys.argv) > 1):
        FILE = sys.argv[1]
    
    n, m, H, g, AI, bI, AE, bE, ref, ref_val = init_from_config(FILE)
    
    
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
    
    #print(A, b)
    x_res, log = ADAL(H, g, A, b, equal_cnt, inequal_cnt, 0.5, show_log = True)
    
    print(f"Solution:", end = " [")
    for i in range(n):
        print(f"{round(x_res[i], 4)}", end = " ")
    print("]")
    #print(f"Objective: {0.5 * x.T @ H @ x + g @ x}")
    print(f"Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, x_res)}")
    for i in range(m):
        if (i < equal_cnt):
            print(f"Equality {i}: {A[i] @ x_res + b[i]}")
        else:
            print(f"Inequality {i}: {A[i] @ x_res + b[i]}")
    
    if (ref is not None) and (ref_val is not None):
        ref_x = np.array(ref)
        ref_val = ref_val
        print(f"Reference: {ref_x}")
        #print(f"Reference Objective: {ref_obj}")
        print(f"Reference Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, ref_x)}")
        print(f"Reference Objective: {ref_val}")
        for i in range(m):
            if (i < equal_cnt):
                print(f"Equality {i}: {A[i] @ ref_x + b[i]}")
            else:
                print(f"Inequality {i}: {A[i] @ ref_x + b[i]}")
    
        diff = np.linalg.norm(x_res - ref_x)
        print(f"Distance: {diff}")
        if diff < 1e-4:
            print("========== Test Passed ==========")
    # Log
    with open("ADAL.log", "w") as f:
        f.write(f"Init: {log[0]}\n")
        for i in range(1, len(log)):
            f.write(f"Iter {i}: {log[i]}\n")