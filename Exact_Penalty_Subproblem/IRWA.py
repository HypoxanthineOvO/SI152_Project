import numpy as np
import os, sys
import scipy.sparse as sp

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

def compute_weights(x_tilde, AE, bE, AI, bI, eps):
    """
    Compute the weights w_i for the sets I1 and I2 based on the given formula.
    Refer to the previously defined formula:
    
    For i ∈ I1:
    w_i = (( (a_i^T x_tilde + b_i)^2 + eps_i^2 ))^{-1/2}
    
    For i ∈ I2:
    w_i = (( max(a_i^T x_tilde + b_i, 0)^2 + eps_i^2 ))^{-1/2}
    """
    equ_cnt = AE.shape[0] if AE is not None else 0
    w1 = None
    w2 = None
    if AE is not None and bE is not None:
        r1 = AE @ x_tilde + bE
        eps_1 = eps[:equ_cnt]
        w1 = 1.0 / np.sqrt(r1**2 + eps_1**2)
    if AI is not None and bI is not None:
        r2 = AI @ x_tilde + bI
        eps_2 = eps[equ_cnt:]
        hinge_r2 = np.maximum(r2, 0)
        w2 = 1.0 / np.sqrt(hinge_r2**2 + eps_2**2)
    
    return w1, w2

def IRWA(H, g, AE, bE, AI, bI, eps_init, init_x, 
         eta = 0.995, gamma = 1/6, M = 10000, 
         sigma = 1e-4, sigma_prime = 1e-8, 
         max_iter = 100000):
    x = init_x.copy()
    
    x_logs = [x]

    A = np.vstack([AE, AI]) if AE is not None and AI is not None else AE if AE is not None else AI
    b = np.concatenate([bE, bI]) if bE is not None and bI is not None else bE if bE is not None else bI

    #eps_k = eps_init if not np.isscalar(eps_init) else np.full(A.shape[0], eps_init)
    if np.isscalar(eps_init):
        eps_k = np.full(A.shape[0], eps_init)
    else:
        eps_k = eps_init.copy()
    
    for _ in range(max_iter):
        # Step 1: Compute weights and solve the reweighted subproblem
        w1, w2 = compute_weights(x, AE, bE, AI, bI, eps_k)
        
        if w1 is not None and w2 is not None:
            #W = sp.diags(np.concatenate([w1, w2]), format="csc")
            W = np.diag(np.concatenate([w1, w2]))
            v = np.concatenate([bE, np.maximum(-AI @ x, bI)])
        elif w1 is not None:
            #W = sp.diags(w1, format="csc")
            W = np.diag(w1)
            v = bE
        else:
            #W = sp.diags(w2, format="csc")
            W = np.diag(w2)
            v = np.maximum(-AI @ x, bI)

        # Solve the linear system: (H + A^T W A) x + (g + A^T W v) = 0
        lhs = H + A.T @ W @ A
        rhs = -(g + A.T @ W @ v)
        # x_next = conjugate_gradient(lhs, rhs, x0=x_k)
        x_next = np.linalg.solve(lhs, rhs)
        
        # Step 2: Update eps
        q_k = A @ (x_next - x)
        r_k = (1.0 - v) * (A @ x + b)
        
        # Check condition for eps updating
        lhs_condition = np.abs(q_k)
        rhs_condition = M * ((r_k**2 + eps_k**2)**(0.5 + gamma)) 

        if np.all(lhs_condition <= rhs_condition):
            eps_next = eta * eps_k
        else:
            eps_next = eps_k
        
        # Step 3: Check stopping criteria
        diff_x = np.linalg.norm(x_next - x, 2)
        diff_eps = np.linalg.norm(eps_k, 2)
        
        if (diff_x <= sigma) and (diff_eps <= sigma_prime):
            break
        
        x = x_next
        eps_k = eps_next
        
        x_logs.append(x)
    
    return x, x_logs

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
    
    
    init_x = np.zeros(n)
    eps = np.ones(m) * 2e3
    x, log = IRWA(
        H, g, AE, bE, AI, bI, eps, init_x,
        eta = 0.997, gamma = 1/6, M = 1e4, sigma = 1e-4, sigma_prime = 1e-4, max_iter = 100000,
    )
    #print(log)
    
    print(f"Solution:", end = "\t[")
    for i in range(n):
        print(f"{round(x[i], 4)}", end = " ")
    print("]")
    print(f"Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, x)}")
    
    if m < 10:
        for i in range(m):
            if (i < equal_cnt):
                print(f"Equality {i}: {A[i] @ x + b[i]}")
            else:
                print(f"Inequality {i}: {A[i] @ x + b[i]}")
        
    if (ref is not None) and (ref_val is not None):
        ref_x = np.array(ref)
        ref_val = ref_val
        print(f"Reference:", end = "\t[")
        for i in range(n):
            print(f"{round(ref_x[i], 4)}", end = " ")
        print("]")
        #print(f"Reference Objective: {ref_obj}")
        print(f"Reference Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, ref_x)}")
        print(f"Reference Objective: {ref_val}")
        
        if m < 10:
            for i in range(m):
                if (i < equal_cnt):
                    print(f"Equality {i}: {A[i] @ ref_x + b[i]}")
                else:
                    print(f"Inequality {i}: {A[i] @ ref_x + b[i]}")
    
        diff = np.linalg.norm(x - ref_x) / n
        print(f"Distance: {diff}")
        if diff < 1e-3:
            print("========== Test Passed ==========")
        else:
            print("========== Test Failed ==========")
            diff_vec = np.abs(x - ref_x)
            # Get the 10 largest difference and its index
            diff_idx = np.argsort(diff_vec)[-1:-10:-1]
            print("Largest Difference Index: ", diff_idx)
            print("Largest Difference: ", diff_vec[diff_idx])
    # Log
    with open("IRWA.log", "w") as f:
        f.write(f"Init: {log[0]}\n")
        for i in range(1, len(log)):
            f.write(f"Iter {i}: {log[i]}\n")