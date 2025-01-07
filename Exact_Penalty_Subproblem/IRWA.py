import numpy as np
import os, sys
from Transform_Test_To_Lingo import init_from_config
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
    m1 = 0
    w1 = w2 = None
    if AE is not None and bE is not None:
        r1 = AE @ x_tilde + bE
        m1 = AE.shape[0]
        eps_1 = eps[:m1]
        w1 = 1.0 / np.sqrt(r1**2 + eps_1**2)
    if AI is not None and bI is not None:
        r2 = AI @ x_tilde + bI
        eps_2 = eps[m1:]
        hinge_r2 = np.maximum(r2, 0)
        w2 = 1.0 / np.sqrt(hinge_r2**2 + eps_2**2)
    
    return w1, w2

def IRWA(H, g, AE, bE, AI, bI, eps_init, x_init, 
         eta=0.7, gamma=1/6, M=10000, 
         sigma=1e-4, sigma_prime=1e-8, 
         max_iter=1000):
    """
    Iteratively solve the reweighted QP problem using the IRWA algorithm.
    Parameters
    ----------
    H : ndarray (n x n)
        Hessian matrix of the quadratic problem.
    g : ndarray (n,)
        Gradient vector of the linear part of the objective.
    AE : ndarray (m1 x n)
        Constraints for I1 set.
    bE : ndarray (m1,)
    AI : ndarray (m2 x n)
        Constraints for I2 set.
    bI : ndarray (m2,)
    eps_init : float or ndarray (m1+m2,)
        Initial eps vector.
    x_init : ndarray (n,)
        Initial solution guess.
    eta : float
        Scaling parameter for eps update (0 < eta < 1).
    gamma : float
        Parameter for eps update condition.
    M : float
        Parameter for eps update condition.
    sigma : float
        Tolerance for step-size stopping criterion.
    sigma_prime : float
        Tolerance for relaxation parameter stopping criterion.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x : ndarray (n,)
        The solution after iterations.
    """
    x = x_init.copy()
    
    x_logs = [x]

    A = np.vstack([AE, AI]) if AE is not None and AI is not None else AE if AE is not None else AI
    b = np.concatenate([bE, bI]) if bE is not None and bI is not None else bE if bE is not None else bI

    eps_k = eps_init if not np.isscalar(eps_init) else np.full(A.shape[0], eps_init)
    for _ in range(max_iter):
        # Step 1: Compute weights and solve the reweighted subproblem
        w1, w2 = compute_weights(x, AE, bE, AI, bI, eps_k)
        # w = np.concatenate([w1, w2]) 
        # W = np.diag(w)
        # v_I1 = bE
        # v_I2 = np.maximum(-AI @ x_k, bI) 
        # v = np.concatenate([v_I1, v_I2]) 
        if w1 is not None and w2 is not None:
            W = sp.diags(np.concatenate([w1, w2]), format="csc")
            v = np.concatenate([bE, np.maximum(-AI @ x, bI)])
        elif w1 is not None:
            W = sp.diags(w1, format="csc")
            v = bE
        else:
            W = sp.diags(w2, format="csc")
            v = np.maximum(-AI @ x, bI)

        # Solve the linear system: (H + A^T W A) x = - (g + A^T W v)
        lhs = H + A.T @ W @ A
        rhs = -(g + v.T @ W @ A)
        # x_next = conjugate_gradient(lhs, rhs, x0=x_k)
        x_next = np.linalg.solve(lhs, rhs)
        
        # Step 2: Update eps
        q_k = A @ (x_next - x)
        r_k = (1.0 - v) * (A @ x + b)
        
        # Check condition for eps updating
        lhs_condition = np.abs(q_k)
        rhs_condition = M * ((r_k**2 + eps_k**2)**(0.5 + gamma))
        # print(f"lhs_condition: {lhs_condition}, rhs_condition: {rhs_condition}")    
        
        if np.all(lhs_condition <= rhs_condition):
            eps_next = eta * eps_k
        else:
            eps_next = eps_k
        
        # Step 3: Check stopping criteria
        diff_x = np.linalg.norm(x_next - x, 2)
        diff_eps = np.linalg.norm(eps_k, 2)
        
        if diff_x <= sigma and diff_eps <= sigma_prime:
            break
        
        x = x_next
        eps_k = eps_next
        
        x_logs.append(x)
    
    return x, x_logs

if __name__ == "__main__":
    FILE = "./Tests/00-Easy.txt"
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
    #x, log = ADAL(H, g, A, b, equal_cnt, inequal_cnt, 0.5)
    init_x = np.zeros(n)
    eps = 1e4
    x, log = IRWA(H, g, AE, bE, AI, bI, eps, init_x)
    print(log)
    
    print(f"Solution:", end = " [")
    for i in range(n):
        print(f"{round(x[i], 4)}", end = " ")
    print("]")
    #print(f"Objective: {0.5 * x.T @ H @ x + g @ x}")
    print(f"Objective: {eval_exact_penalty(H, g, A, b, equal_cnt, inequal_cnt, x)}")
    for i in range(m):
        if (i < equal_cnt):
            print(f"Equality {i}: {A[i] @ x + b[i]}")
        else:
            print(f"Inequality {i}: {A[i] @ x + b[i]}")
    
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
    
    # Log
    with open("ADAL.log", "w") as f:
        f.write(f"Init: {log[0]}\n")
        for i in range(1, len(log)):
            f.write(f"Iter {i}: {log[i]}\n")