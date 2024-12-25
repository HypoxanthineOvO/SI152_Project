import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sp
from utils import init_from_config, check_feasible, printVec
from reference import reference

MAX_ITER = 10000
EPS = 1e-5

def compute_weights(x_tilde, AE, bE, AI, bI, eps):
    """
    Compute the weights w_i for the sets I1 and I2 based on the given formula.
    Refer to the previously defined formula:
    
    For i ∈ I1:
    w_i = (( (a_i^T x_tilde + b_i)^2 + eps_i^2 ))^{-1/2}
    
    For i ∈ I2:
    w_i = (( max(a_i^T x_tilde + b_i, 0)^2 + eps_i^2 ))^{-1/2}
    """
    r1 = AE @ x_tilde + bE
    r2 = AI @ x_tilde + bI
    
    m1 = AE.shape[0]
    eps_1 = eps[:m1]
    eps_2 = eps[m1:]
    
    w1 = 1.0 / np.sqrt(r1**2 + eps_1**2)
    hinge_r2 = np.maximum(r2, 0)
    w2 = 1.0 / np.sqrt(hinge_r2**2 + eps_2**2)
    
    return w1, w2

def IRWA(H, g, AE, bE, AI, bI, eps_init, x_init, 
         eta=0.7, gamma=1/6, M=10000, 
         sigma=1e-5, sigma_prime=1e-8, 
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
    x_k = x_init.copy()
    eps_k = eps_init if not np.isscalar(eps_init) else np.full(AE.shape[0] + AI.shape[0], eps_init)
    
    # Stack A and b for convenience
    A = np.vstack([AE, AI])
    b = np.concatenate([bE, bI])
    
    for k in range(max_iter):
        # Step 1: Compute weights and solve the reweighted subproblem
        w1, w2 = compute_weights(x_k, AE, bE, AI, bI, eps_k)
        w = np.concatenate([w1, w2])
        W = np.diag(w)
        v_I1 = bE
        v_I2 = np.maximum(-AI @ x_k, bI)
        v = np.concatenate([v_I1, v_I2])

        # Solve the linear system: (H + A^T W A) x = - (g + A^T W v)
        lhs = H + A.T @ W @ A
        rhs = -(g + v.T @ W @ A)
        # x_next = conjugate_gradient(lhs, rhs, x0=x_k)
        x_next = np.linalg.solve(lhs, rhs)
        
        # Step 2: Update eps
        q_k = A @ (x_next - x_k)
        r_k = (1.0 - v) * (A @ x_k + b)
        
        # Check condition for eps updating
        lhs_condition = np.abs(q_k)
        rhs_condition = M * ((r_k**2 + eps_k**2)**(0.5 + gamma))
        # print(f"lhs_condition: {lhs_condition}, rhs_condition: {rhs_condition}")    
        
        if np.all(lhs_condition <= rhs_condition):
            eps_next = eta * eps_k
        else:
            eps_next = eps_k
        
        # Step 3: Check stopping criteria
        diff_x = np.linalg.norm(x_next - x_k, 2)
        diff_eps = np.linalg.norm(eps_k, 2)
        
        if diff_x <= sigma and diff_eps <= sigma_prime:
            return x_next
        
        x_k = x_next
        eps_k = eps_next
        
    return x_k

def QP_solver(AE: np.ndarray, AI: np.ndarray, bE: np.ndarray, bI: np.ndarray, g: np.ndarray, H: np.ndarray, n: int, m: int):
    x_k = np.zeros(n)
    penalty = 1
    penalties = []
    AE_original = AE.copy()
    AI_original = AI.copy()
    bE_original = bE.copy()
    bI_original = bI.copy()
    eps = 1e4
    lst1 = []
    lst2 = []
    for k in range(10000):
        AE = AE_original * penalty
        AI = AI_original * penalty
        bE = bE_original * penalty
        bI = bI_original * penalty
        
        x_next = IRWA(H, g, AE, bE, AI, bI, eps, x_k)

        
        if abs(AE_original @ x_next + bE_original) <= EPS and np.all(AI_original @ x_next + bI_original <= EPS) and np.all(np.abs(x_next - x_k)<= EPS):
            break
        
        penalties.append(penalty)
        penalty *= (np.exp(-k / 10) + 1)
        # penalty *= 1.1
        lst1.append(np.mean(x_next - x_k))
        lst2.append(np.exp(-k / 10) + 1)
        x_k = x_next
        value = 1 / 2 * x_k.T @ H @ x_k + g @ x_k

        print(f"The {k+1}th iter: {x_k} and function value is {value}, penalty is {penalty}")
        
    # plt.figure(figsize = (15, 6))
    # plt.subplot(1, 3, 1)
    # plt.plot(lst1)
    # plt.subplot(1, 3, 2)
    # plt.plot(penalties)
    # plt.xlabel('Iteration')
    # plt.ylabel('Penalty')
    # plt.subplot(1, 3, 3)
    # plt.plot(lst2)
    # plt.title('Penalty Variation Over Iterations')
    # plt.show()
    
    return x_k


# Simple QP Problem Example
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "./Testcases/reference.txt"
    n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_file)
    
    x = QP_solver(AE, AI, bE, bI, g, H, n, m)
    print("x:", end=" ")
    printVec(x)
    print("Objective Value: ", round(1/2 * x.T@H@x + g @ x, 4))
    
    if (AI is not None) and (bI is not None):
        check_feasible(x, AI, bI, "inequ", optimal_check_eps=1e-4)
    if (AE is not None) and (bE is not None):
        check_feasible(x, AE, bE, "equ", optimal_check_eps=1e-4)
        
    ans = reference(cfg_file)


    
    
    
    


