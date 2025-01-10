import numpy as np
import os, sys
from Exact_Penalty_Subproblem.ADAL import ADAL
from Exact_Penalty_Subproblem.IRWA import IRWA
from utils import init_from_config, check_feasible, printVec, eval_penalty
from reference import reference

def QP_solver(AE: np.ndarray, AI: np.ndarray, bE: np.ndarray, bI: np.ndarray, 
              g: np.ndarray, H: np.ndarray,
              solver: str = "ADAL"):
    # Dimension Check
    AI_len = AI.shape[0] if AI is not None else 0
    bI_len = bI.shape[0] if bI is not None else 0
    assert AI_len == bI_len, "Inequality constraints do not match the dimension of the problem"
    AE_len = AE.shape[0] if AE is not None else 0
    bE_len = bE.shape[0] if bE is not None else 0
    assert AE_len == bE_len, "Equality constraints do not match the dimension of the problem"
    eq_cnt, ineq_cnt = AE_len, AI_len
    
    n = H.shape[0]
    m = eq_cnt + ineq_cnt
    assert eq_cnt + ineq_cnt == m, "Inequality and equality constraints do not match the dimension of the problem"
    
    
    # x for iteration
    x = np.zeros(n)
    
    M_eq = 1
    # np.diag(
    #     np.ones(eq_cnt)
    # )
    M_ineq = 1
    # np.diag(
    #     np.ones(ineq_cnt)
    # )
    # Generate A and l,u from AI, bI, AE, bE
    A = np.zeros((m, n))
    b = np.zeros(m)
    
    if (AE is not None) and (bE is not None):
        A[:eq_cnt] = AE
        b[:eq_cnt] = bE
    if (AI is not None) and (bI is not None):
        A[eq_cnt:] = AI
        b[eq_cnt:] = bI
    
    A_iter, b_iter = A.copy(), b.copy()
    AE_iter, AI_iter = A_iter[:eq_cnt], A_iter[eq_cnt:]
    bE_iter, bI_iter = b_iter[:eq_cnt], b_iter[eq_cnt:]
    
    for subproblem_iter in range(1000):
        # Solve Subproblem
        if (solver == "ADAL"):
            x_new, _ = ADAL(
                H, g, A_iter, b_iter, eq_cnt, ineq_cnt, 0.5, init_x = x
            )
        elif (solver == "IRWA"):
            x_new, _ = IRWA(
                H, g, AE_iter, bE_iter, AI_iter, bI_iter, 
                1e4, x_init = x, eta = 0.9, max_iter= 1000
            )
        else:
            raise ValueError(f"Solver {solver} not supported.")
        
        # Check Feasibility
        feasible = True
        if (AI is not None) and (bI is not None):
            ineq_res = check_feasible(x_new, AI, bI, "inequ", optimal_check_eps = 1e-5 , printResult=False)
            feasible = feasible and ineq_res
        if (AE is not None) and (bE is not None):
            eq_res = check_feasible(x_new, AE, bE, "equ", optimal_check_eps = 1e-5, printResult=False)
            feasible = feasible and eq_res
        
        delta_x = np.linalg.norm(x_new - x) / n
        print("-" * 50)
        print(f"Subproblem Iter {subproblem_iter}, Objective: {round(0.5 * x_new.T @ H @ x_new + g @ x_new, 4)}, Loss: {delta_x}, Feasible: {feasible}")
        print("x: ", end = "")
        printVec(x_new[:20])
        # Update x
        x = x_new
        
        if (feasible) and (delta_x < 1e-6):
            print("========== Algorithm Converged ==========")
            return x
        
        all_feasible = True
        if (eq_cnt) and not check_feasible(x_new, AE, bE, "equ", optimal_check_eps=1e-5, printResult=False):
            all_feasible = False
            #penalty_eq_vec = eval_penalty(x_new, AE, bE, "equ")
            penalty_eq_vec = eval_penalty(AE, bE, x_new, "equ")
            penalty_eq = np.max(penalty_eq_vec)
            if (penalty_eq > n):
                penalty_eq = n
            if (penalty_eq < 0.1):
                penalty_eq = 0.1
        else:
            penalty_eq = 0
        if (ineq_cnt) and not check_feasible(x_new, AI, bI, "inequ", optimal_check_eps=1e-5, printResult=False):
            all_feasible = False
            #penalty_ineq_vec = eval_penalty(x_new, AI, bI, "inequ")
            penalty_ineq_vec = eval_penalty(AI, bI, x_new, "inequ")
            penalty_ineq = np.max(penalty_ineq_vec)
            if (penalty_ineq > n):
                penalty_ineq = n
            if (penalty_ineq < 0.1):
                penalty_ineq = 0.1
        else:
            penalty_ineq = 0
        # Update A, b: Multiply by M_eq and M_ineq
        A_iter_eq = A[:eq_cnt] * M_eq
        A_iter_ineq = A[eq_cnt:] * M_ineq
        A_iter = np.concatenate([A_iter_eq, A_iter_ineq], axis=0)
        AE_iter = A_iter[:eq_cnt]
        AI_iter = A_iter[eq_cnt:]
        
        
        b_iter_eq = b[:eq_cnt] * M_eq
        b_iter_ineq = b[eq_cnt:] * M_ineq
        b_iter = np.concatenate([b_iter_eq, b_iter_ineq], axis=0)
        bE_iter = b_iter[:eq_cnt]
        bI_iter = b_iter[eq_cnt:]
        # Update M_eq and M_ineq
        ## Compute Scaling Factor. 
        ## We want M to be large than n
        #scale_eq = np.maximum(1, 2 * n / M_eq)
        #scale_ineq = np.maximum(1, 2 * n / M_ineq)
        M_eq = M_eq * np.exp(penalty_eq / n) #* scale_eq
        M_ineq = M_ineq * np.exp(penalty_ineq / n) #* scale_ineq
        print(f"Penalty Eq: {round(penalty_eq, 4)}, Penalty Ineq: {round(penalty_ineq, 4)}")
        print(f"M_eq: {round(M_eq, 4)}, M_ineq: {round(M_ineq, 4)}")

        if (M_eq > 100 * n * n) or (M_ineq > 100 * n * n):
            raise ValueError("Penalty too large. Stopping.")
    return x


if __name__ == "__main__":
    SOLVER = "ADAL"
    # SOLVER = "IRWA"
    if len(sys.argv) > 2:
        SOLVER = sys.argv[2]
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "./Testcases/reference.txt"
    n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_file)

    # Check Dimensions
    I_n = np.identity(n)
    I_m = np.identity(m)

    print("==================== QP_Solver ====================")
    x = QP_solver(AE, AI, bE, bI, g, H, solver = SOLVER)
    
    print("x: ", end = "")
    printVec(x[:20])
    our_objctive = 1/2 * x.T @ H @ x + g @ x
    print("Objective Value: ", round(our_objctive, 4))
        
    if (AI is not None) and (bI is not None):
        print("*", end=" ")
        check_feasible(x, AI, bI, "inequ", optimal_check_eps=1e-4)
    else:
        print("* No inequality constraints.")
    if (AE is not None) and (bE is not None):
        print("*", end=" ")
        check_feasible(x, AE, bE, "equ", optimal_check_eps=1e-4)
    else:
        print("* No equality constraints.")

    ans = reference(cfg_file)
    
    ref_objective = 1/2 * ans.T @ H @ ans + g @ ans
    print("LOSS: ", np.abs(our_objctive - ref_objective) / (np.abs(ref_objective) * n))
    if (np.abs(our_objctive - ref_objective) / (np.abs(ref_objective) * n) < 1e-6):
        print("========== Test Passed! ==========")
    else:
        print("========== Test Failed! ==========")
        