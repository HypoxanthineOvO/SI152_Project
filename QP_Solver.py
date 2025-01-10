import numpy as np
import os, sys
from Exact_Penalty_Subproblem.ADAL import ADAL
from Exact_Penalty_Subproblem.IRWA import IRWA
from utils import init_from_config, check_feasible, printVec
from reference import reference

def eval_M(
    x: np.ndarray, A: np.ndarray, b: np.ndarray,
    eq_cnt: int, ineq_cnt: int
):
    m = eq_cnt + ineq_cnt
    assert m == A.shape[0], "Inequality and equality constraints do not match the dimension of the problem"
    
    Delta_M = np.zeros(m)
    
    for i in range(m):
        if (i < eq_cnt): # i < eq_cnt, equality constraints
            residual = np.abs(A[i] @ x + b[i])
        else: # i >= eq_cnt, inequality constraints
            residual = np.maximum(0, A[i] @ x + b[i])
        Delta_M[i] = residual
    return Delta_M


def QP_solver(AE: np.ndarray, AI: np.ndarray, bE: np.ndarray, bI: np.ndarray, 
              g: np.ndarray, H: np.ndarray,
              solver: str = "ADAL",
              max_iter: int = 1000):
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
    
    M = np.ones(m)
    infeasible_cnt = np.zeros(m)
    
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
    
    for sp_iter in range(max_iter):
        # Solve Subproblem
        if (solver == "ADAL"):
            x_new, _ = ADAL(
                H, g, A_iter, b_iter, eq_cnt, ineq_cnt, 0.5, init_x = x
            )
        elif (solver == "IRWA"):
            # feasible = True
            # if (AI is not None) and (bI is not None):
            #     ineq_res = check_feasible(x, AI, bI, "inequ", optimal_check_eps = 1e-5 , printResult=False)
            #     feasible = feasible and ineq_res
            # if (AE is not None) and (bE is not None):
            #     eq_res = check_feasible(x, AE, bE, "equ", optimal_check_eps = 1e-5, printResult=False)
            #     feasible = feasible and eq_res
            
            # if feasible and (sp_iter > 0):
            #     eta_val = 0.995
            # else:
            #     eta_val = 0.8  + 0.19 * (1 - np.exp(-(sp_iter * 5) / max_iter))
            # print(f"IRWA Eta: {round(eta_val, 6)}")
            eta_val = 0.9975
            x_new, _ = IRWA(
                H, g, AE_iter, bE_iter, AI_iter, bI_iter, 
                1e4, x_init = x, eta = eta_val , max_iter= 10000
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
        
        delta_x = np.linalg.norm(x_new - x) / (n * np.linalg.norm(x) + 1e-6)
        
        x = x_new
        if (feasible) and ((delta_x < 5e-6) #or (
            #(solver == "IRWA") and (eta_val > 0.995)
        #)
        ):
            break
        
        # Update x
        
        # Print Info
        print("=" * 60)
        print(f"Subproblem Iteration {sp_iter}", end = " | ")
        obj_val = 0.5 * x.T @ H @ x + g @ x
        print(f"Objective Value: {round(obj_val, 4)}", end = " | ")
        print(f"Feasible: {feasible}", end = "\n")
        print(f"x: ", end = "")
        printVec(x[:10])
        
        # Update Penalty
        Delta_M = eval_M(x, A, b, eq_cnt, ineq_cnt)
        #print("Delta_M: ", end = "")
        #printVec(Delta_M[:10])
        infeasible_cnt += (Delta_M > 1e-6)
        #print("Infeasible Count: ", end = "")
        #printVec(infeasible_cnt[:10])
        M_Penalty_Param = (Delta_M * (infeasible_cnt + 1) * (infeasible_cnt * 1.1))
        
        M = M * np.diag(
            np.clip(
                #np.exp(M_Penalty_Param / n)
                np.log(1 + M_Penalty_Param)
                , 1, (n + 1)
            )
        )
        ## Show M's diagonal values
        print("Delta_M: ", end = "")
        printVec(Delta_M[:10])
        print("M: ", end = "")
        printVec(np.diag(M)[:10])
        print(f"Max M: {np.max(M)}")
        # If Any M is too large: means the problem is infeasible
        if np.any(M > 1e4 * n * n):
            raise ValueError("Problem is infeasible.")
        
        # Update A, b by M: let result as M(Ax+b)
        A_iter = M @ A
        b_iter = M @ b
        AE_iter, AI_iter = A_iter[:eq_cnt], A_iter[eq_cnt:]
        bE_iter, bI_iter = b_iter[:eq_cnt], b_iter[eq_cnt:]
    
    return x

if __name__ == "__main__":
    SOLVER = "ADAL"
    # SOLVER = "IRWA"
    if len(sys.argv) > 2:
        SOLVER = sys.argv[2]
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "./Tests/TestCases/reference.txt"
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
    if (np.abs(our_objctive - ref_objective) / (np.abs(ref_objective) * n) < 1e-5):
        print("========== Test Passed! ==========")
    else:
        print("========== Test Failed! ==========")
        