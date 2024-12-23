import numpy as np
import sys
import scipy.sparse as sp
from qpsolvers import solve_qp
from utils import init_from_config, check_feasible

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

    I_FLAG = False
    E_FLAG = False

    H = sp.csc_matrix(H)
    if (AI is not None) and (bI is not None):
        # AI x <= bI
        AI = sp.csc_matrix(AI)
        #bI = sp.csc_matrix(bI)
        I_FLAG = True
    if (AE is not None) and (bE is not None):
        AE = sp.csc_matrix(AE)
        #bE = sp.csc_matrix(bE)
        E_FLAG = True
    if (I_FLAG and E_FLAG):
        x = solve_qp(H, g, G = AI, h = bI, A = AE, b = bE, solver = "osqp")
    elif I_FLAG:
        x = solve_qp(H, g, G = AI, h = bI, solver = "osqp")
    elif E_FLAG:
        x = solve_qp(H, g, A = AE, b = bE, solver = "osqp")
    else:
        x = solve_qp(H, g, solver = "osqp")
    
    print("Solution: ", end = "")
    printVec(x)
    print("Objective Value: ", end = "")
    print(round(0.5 * x.T @ H @ x + g.T @ x, 4))
    # Feasibility Check
    if I_FLAG:
        # print("Inequality Feasibility: ")
        # INEQU_RES = AI @ x - bI
        # inequ_feas = np.all(INEQU_RES <= 0)
        # if (inequ_feas):
        #     print("Inequality Feasible Check Passed")
        # else:
        #     inequ_cnt = bI.shape[0]
        #     for i in range(inequ_cnt):
        #         if INEQU_RES[i] > 0:
        #             print(f"Inequality Feasibility Check Failed at {i}th constraint")
        #             print(f"Constraint: {AI[i].toarray()} x <= {bI[i]}")
        #             print(f"Result: {round(INEQU_RES[i], 4)}")
        check_feasible(x, AI, bI, "inequ")
    if E_FLAG:
        # print("Equality Feasibility: ")
        # EQU_RES = AE @ x - bE
        # equ_feas = np.all(np.abs(EQU_RES) <= 1e-5)
        # print(f"Equality Feasible: {equ_feas}")
        check_feasible(x, AE, bE, "equ")