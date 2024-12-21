import numpy as np
import sys
import scipy.sparse as sp
from qpsolvers import solve_qp
from utils import init_from_config

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

    if (AI is not None) and (bI is not None):
        AI = sp.csc_matrix(AI)
        bI = sp.csc_matrix(bI)
        I_FLAG = True
    if (AE is not None) and (bE is not None):
        AE = sp.csc_matrix(AE)
        bE = sp.csc_matrix(bE)
        
    if (I_FLAG and E_FLAG):
        x = solve_qp(H, g, G = AI, H = AE, A = AI, b = bI, d = bE, solver = "osqp")
    elif I_FLAG:
        x = solve_qp(H, g, G = AI, h = bI, solver = "osqp")
    elif E_FLAG:
        x = solve_qp(H, g, A = AE, b = bE, solver = "osqp")
    else:
        x = solve_qp(H, g, solver = "osqp")
    
    printVec(x)