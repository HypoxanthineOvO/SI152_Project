import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp
import os, sys

def printVec(x: np.ndarray, num_digits: int = 4):
    for i in range(x.shape[0]):
        rounded_x = round(x[i], num_digits)
        print("{:.4f}".format(rounded_x), end = " ")
    print()

if __name__ == "__main__":
    n = 2
    I = np.identity(n)
    H = 2 * I
    # Convert to scipy sparse matrix
    H = sp.csc_matrix(H)
    g = np.array([-2, -5])
    AI = np.array([
        [-1, 2],
        [1, 2],
        [1, -2],
        [-1, 0],
        [0, -1]
    ])
    AI = sp.csc_matrix(AI)
    bI = np.array([2, 6, 2, 0, 0])
    
    x = solve_qp(H, g, AI, bI, solver = "osqp")
    
    printVec(x)