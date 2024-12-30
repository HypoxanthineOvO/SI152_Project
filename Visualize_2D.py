import numpy as np
import os, sys
from utils import init_from_config, check_feasible
from ADAL_OSQP import QP_solver as ADAL_Solver
from IRWA import QP_solver as IRWA_Solver
from qpsolvers import solve_qp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "./Testcases/reference.txt"
    n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_file)

    # Check Dimensions 
    I_n = np.identity(n)
    I_m = np.identity(m)

    # Generate H and g
    H = np.array([
        [1, 0],
        [0, -3]
    ])
    g = np.array([0, 0])
    
    # Bounds
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    
    xs = np.linspace(x_min, x_max, 100)
    ys = np.linspace(y_min, y_max, 100)
    
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = 0.5 * x.T @ H @ x + g @ x
    
    OPT_REF = solve_qp(P = H, q = g, solver = "osqp")
    print("OPT_REF: ", OPT_REF)
    
    
    # Visualization
    plt.figure(figsize = (10, 8))
    plt.contour(X, Y, Z, 50, cmap = "RdGy")
    plt.colorbar()
    
    plt.scatter(OPT_REF[0], OPT_REF[1], color = "red", label = "OPT_REF")
    
    plt.show()




    # print("==================== ADAL ====================")
    # x = ADAL_Solver(AE, AI, bE, bI, g, H)
    
    # print("x: ", end = "")
    # print(x)
    # print("Objective Value: ", round(1/2 * x.T@H@x + g @ x, 4))
        
    # if (AI is not None) and (bI is not None):
    #     print("*", end=" ")
    #     check_feasible(x, AI, bI, "inequ", optimal_check_eps=1e-4)
    # else:
    #     print("* No inequality constraints.")
    # if (AE is not None) and (bE is not None):
    #     print("*", end=" ")
    #     check_feasible(x, AE, bE, "equ", optimal_check_eps=1e-4)
    
    # print("==================== IRWA ====================")
    # x = IRWA_Solver(AE, AI, bE, bI, g, H)
    # if x is not None:
    #     print("x:", end=" ")
    #     print(x)
    #     print("Objective Value: ", round(1/2 * x.T@H@x + g @ x, 4))
    
    #     if (AI is not None) and (bI is not None):
    #         print("*", end=" ")
    #         check_feasible(x, AI, bI, "inequ", optimal_check_eps=1e-4)
    #     else:
    #         print("* No inequality constraints.")
    #     if (AE is not None) and (bE is not None):
    #         print("*", end=" ")
    #         check_feasible(x, AE, bE, "equ", optimal_check_eps=1e-4)