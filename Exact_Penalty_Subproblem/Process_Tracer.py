import numpy as np

import os, sys
import matplotlib.pyplot as plt

from Transform_Test_To_Lingo import init_from_config
from ADAL import eval_exact_penalty

def parse_log(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    
    x_log = []
    for line in lines:
        # Form: Iter 0: [ 1.08743372 -0.56945478]
        if  (line.startswith("Init")) or (line.startswith("Iter")):
            raw = line.split(":")[1].strip()
            x_str = raw[1:-1].split()
            x = np.array([float(x_str[0]), float(x_str[1])])
            x_log.append(x)
    
    return np.array(x_log)

def Draw_Linear_Constraint(Ai: np.ndarray, bi: np.ndarray, X: np.ndarray, Y: np.ndarray):
    Z = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = Ai @ x + bi
    
    return Z

if __name__ == "__main__":
    FILE = "./Tests/00-Easy.txt"
    if (len(sys.argv) > 1):
        FILE = sys.argv[1]
    
    LOG_FILE = "ADAL.log"
    
    n, m, H, g, AI, bI, AE, bE, ref, ref_val = init_from_config(FILE)
    
    x_log = parse_log(LOG_FILE)
    
    assert n == 2, "Only support 2D visualization"
    
    equ_cnt = AE.shape[0] if AE is not None else 0
    inequ_cnt = AI.shape[0] if AI is not None else 0
    
    A, b = np.zeros((m, n)), np.zeros(m)
    if (AE is not None) and (bE is not None):
        A[:equ_cnt] = AE
        b[:equ_cnt] = bE
    if (AI is not None) and (bI is not None):
        A[equ_cnt:] = AI
        b[equ_cnt:] = bI
    
    
    
    # Bounds
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    
    xs = np.linspace(x_min, x_max, 100)
    ys = np.linspace(y_min, y_max, 100)
    
    X, Y = np.meshgrid(xs, ys)
    Z1 = np.zeros(X.shape)
    Z2 = np.zeros(X.shape)
    mj_xy = np.array([x_min, y_min])
    mj = 1e5
    
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z1[i, j] = 0.5 * x.T @ H @ x + g @ x
            Z2[i, j] = eval_exact_penalty(H, g, A, b, equ_cnt, inequ_cnt, x)
            if Z2[i, j] < mj:
                mj = Z2[i, j]
                mj_xy = x
    
    constraints = []
    colors = ["green", "skyblue", "pink", "purple", "orange", "yellow", "black", "brown"]

    for i in range(A.shape[0]):
        constraints.append(Draw_Linear_Constraint(A[i], b[i], X, Y))
    
    # Visualization
    plt.figure(figsize = (18, 8))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, Z1, 100, cmap = "rainbow", levels=np.linspace(-10, 100, 110))
    plt.colorbar()
    for i, constraint in enumerate(constraints):
        if i < equ_cnt:
            plt.contour(X, Y, constraint, 0, colors = "skyblue")
        else:
            plt.contour(X, Y, constraint, 0, colors = "orange")
    # Draw x_axis and y_axis
    plt.axhline(0, color = "darkgray", linewidth = 3)
    plt.axvline(0, color = "darkgray", linewidth = 3)
    
    # Draw x_log
    for i in range(len(x_log) - 1):
        plt.plot([x_log[i][0], x_log[i + 1][0]], [x_log[i][1], x_log[i + 1][1]], color = "red", linewidth = 2)
        #plt.scatter(x_log[i][0], x_log[i][1], color = "red", s = 5)
    plt.scatter(x_log[-1][0], x_log[-1][1], color = "red", s = 60)
    #plt.scatter(mj_xy[0], mj_xy[1], color = "blue", s = 60)
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z2, 100, cmap = "rainbow", levels=np.linspace(-10, 100, 110))
    plt.colorbar()
    for i, constraint in enumerate(constraints):
        if i < equ_cnt:
            plt.contour(X, Y, constraint, 0, colors = "skyblue")
        else:
            plt.contour(X, Y, constraint, 0, colors = "orange")
    # Draw x_axis and y_axis
    plt.axhline(0, color = "darkgray", linewidth = 3)
    plt.axvline(0, color = "darkgray", linewidth = 3)
    
    # Draw x_log
    for i in range(len(x_log) - 1):
        plt.plot([x_log[i][0], x_log[i + 1][0]], [x_log[i][1], x_log[i + 1][1]], color = "red", linewidth = 2)
        #plt.scatter(x_log[i][0], x_log[i][1], color = "red", s = 5)
    plt.scatter(x_log[-1][0], x_log[-1][1], color = "red", s = 60)
    #plt.scatter(mj_xy[0], mj_xy[1], color = "blue", s = 60)
    plt.savefig("Logs.png")