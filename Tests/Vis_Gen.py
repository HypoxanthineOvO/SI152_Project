import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

def Draw_Linear_Constraint(Ai: np.ndarray, bi: np.ndarray, X: np.ndarray, Y: np.ndarray):
    Z = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = Ai @ x - bi
    
    return Z

if __name__ == "__main__":
    # Generate H and g
    H = np.array([
        [1, 0],
        [0, -6]
    ])
    g = np.array([0, 0])
    
    A = np.array([
        [1, 1],
        [-1, 1]
    ])
    b = np.array([
        [1],
        [1]
    ])
    
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
    
    constraints = []
    colors = ["green", "blue", "red", "purple", "orange", "yellow", "black", "brown"]
    #constraint = Draw_Linear_Constraint(A, b, X, Y)
    for i in range(A.shape[0]):
        constraints.append(Draw_Linear_Constraint(A[i], b[i], X, Y))
    
    # Visualization
    plt.figure(figsize = (10, 8))
    plt.contour(X, Y, Z, 50, cmap = "RdGy")
    plt.colorbar()
    for i, constraint in enumerate(constraints):
        plt.contour(X, Y, constraint, 0, colors = colors[i])
    plt.show()