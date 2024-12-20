# Quadric Programming Example

## QP_Solver
```python
x = solve_qp(H, g, AI, bI, solver = "osqp")
```

## Example 1: In Slides 14 Page 15:
### Objective Function

$$
\begin{align*}
    \min_x \quad& (x_1-1)^2+(x_2-2.5)^2 = x_1^2 + x_2^2 - 2x_1 - 5x_2 + 5.25\\
    \text{subject to }\quad & x_1-2x_2+2\geq 0\\
    & -x_1-2x_2+6\geq 0\\
    & -x_1+2x_2+2\geq 0\\
    & x_1\geq 0\\
    & x_2\geq 0
\end{align*}
$$

We have $H = 2I$, $g = [-2, -5]^T$, $AI = \begin{bmatrix} -1 & 2 \\ 1 & 2 \\ 1 & -2 \\ -1 & 0 \\ 0 & -1 \end{bmatrix}$, $bI = [2, 6, 2, 0, 0]^T$

### Solution
The optimal solution = [1.4000, 1.7000]

### Code
```python
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
    
    
```

## Example 2: QP_Solver Example
```python
n = 3
m = 4
M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
H = M.T @ M  # this is a positive definite matrix
g = np.array([3.0, 2.0, 3.0]) @ M
AI = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
bI = np.array([3.0, 2.0, -2.0])
AE = np.array([1.0, 1.0, 1.0])
bE = np.array([1.0])
```

The Optimal solution = [0.30769231, 0.69230769, 1.0]