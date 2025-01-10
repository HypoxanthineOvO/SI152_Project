import numpy as np
import os, sys, argparse
from scipy.optimize import linprog

def is_constraints_nonempty(A, b, C, d):
    """
    判断给定的线性等式约束和不等式约束是否非空。
    
    参数:
        A: 等式约束矩阵 (m x n)
        b: 等式约束右侧向量 (m,)
        C: 不等式约束矩阵 (p x n)
        d: 不等式约束右侧向量 (p,)
    
    返回:
        True: 约束非空
        False: 约束为空
    """
    m, n = A.shape
    
    # 步骤 1: 检查等式约束是否有解
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack((A, b)))
    
    if rank_A != rank_Ab:
        print("Equal constraints have no solution")
        return False
    
    # 步骤 2: 将等式约束的解空间表达出来
    # 求解特解 x0
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # 求解零空间基矩阵 N
    _, _, V = np.linalg.svd(A)
    null_space_dim = n - rank_A
    if null_space_dim == 0:
        # 如果零空间维度为 0，则解唯一
        N = np.zeros((n, 1))
    else:
        N = V[-null_space_dim:].T
    
    # 步骤 3: 将不等式约束代入解空间
    # 不等式约束变为 C @ (x0 + N @ z) <= d
    # 即 (C @ N) @ z <= d - C @ x0
    c_new = C @ N
    d_new = d - C @ x0
    
    # 步骤 4: 判断不等式是否有解
    # 使用线性规划检查可行性
    # 目标函数为 0，变量为 z，约束为 c_new @ z <= d_new
    # 如果存在解，则约束非空
    c_obj = np.zeros(c_new.shape[1])  # 目标函数为 0
    bounds = [(None, None)] * c_new.shape[1]  # z 无界
    res = linprog(c_obj, A_ub=c_new, b_ub=d_new, bounds=bounds, method='highs')
    
    if res.success:
        print("Constraints are nonempty")
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The file to save the generated testcase")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--n", type=int, default=1000, help="The dimension of the problem")
    parser.add_argument("--m", type=int, default=600, help="The number of constraints")
    
    args = parser.parse_args()
    
    FILE = args.file
    n = args.n
    m = args.m
    if args.seed != -1:
        np.random.seed(args.seed)
    # Generate H: Random matrix
    ## L: n\times n, Gaussian with mean = 1, std = 2 
    L = np.random.normal(1, 2, (n, n))
    ## H = 0.1 I + LL^T
    H = 0.1 * np.identity(n) + L @ L.T
    
    # Generate g: Random vector
    mean_g = np.random.randint(-100, 100)
    std_g = np.random.randint(1, 100)
    g = np.random.normal(mean_g, std_g, n)
    
    # Generate A: Random matrix
    mean_A = np.random.randint(1, 10)
    std_A = np.random.randint(1, 10)
    A = np.random.normal(mean_A, std_A, (m, n))
    
    # Generate b: Random vector
    mean_b = np.random.randint(-100, 100)
    std_b = np.random.randint(1, 100)
    b = np.random.normal(mean_b, std_b, m)
    
    AI = A[:m // 2]
    bI = b[:m // 2]
    AE = A[m // 2:]
    bE = b[m // 2:]
    
    # Check if the constraints are nonempty
    if not is_constraints_nonempty(AI, bI, AE, bE):
        print("Constraints are empty. Regenerating...")
        exit(1)
    
    # Save the variables as code to a file
    with open(FILE, "w") as f:
        f.write(f"n: {n}\n")
        f.write(f"m: {m}\n")
        # To flatten the matrix, we use the .tolist() method
        f.write(f"H: {H.tolist()}\n")
        f.write(f"g: {g.tolist()}\n")
        f.write(f"AI: {A[:m // 2].tolist()}\n")
        f.write(f"bI: {b[:m // 2].tolist()}\n")
        f.write(f"AE: {A[m // 2:].tolist()}\n")
        f.write(f"bE: {b[m // 2:].tolist()}\n")
    
    print("========== IRWA Experiment 1 Generated ==========")
    print(f"n: {n}, m: {m}")