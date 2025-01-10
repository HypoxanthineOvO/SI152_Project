import numpy as np
import os, sys

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
        print("等式约束无解")
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
    np.random.seed(0)
    
    if (len(sys.argv) > 1):
        FILE = sys.argv[1]
    else:
        FILE = "../IRWA_3.txt"
    