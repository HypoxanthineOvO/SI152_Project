import numpy as np
from scipy.optimize import linprog
import os

# Config Utils
def loadConfig(cfg_path: str):
    assert os.path.exists(cfg_path), f"Config file {cfg_path} does not exist"
    with open(cfg_path, "r") as f:
        rows = f.readlines()
    # Split the lines into key-value pairs. key is start with xxx, split by ':', value are number or lists
    config = {}
    for row in rows:
        if row.strip() == "":
            continue
        key, value = row.split(":")
        key = key.strip()
        value = value.strip()
        if value == "None":
            config[key] = None
        elif value.isdigit():
            config[key] = int(value)
        else:
            # Value maybe an 1d or 2d array.
            # e.g. [[65., -22., -16.], [-22.,  14.,   7.], [-16., 7., 5.]]
            value = value.replace("[", "").replace("]", "").replace("\n", "")
            value = value.split(",")
            value = [float(v) for v in value]
            config[key] = value
    if "bI" in config and config["bI"] is not None:
        config["num_inequ"] = len(config["bI"])
    else:
        config["num_inequ"] = 0
    if "bE" in config and config["bE"] is not None:
        config["num_equ"] = len(config["bE"])
    else:
        config["num_equ"] = 0
    return config

def parseConfig(config: dict):
    n = config["n"]
    m = config["m"]
    
    if "H" not in config:
        if "P" in config:
            H = np.array(config["P"]).reshape(n, n)
        else:
            raise ValueError("H or P should be in the config")
    else:
        H = np.array(config["H"]).reshape(n, n)
    if "g" not in config:
        if "q" in config:
            g = np.array(config["q"])
        else:
            raise ValueError("g or q should be in the config")
    else:
        g = np.array(config["g"])
    if "AI" not in config or config["AI"] == None:
        AI = None
    else:
        AI = np.array(config["AI"]).reshape(
            config["num_inequ"], n
        )
    if "bI" not in config or config["bI"] == None:
        bI = None
    else:
        bI = np.array(config["bI"])
    if "AE" not in config or config["AE"] == None:
        AE = None
    else:
        AE = np.array(config["AE"]).reshape(
            config["num_equ"], n
        )
    if "bE" not in config or config["bE"] == None:
        bE = None
    else:
        bE = np.array(config["bE"])
    
    # AI, bI should have same existence
    if (AI is None) ^ (bI is None):
        raise ValueError("AI and bI should have the same existence")
    # AE, bE should have same existence
    if (AE is None) ^ (bE is None):
        raise ValueError("AE and bE should have the same existence")
    
    return n, m, H, g, AI, bI, AE, bE

def init_from_config(path: str):
    cfg = loadConfig(path)
    n, m, H, g, AI, bI, AE, bE = parseConfig(cfg)
    return n, m, H, g, AI, bI, AE, bE


# Mathematical Utils
def check_feasible(x: np.ndarray, A: np.ndarray, b: np.ndarray, type: str, optimal_check_eps: float = 0.01, printResult: bool = True):
    if type == "inequ":
        res = A @ x + b
        feas = np.all(res <= optimal_check_eps)
        if feas:
            if printResult:
                print("Inequality Feasible Check Passed")
            return True
        else:
            cnt = b.shape[0]
            if printResult:
                for i in range(cnt):
                    if res[i] > optimal_check_eps:
                        print(f"Inequality Feasibility Check Failed at {i}th constraint")
                        Ax = A[i] @ x
                        print(f"Ax + b = {round(Ax, 4)} + {round(b[i], 4)} = {round(Ax + b[i],6)}")
            return False
    elif type == "equ":
        res = A @ x + b
        feas = np.all(np.abs(res) <= optimal_check_eps)
        if feas:
            if printResult:
                print("Equality Feasible Check Passed")
            return True
        else:
            cnt = b.shape[0]
            if printResult:
                for i in range(cnt):
                    if np.abs(res[i]) > optimal_check_eps:
                        print(f"Equality Feasibility Check Failed at {i}th constraint")
                        Ax = A[i] @ x
                        print(f"Ax = {Ax}, b = {b[i]}")
                        
            return False

def eval_penalty(x: np.ndarray, A: np.ndarray, b: np.ndarray, type: str):
    if type == "inequ":
        res = A @ x + b
        res = np.maximum(res, 0)
    elif type == "equ":
        res = A @ x + b
        res = np.abs(res)
    return res

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
        print("约束非空")
        return True
    else:
        print("约束为空")
        return False

# Print Utils
def printVec(x: np.ndarray, num_digits: int = 4):
    for i in range(x.shape[0]):
        rounded_x = round(x[i], num_digits)
        print("{:.4f}".format(rounded_x), end = " ")
    print()


# Visualization Utils


if __name__ == "__main__":
    cfg = loadConfig("./Testcases/reference.txt")
    n, m, H, g, AI, bI, AE, bE = parseConfig(cfg)
    print(n, m)
    print(H)
    print(g)
    print(AI)
    print(bI)
    print(AE)
    print(bE)