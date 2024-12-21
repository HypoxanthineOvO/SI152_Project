import numpy as np
import os

def loadConfig(cfg_path: str):
    assert os.path.exists(cfg_path), f"Config file {cfg_path} does not exist"
    with open(cfg_path, "r") as f:
        raws = f.readlines()
    # Split the lines into key-value pairs. key is start with xxx, split by ':', value are number or lists
    config = {}
    for raw in raws:
        if raw.strip() == "":
            continue
        key, value = raw.split(":")
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

def check_feasible(x: np.ndarray, A: np.ndarray, b: np.ndarray, type: str, optimal_check_eps: float = 0.01):
    if type == "inequ":
        res = A @ x - b
        feas = np.all(res <= optimal_check_eps)
        if feas:
            print("Inequality Feasible Check Passed")
            return True
        else:
            cnt = b.shape[0]
            for i in range(cnt):
                if res[i] > optimal_check_eps:
                    print(f"Inequality Feasibility Check Failed at {i}th constraint")
                    Ax = A[i] @ x
                    print(f"Ax = {Ax}, b = {b[i]}")
            return False
    elif type == "equ":
        res = A @ x - b
        feas = np.all(np.abs(res) <= optimal_check_eps)
        if feas:
            print("Equality Feasible Check Passed")
            return True
        else:
            cnt = b.shape[0]
            for i in range(cnt):
                if np.abs(res[i]) > optimal_check_eps:
                    print(f"Equality Feasibility Check Failed at {i}th constraint")
                    Ax = A[i] @ x
                    print(f"Ax = {Ax}, b = {b[i]}")
            return False


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