import numpy as np
import os, sys

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
    
    # Ref & Ref_Val
    if "ref" in config:
        ref = np.array(config["ref"])
    else:
        ref = None
    if "ref_val" in config:
        ref_val = config["ref_val"][0]
    else:
        ref_val = None
    return n, m, H, g, AI, bI, AE, bE, ref, ref_val

def init_from_config(path: str):
    cfg = loadConfig(path)
    n, m, H, g, AI, bI, AE, bE, ref, ref_val = parseConfig(cfg)
    return n, m, H, g, AI, bI, AE, bE, ref, ref_val

def parse_config_to_lingo(
    n,m,H,g,AI,bI,AE,bE,
    is_Exact_Penalty: bool = True
):
    xs = [f"x{i}" for i in range(1, n + 1)]
    
    OBJECTIVE_STR = "MIN = 0.5 * ("
    # Parse H
    for i in range(n):
        for j in range(n):
            if (np.abs(H[i, j]) < 1e-6):
                continue
            if (np.abs(H[i, j] - 1) < 1e-6):
                OBJECTIVE_STR += f"{xs[i]} * {xs[j]} "
            else:
                OBJECTIVE_STR += f"{H[i, j]} * {xs[i]} * {xs[j]} "
            if i != n - 1 or j != n - 1:
                OBJECTIVE_STR += "+ "
    OBJECTIVE_STR += ") + ("
    # Parse g
    for i in range(n):
        if (np.abs(g[i]) < 1e-6):
            continue
        OBJECTIVE_STR += f"{g[i]} * {xs[i]} "
        if (i != n - 1):
            OBJECTIVE_STR += "+ "
    ## If end with a "+", remove it
    if OBJECTIVE_STR[-2:] == "+ ":
        OBJECTIVE_STR = OBJECTIVE_STR[:-2]
    if (is_Exact_Penalty): 
        OBJECTIVE_STR += ")"
        # Parse AE, bE
        if (AE is not None) and (bE is not None):
            for i in range(AE.shape[0]):
                OBJECTIVE_STR += " + @abs("
                for j in range(n):
                    if (np.abs(AE[i, j]) < 1e-6):
                        continue
                    OBJECTIVE_STR += f"{AE[i, j]} * {xs[j]} "
                    if j != n - 1:
                        OBJECTIVE_STR += "+ "
                OBJECTIVE_STR += f"+ {bE[i]}) "
        # Parse AI, bI
        if (AI is not None) and (bI is not None):
            for i in range(AI.shape[0]):
                OBJECTIVE_STR += " + @smax(0, "
                for j in range(n):
                    if (np.abs(AI[i, j]) < 1e-6):
                        continue
                    OBJECTIVE_STR += f"{AI[i, j]} * {xs[j]} "
                    if j != n - 1:
                        OBJECTIVE_STR += "+ "
                if OBJECTIVE_STR[-2:] == "+ ":
                    OBJECTIVE_STR = OBJECTIVE_STR[:-2]
                OBJECTIVE_STR += f"+ {bI[i]}) "
        
        OBJECTIVE_STR += ";\n"
    else:
        OBJECTIVE_STR += ");\n"
        #TODO: Parse by add constraints
    print(OBJECTIVE_STR)

if __name__ == "__main__":
    FILE = "./Tests/00-Easy.txt"
    if (len(sys.argv) > 1):
        FILE = sys.argv[1]
    n, m, H, g, AI, bI, AE, bE,_, _ = init_from_config(FILE)
    parse_config_to_lingo(n, m, H, g, AI, bI, AE, bE)