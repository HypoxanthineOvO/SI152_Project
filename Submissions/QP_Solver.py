import numpy as np
import os, sys

#################### Utils ####################
# Print Utils
def printVec(x: np.ndarray, num_digits: int = 4):
    for i in range(x.shape[0]):
        rounded_x = round(x[i], num_digits)
        print("{:.4f}".format(rounded_x), end = " ")
    print()

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

def eval_exact_penalty(
    H: np.ndarray, g: np.ndarray, 
    A: np.ndarray, b: np.ndarray,
    equal_cnt: int, ineq_cnt: int, 
    x: np.ndarray,):
    val = 0.5 * x.T @ H @ x + g @ x
    for i in range(equal_cnt):
        val += np.abs(A[i] @ x + b[i])
    for i in range(ineq_cnt):
        val += np.maximum(0, A[i + equal_cnt] @ x + b[i + equal_cnt])
    return val

#################### Subproblem Solver ####################

def ADAL(
    H: np.ndarray, g: np.ndarray,
    A: np.ndarray, b: np.ndarray, eq_cnt: int, ineq_cnt: int,
    mu: float, sigma: float = 1e-6, sigmapp: float = 1e-5,
    init_x: np.ndarray = None,
    show_log: bool = False
):
    n = H.shape[0]
    m = ineq_cnt + eq_cnt
    assert m == A.shape[0], "Inequality and equality constraints do not match the dimension of the problem"
    
    # Initialize
    if init_x is not None:
        x = init_x
    else:
        x = np.zeros(n)
    u = np.zeros(m)
    p = np.zeros(m)
    
    x_log = [x]
    if show_log:
        print(f"Initialize, Objective: {round(eval_exact_penalty(H, g, A, b, eq_cnt, ineq_cnt, x), 4)}, loss = 0.000000 & {round(np.linalg.norm(A @ x + b - p), 6)}")
    
    for iter in range(10000):
        # Step 1: Solve the augmented Lagrangian subproblem for (x^{k+1}, p^{k+1})
        ## Step 1.1: optimize p
        s = A @ x + b + mu * u
        p_new = np.zeros(m)
        for i in range(m):
            if (i < eq_cnt): # i < eq_cnt, equality constraints
                proj_pt = 0
            else: # i >= eq_cnt, inequality constraints
                proj_pt = 0
                if s[i] <= 0:
                    proj_pt = s[i]
            dist = np.linalg.norm(s[i] - proj_pt)
            if (dist <= mu):
                p_new[i] = proj_pt
            else:
                p_new[i] = s[i] - mu / dist * (s[i] - proj_pt)
        ## Step 1.2: optimize x
        ### 操，正经人谁玩梯度下降啊
        left_Matrix = H + 1 / mu * A.T @ A
        right_Vector = g + 1 / mu * A.T @ (b - p_new + mu * u)
        x_new = np.linalg.solve(left_Matrix, -right_Vector)
        
        # Step 2: Optimize u
        u_new = u + 1 / mu * (A @ x_new + b - p_new)
        
        # Step 3: Check convergence
        loss_1 = np.linalg.norm(x_new - x)
        loss_2 = np.linalg.norm(A @ x_new + b - p_new)
        
        if loss_1 < sigma and loss_2 < sigmapp:
            if show_log:
                print(f"\nAlgorithm Converged at iter {iter}: loss 1 = {loss_1}, loss 2 = {loss_2}")
            break
        # Step 4: Update Variables
        x = x_new
        p = p_new
        u = u_new
        
        x_log.append(x)
        if show_log:
            print(f"Iter {iter:5}, Objective: {round(eval_exact_penalty(H, g, A, b, eq_cnt, ineq_cnt, x), 4)}, loss = {round(loss_1, 6):.6f} & {round(loss_2, 6):.6f}")
    return x, x_log

def IRWA(H, g, AE, bE, AI, bI, eps_init, x_init, 
         eta = 0.9, gamma = 1/6, M = 10000, 
         sigma = 1e-6, sigma_prime = 1e-8, 
         max_iter = 5000):
    def compute_weights(x_tilde, AE, bE, AI, bI, eps):
        equ_cnt = AE.shape[0] if AE is not None else 0
        w1 = None
        w2 = None
        if AE is not None and bE is not None:
            r1 = AE @ x_tilde + bE
            eps_1 = eps[:equ_cnt]
            w1 = 1.0 / np.sqrt(r1**2 + eps_1**2)
        if AI is not None and bI is not None:
            r2 = AI @ x_tilde + bI
            eps_2 = eps[equ_cnt:]
            hinge_r2 = np.maximum(r2, 0)
            w2 = 1.0 / np.sqrt(hinge_r2**2 + eps_2**2)
        
        return w1, w2


    x = x_init.copy()
    
    x_logs = [x]

    A = np.vstack([AE, AI]) if AE is not None and AI is not None else AE if AE is not None else AI
    b = np.concatenate([bE, bI]) if bE is not None and bI is not None else bE if bE is not None else bI

    l = AE.shape[0] if AE is not None else 0
    
    eps_k = eps_init if not np.isscalar(eps_init) else np.full(A.shape[0], eps_init)
    
    for _ in range(max_iter):
        # Step 1: Compute weights and solve the reweighted subproblem
        w1, w2 = compute_weights(x, AE, bE, AI, bI, eps_k)
        
        if w1 is not None and w2 is not None:
            W = np.diag(np.concatenate([w1, w2]))
            v = np.concatenate([bE, np.maximum(-AI @ x, bI)])
        elif w1 is not None:
            W = np.diag(w1)
            v = bE
        else:
            W = np.diag(w2)
            v = np.maximum(-AI @ x, bI)

        x_next = np.linalg.solve(H + A.T @ W @ A, -g - A.T @ W @ v)

        # Step 2: Update eps
        q_k = A @ (x_next - x)
        r_k = (1.0 - v) * (A @ x + b)
        
        # Check condition for eps updating
        lhs_condition = np.abs(q_k)
        rhs_condition = M * ((r_k**2 + eps_k**2)**(0.5 + gamma)) 

        if np.all(lhs_condition <= rhs_condition):
            eps_next = np.zeros_like(eps_k)
            for i in range(l):
                eps_next[i] = eta * eps_k[i]
            for i in range(l, len(eps_k)):
                if A[i] @ x_next + b[i] < - eps_k[i]:
                    eps_next[i] = eps_k[i]
                else:
                    eps_next[i] = eta * eps_k[i]
              
                
            
        else:
            eps_next = eps_k
        
        # Step 3: Check stopping criteria
        diff_x = np.linalg.norm(x_next - x, 2)
        diff_eps = np.linalg.norm(eps_next - eps_k, 2)
    
        if (diff_x <= sigma) and (diff_eps <= sigma_prime):
            break
        
        x = x_next
        eps_k = eps_next
        
        
        x_logs.append(x)
    
    return x, x_logs

def OSQP(A: np.ndarray, l: np.ndarray, u: np.ndarray, 
         g: np.ndarray, H: np.ndarray,
         rho: float = 1,
         sigma: float = 1,
         alpha: float = 1.5,
         eps = 1e-5):
    
    def projection(pt: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Clip the values
        return np.clip(pt, l, u)

    r_primal_records = []
    r_dual_records = []

    ## Step 1: Initialize
    n = H.shape[0]
    m = A.shape[0]
    I_n = np.identity(n)
    I_m = np.identity(m)
    
    x = np.zeros(n)
    z = np.zeros(m)
    y = np.zeros(m)
    ## Step 2: Run OSQP
    for iter in range(10000):
        # Step 0: Check x^THx is positive
        check_vec = x.T @ H @ x
        if check_vec < -eps:
            raise ValueError("H is not positive definite!")
        
        # Step 1: Solve the linear system subproblem
        ## Step 1.1: Compute the matrix and vector
        
        Left_Matrix = np.zeros((n + m, n + m))
        Left_Matrix[:n, :n] = H + rho * I_n # Left-Top
        Left_Matrix[n:, :n] = A # Left-Bottom
        Left_Matrix[:n, n:] = A.T # Right-Top
        Left_Matrix[n:, n:] = -(1/rho) * I_m # Right-Bottom
        
        Right_Vector = np.zeros(n + m)
        Right_Vector[:n] = sigma * x - g
        Right_Vector[n:] = z - (1/rho * y)
        
        sol_subproblem = np.linalg.solve(Left_Matrix, Right_Vector)
        
        x_aul_new = sol_subproblem[:n]
        nu_new = sol_subproblem[n:]
        
        ## Step 1.2: Update z
        z_aul_new = z + (1/rho) * (nu_new - y)
        ## Step 1.3: Update x
        x_new = alpha * x_aul_new + (1 - alpha) * x
        ## Step 1.4: Update z
        z_before_proj = alpha * z_aul_new + (1 - alpha) * z + (1/rho) * y
        z_new = projection(z_before_proj, l, u)

        ## Step 1.5: Update y
        y_new = y + rho * (alpha * z_aul_new + (1 - alpha) * z - z_new)
        
        # Step 2: Check convergence
        r_primal = np.linalg.norm(A @ x_new - z_new)
        r_dual = np.linalg.norm(H @ x_new + g + A.T @ y_new)
        
        r_primal_records.append(r_primal)
        r_dual_records.append(r_dual)
        if r_primal < eps and r_dual < eps:
            break
        # Step 3: Update rho and sigma
        x = x_new
        z = z_new
        y = y_new
    
    return x, r_primal_records, r_dual_records, iter


def QP_solver(AE: np.ndarray, AI: np.ndarray, bE: np.ndarray, bI: np.ndarray, 
              g: np.ndarray, H: np.ndarray,
              solver: str = "OSQP",
              max_iter: int = 1000):
    def eval_M(
        x: np.ndarray, A: np.ndarray, b: np.ndarray,
        eq_cnt: int, ineq_cnt: int
    ):
        m = eq_cnt + ineq_cnt
        assert m == A.shape[0], "Inequality and equality constraints do not match the dimension of the problem"
        
        Delta_M = np.zeros(m)
        
        for i in range(m):
            if (i < eq_cnt): # i < eq_cnt, equality constraints
                residual = np.abs(A[i] @ x + b[i])
            else: # i >= eq_cnt, inequality constraints
                residual = np.maximum(0, A[i] @ x + b[i])
            Delta_M[i] = residual
        return Delta_M

    # Dimension Check
    AI_len = AI.shape[0] if AI is not None else 0
    bI_len = bI.shape[0] if bI is not None else 0
    assert AI_len == bI_len, "Inequality constraints do not match the dimension of the problem"
    AE_len = AE.shape[0] if AE is not None else 0
    bE_len = bE.shape[0] if bE is not None else 0
    assert AE_len == bE_len, "Equality constraints do not match the dimension of the problem"
    eq_cnt, ineq_cnt = AE_len, AI_len
    
    n = H.shape[0]
    m = eq_cnt + ineq_cnt
    assert eq_cnt + ineq_cnt == m, "Inequality and equality constraints do not match the dimension of the problem"
    
    
    # x for iteration
    x = np.zeros(n)
    
    M = np.ones(m)
    infeasible_cnt = np.zeros(m)
    
    # Generate A and l,u from AI, bI, AE, bE
    A = np.zeros((m, n))
    b = np.zeros(m)
    l = np.zeros(m)
    u = np.zeros(m)
    
    if (AE is not None) and (bE is not None):
        A[:eq_cnt] = AE
        b[:eq_cnt] = bE
        l[AI_len:] = -bE
        u[AI_len:] = -bE
    if (AI is not None) and (bI is not None):
        A[eq_cnt:] = AI
        b[eq_cnt:] = bI
        l[:AI_len] = -np.inf
        u[:AI_len] = -bI
    
    A_iter, b_iter = A.copy(), b.copy()
    AE_iter, AI_iter = A_iter[:eq_cnt], A_iter[eq_cnt:]
    bE_iter, bI_iter = b_iter[:eq_cnt], b_iter[eq_cnt:]
    
    if solver == "OSQP":
        x, _, _, iter = OSQP(
            A, l, u, g, H
        )
        return x, iter
    
    for sp_iter in range(max_iter):
        # Solve Subproblem
        if (solver == "ADAL"):
            x_new, _ = ADAL(
                H, g, A_iter, b_iter, eq_cnt, ineq_cnt, 0.5, init_x = x
            )
        elif (solver == "IRWA"):
            x_new, _ = IRWA(
                H, g, AE_iter, bE_iter, AI_iter, bI_iter, 
                1e3, x_init = x, eta = 0.9975 , max_iter= 10000
            )
        else:
            raise ValueError(f"Solver {solver} not supported.")
        
        # Check Feasibility
        feasible = True
        if (AI is not None) and (bI is not None):
            ineq_res = check_feasible(x_new, AI, bI, "inequ", optimal_check_eps = 1e-5 , printResult=False)
            feasible = feasible and ineq_res
        if (AE is not None) and (bE is not None):
            eq_res = check_feasible(x_new, AE, bE, "equ", optimal_check_eps = 1e-5, printResult=False)
            feasible = feasible and eq_res
        
        delta_x = np.linalg.norm(x_new - x) / (n * np.linalg.norm(x) + 1e-6)
        
        # Update x
        x = x_new
        if (feasible) and (delta_x < 5e-6):
            break
        
        if np.any(np.isnan(x)):
            return None, sp_iter 
            
        
        # Print Info
        print("=" * 60)
        print(f"Subproblem Iteration {sp_iter}", end = " | ")
        obj_val = 0.5 * x.T @ H @ x + g @ x
        print(f"Objective Value: {round(obj_val, 4)}", end = " | ")
        print(f"Feasible: {feasible}", end = "\n")
        print(f"x: ", end = "")
        printVec(x[:10])
        
        # Update Penalty
        Delta_M = eval_M(x, A, b, eq_cnt, ineq_cnt)
        infeasible_cnt += (Delta_M > 1e-6)
        M_Penalty_Param = (Delta_M * (infeasible_cnt + 1) * (infeasible_cnt * 1.1))
        
        M = M * np.diag(
            np.clip(
                np.log(1 + M_Penalty_Param)
                , 1, (n + 1)
            )
        )
        ## Show M's diagonal values
        print("Delta_M: ", end = "")
        printVec(Delta_M[:10])
        print("M: ", end = "")
        printVec(np.diag(M)[:10])
    
        # If Any M is too large: means the problem is infeasible
        if np.any(M > 1e4 * n * n):
            raise ValueError("Problem is infeasible.")
        
        # Update A, b by M: let result as M(Ax+b)
        A_iter = M @ A
        b_iter = M @ b
        AE_iter, AI_iter = A_iter[:eq_cnt], A_iter[eq_cnt:]
        bE_iter, bI_iter = b_iter[:eq_cnt], b_iter[eq_cnt:]
    
    return x, sp_iter


if __name__ == "__main__":
    SOLVER = "OSQP"
    if len(sys.argv) > 2:
        SOLVER = sys.argv[2]
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        raise ValueError("Config file not provided.")
    n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_file)

    # Check Dimensions
    I_n = np.identity(n)
    I_m = np.identity(m)

    print("==================== QP_Solver ====================")
    x = QP_solver(AE, AI, bE, bI, g, H, solver = SOLVER)
    
    print("x: ", end = "")
    printVec(x[:20])
    our_objctive = 1/2 * x.T @ H @ x + g @ x
    print("Objective Value: ", round(our_objctive, 4))
        
    if (AI is not None) and (bI is not None):
        print("*", end=" ")
        check_feasible(x, AI, bI, "inequ", optimal_check_eps=1e-4)
    else:
        print("* No inequality constraints.")
    if (AE is not None) and (bE is not None):
        print("*", end=" ")
        check_feasible(x, AE, bE, "equ", optimal_check_eps=1e-4)
    else:
        print("* No equality constraints.")
    
    
    print("==================== Optimize Done! ====================")