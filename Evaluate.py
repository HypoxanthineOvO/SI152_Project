import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from OSQP import ADAL
from IRWA import IRWA
from utils import init_from_config, check_feasible
from reference import reference

def ADAL_UnitTest(cfg_path: str):
    
    n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_path)
    
        # Dimension Check
    AI_len = AI.shape[0] if AI is not None else 0
    bI_len = bI.shape[0] if bI is not None else 0
    assert AI_len == bI_len, "Inequality constraints do not match the dimension of the problem"
    AE_len = AE.shape[0] if AE is not None else 0
    bE_len = bE.shape[0] if bE is not None else 0
    
    n = H.shape[0]
    m = AI_len + AE_len
    assert AE_len == bE_len, "Equality constraints do not match the dimension of the problem"
    assert AI_len + AE_len == m, "Inequality and equality constraints do not match the dimension of the problem"
    
    
    
    
    # Generate A and l,u from AI, bI, AE, bE
    A = np.zeros((m, n))
    l = np.zeros(m)
    u = np.zeros(m)
    
    if AI is not None:
        A[:AI_len] = AI
        l[:AI_len] = -np.inf
        u[:AI_len] = -bI
    if AE is not None:
        A[AI_len:] = AE
        l[AI_len:] = -bE
        u[AI_len:] = -bE
    
    # Do OSQP
    
    rho = 1
    sigma = 1
    alpha = 1.6
    
    x, primal_r, dual_r = ADAL(A, l, u, g, H, rho, sigma, alpha)
    
    return x, primal_r, dual_r

def UnitTest(
    Solver_Func: callable,
    Testcase_Type: int,
    Testcase_Index: int
):
    Testcase_Config = f"./Paper_Tests/{Testcase_Type}/TestCase_{Testcase_Type}_case{Testcase_Index}.txt"
    if not os.path.exists(Testcase_Config):
        print(f"Testcase {Testcase_Config} does not exist")
        return False
    
    x, primal_r, dual_r = Solver_Func(Testcase_Config)
    # Check feasibility
    #feas = check_feasible(A, l, u, x)
    #assert feas, "The solution is not feasible"
    
    plt.figure(figsize = (10, 6))
    plt.plot(primal_r[10:], label = "Primal Residual")
    plt.plot(dual_r[10:], label = "Dual Residual")
    plt.legend()
    plt.savefig(f"./Tests.png")

def TypeTest(
    Solver_Func: callable,
    Testcase_Type: int
):
    Testcase_Dir = f"./Paper_Tests/{Testcase_Type}"
    num_of_cases = len([name for name in os.listdir(Testcase_Dir) if os.path.isfile(os.path.join(Testcase_Dir, name))])
    #print(f"Running {num_of_cases} testcases for Testcase Type {Testcase_Type}")
    
    steps = []
    
    for i in trange(num_of_cases):
        x, primal_r, dual_r = Solver_Func(f"{Testcase_Dir}/TestCase_{Testcase_Type}_case{i}.txt")
        
        # Get the duality gap reduced by 50%, 75%, 90%, 95%, 99
        primal_val_stage1 = 1
        dual_val_stage1 = 1
        primal_stage1_index = next((i for i, x in enumerate(primal_r) if x <= primal_val_stage1), None)
        dual_stage1_index = next((i for i, x in enumerate(dual_r) if x <= dual_val_stage1), None)
        
        primal_val_stage2 = 0.1
        dual_val_stage2 = 0.1
        primal_stage2_index = next((i for i, x in enumerate(primal_r) if x <= primal_val_stage2), None)
        dual_stage2_index = next((i for i, x in enumerate(dual_r) if x <= dual_val_stage2), None)
        
        primal_val_stage3 = 0.01
        dual_val_stage3 = 0.01
        primal_stage3_index = next((i for i, x in enumerate(primal_r) if x <= primal_val_stage3), None)
        dual_stage3_index = next((i for i, x in enumerate(dual_r) if x <= dual_val_stage3), None)
        
        primal_val_stage4 = 0.001
        dual_val_stage4 = 0.001
        primal_stage4_index = next((i for i, x in enumerate(primal_r) if x <= primal_val_stage4), None)
        dual_stage4_index = next((i for i, x in enumerate(dual_r) if x <= dual_val_stage4), None)
        
        primal_val_stage5 = 1e-4
        dual_val_stage5 = 1e-4
        primal_stage5_index = next((i for i, x in enumerate(primal_r) if x <= primal_val_stage5), None)
        dual_stage5_index = next((i for i, x in enumerate(dual_r) if x <= dual_val_stage5), None)
        
        steps.append([
            [primal_stage1_index, dual_stage1_index],
            [primal_stage2_index, dual_stage2_index],
            [primal_stage3_index, dual_stage3_index],
            [primal_stage4_index, dual_stage4_index],
            [primal_stage5_index, dual_stage5_index]
        ])
    plt.figure(figsize = (10, 6))
    # Draw bar chart
    steps = np.array(steps)
    primal_steps = np.mean(steps[:, :, 0], axis = 0)
    dual_steps = np.mean(steps[:, :, 1], axis = 0)
    
    # Draw 2 bar charts with x = [50%, 75%, 90%, 95%, 99%]
    plt.bar(np.arange(5), primal_steps, width = 0.4, label = "Primal Residual")
    plt.bar(np.arange(5) + 0.4, dual_steps, width = 0.4, label = "Dual Residual")
    plt.legend()
    plt.xticks(np.arange(5) + 0.2, [1e0, 1e-1, 1e-2, 1e-3, 1e-4])
    plt.ylabel("Number of Iterations")
    
    plt.savefig(f"./Tests.png")
    

if __name__ == "__main__":
    #UnitTest(ADAL_Unittest, 1, 1)
    TypeTest(ADAL_UnitTest, 1)