import numpy as np
import os, sys
from .Exact_Penalty_Subproblem.ADAL import ADAL


def QP_solver(AE: np.ndarray, AI: np.ndarray, bE: np.ndarray, bI: np.ndarray, 
              g: np.ndarray, H: np.ndarray,
              solver: str = "ADAL"):
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
    
    
    # x for iteration
    x = np.zeros(n)
    
    
    # Generate A and l,u from AI, bI, AE, bE
    if (solver == "ADAL"):
        A = np.zeros((m, n))
        b = np.zeros(m)
    
    
    
    
    return x