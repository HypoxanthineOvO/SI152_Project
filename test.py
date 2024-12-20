import numpy as np
from scipy.linalg import norm
from scipy.optimize import minimize

def projection_onto_ci(s, ci):
    """
    投影到集合 Ci
    """
    if ci == 'non-negative':
        return np.maximum(s, 0)
    elif ci == 'zero':
        return 0
    else:
        raise ValueError("Unsupported set Ci")

def dist2(s, ci):
    """
    计算 s 到集合 Ci 的距离的平方
    """
    projection = projection_onto_ci(s, ci)
    return norm(s - projection) ** 2

def lp_subproblem(xk, p, uk, mu, A, b, Ci):
    """
    解决 Lp(xk, p, uk, mu) 子问题
    """
    n = xk.shape[0]
    pk_next = np.zeros_like(p)
    
    for i in range(n):
        sk = A[i] @ xk + b[i] + mu * uk[i]
        dist_sk_ci = dist2(sk, Ci[i])
        
        if dist_sk_ci <= mu:
            pk_next[i] = projection_onto_ci(sk, Ci[i])
        else:
            term = mu / dist_sk_ci * (sk - projection_onto_ci(sk, Ci[i]))
            pk_next[i] = sk - term
    
    return pk_next
def project_to_halfspace(s: np.ndarray) -> np.ndarray:
    """
    Project s to the halfspace defined by x <= 0
    """
    proj_s = np.maximum(s, 0)
    return proj_s
    
def project_to_hyperplane(s: np.ndarray) -> np.ndarray:
    """
    Project s to the hyperplane defined by 0
    """
    proj_s = 0
    return proj_s
# 示例数据
n = 3
xk = np.array([1.0, 2.0, 3.0])
p = np.array([0.0, 0.0, 0.0])
uk = np.array([1.0, 1.0, 1.0])
mu = 1.1
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = np.array([0, 0, 0])
Ci = ['non-negative', 'non-negative', 'non-negative']

pk_next = lp_subproblem(xk, p, uk, mu, A, b, Ci)


###
s = A @ xk + b + mu * uk
print(s)
m = 3
num_inequ = 3
num_equ = 0
p_new = np.zeros(m)
for i in range(m):
    si = s[i]
    if i < num_inequ:
        # Inequality Constraints
        ## Projection to halfspace
        dist = 0
        if si > 0:
            dist = si ** 2
        Project_s = project_to_halfspace(si)
        if dist <= mu:
            p_new[i] = Project_s
        else:
            p_new[i] = si - mu / dist * (si - Project_s)
    else:
        # Equality Constraints
        dist = np.square(si)
        Project_s = project_to_hyperplane(si)
        if dist <= mu:
            p_new[i] = Project_s
        else:
            p_new[i] = si - mu / dist * (si - Project_s)
print(p_new)
###
print("p_{k+1}:", pk_next)