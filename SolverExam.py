import os, sys
from Submissions.QP_Solver import *
from reference import reference

n_seq = [5, 10, 50, 100]
DATA_TYPE = ['InequalityQP', 'EqualityQP', 'MixedQP', 'TinyIRWA']
SOLVERS = ['ADAL', 'IRWA', 'OSQP']
# SOLVERS = ['ADAL', 'OSQP']
NUM_PROBLEMS = 3
LOG_FILE = 'test_results.log'


with open(LOG_FILE, 'w') as log_file: 
    log_file.write("cfg_file, solver, sp_iter, obj, ref_obj, residual\n")


    for data in DATA_TYPE:
        
        if data != 'TinyIRWA':
        
            for n in n_seq:
                for i in range(1, NUM_PROBLEMS + 1):
                    for solver in SOLVERS:

                        cfg_file = os.path.join('Experiments', 'Dataset', data, f'{n}_{i}.txt')
        
                        n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_file)
                        
                        print(f"\n\nRunning {cfg_file} with {solver} solver")
                        
                        x, sp_iter = QP_solver(AE, AI, bE, bI, g, H, solver=solver)
                        if x is None:
                            log_file.write(f"{cfg_file}, {solver}, {sp_iter}, Failure\n")
                            continue
                        
                        
                        obj = 0.5 * x.T @ H @ x + g.T @ x
                        ref_x = reference(cfg_file)
                        ref_obj = 0.5 * ref_x.T @ H @ ref_x + g.T @ ref_x
                        residual = abs(obj - ref_obj)

                        log_file.write(f"{cfg_file}, {solver}, {sp_iter}, {obj}, {ref_obj}, {residual}\n")
        else:
            for solver in SOLVERS:
                for i in range(1, NUM_PROBLEMS + 1):
                    cfg_file = os.path.join('Experiments', 'Dataset', data, f'300_{i}.txt')
        
                    n, m, H, g, AI, bI, AE, bE = init_from_config(cfg_file)
                    
                    print(f"\n\nRunning {cfg_file} with {solver} solver")
                    
                    x, sp_iter = QP_solver(AE, AI, bE, bI, g, H, solver=solver)
                    if x is None:
                        log_file.write(f"{cfg_file}, {solver}, {sp_iter}, Failure\n")
                        continue
                    
                    
                    obj = 0.5 * x.T @ H @ x + g.T @ x
                    ref_x = reference(cfg_file)
                    ref_obj = 0.5 * ref_x.T @ H @ ref_x + g.T @ ref_x
                    residual = abs(obj - ref_obj)

                    log_file.write(f"{cfg_file}, {solver}, {sp_iter}, {obj}, {ref_obj}, {residual}\n")
