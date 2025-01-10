import os, sys


sys.path.append('..')
from reference import reference

n_seq = [5, 10, 50, 100, 200]

DATA_TYPE = ['InequalityQP', 'EqualityQP', 'MixedQP', 'TinyIRWA']

for data in DATA_TYPE:
    SOL_DIR = os.path.join('Dataset', data, 'Reference.txt')
    if os.path.exists(SOL_DIR):
        os.remove(SOL_DIR)
    
    
    if data != 'TinyIRWA':
        for n in n_seq:
            for i in range(1, 4):
                FILE = os.path.join('Dataset', data, f'{n}_{i}.txt')
                print(f"\nTesting {data} with n = {n}")
                sol = reference(FILE)
                with open(SOL_DIR, 'a') as f:
                    f.write(f"{n}_{i}.txt: {sol}\n")
    else:
        for i in range(1, 4):
            FILE = os.path.join('Dataset', data, f'300_{i}.txt')
            print(f"\nTesting {data} with n = 300")
            sol = reference(FILE)
            with open(SOL_DIR, 'a') as f:
                f.write(f"300_{i}.txt: {sol}\n")