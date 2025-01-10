import os, sys


sys.path.append('..')
from reference import reference

n_seq = [5, 10, 50, 100, 500]

REFER = os.path.join('..', 'reference.py')
DATA_TYPE = ['InequalityQP', 'EqualityQP', 'MixedQP', 'TinyIRWA']

for data in DATA_TYPE:
    if data != 'TinyIRWA':
        for n in n_seq:
            print(f"\n\nTesting {data} with n = {n}")
            reference(os.path.join('Dataset', data, f'{n}.txt'))
    else:
        FILE = os.path.join('Dataset', data, f'{300}.txt')
        print(f"Testing {data} with n = 300")
        reference(FILE)