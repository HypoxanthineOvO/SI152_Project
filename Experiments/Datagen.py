import os, sys


"""
Type of Data generated:

python Datagen.py 0 -> InequalityQP
python Datagen.py 1 -> EqualityQP
python Datagen.py 2 -> MixedQP
python Datagen.py 3 -> TinyIRWA
"""

n_seq = [5, 10, 50, 100, 200]
random_seed = 42

DATA_TYPE = ['InequalityQP', 'EqualityQP', 'MixedQP', 'TinyIRWA']
DATA_DIR = "Dataset"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        assert int(sys.argv[1]) < len(DATA_TYPE), f"Invalid data type: {sys.argv[1]}"
        FILE = os.path.join(DATA_DIR, DATA_TYPE[int(sys.argv[1])])
    else:
        FILE = os.path.join(DATA_DIR, 'InequalityQP')    
    
    if not os.path.exists(FILE):
        os.makedirs(FILE)
    
    if DATA_TYPE[int(sys.argv[1])] == 'InequalityQP':
        for n in n_seq:
            command = f"python {os.path.join('TestGen', '01_Inequality.py')} {n} {random_seed} {os.path.join(FILE, f'{n}.txt')}"
            os.system(command)
    elif DATA_TYPE[int(sys.argv[1])] == 'EqualityQP':
        for n in n_seq:
            command = f"python {os.path.join('TestGen', '02_Equality.py')} {n} {random_seed} {os.path.join(FILE, f'{n}.txt')}"
            os.system(command)
    elif DATA_TYPE[int(sys.argv[1])] == 'MixedQP':
        for n in n_seq:
            command = f"python {os.path.join('TestGen', '03_Mixed.py')} {n} {random_seed} {os.path.join(FILE, f'{n}.txt')}"
            os.system(command)
    elif DATA_TYPE[int(sys.argv[1])] == 'TinyIRWA':
        command = f"python {os.path.join('TestGen', '04_TinyIRWA.py')} --file {os.path.join(FILE, f'{300}.txt')} --seed {random_seed} --n {300} --m {300}"
        os.system(command)
    

        