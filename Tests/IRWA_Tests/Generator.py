import os, shutil
from tqdm import trange

num_testcases = [10, 10, 20]

if __name__ == "__main__":
    # Test 1
    shutil.rmtree("1", ignore_errors=True)
    os.makedirs("1", exist_ok=True)
    for i in trange(num_testcases[0]):
        os.system(f"python IRWA_1.py --file ./1/TestCase_1_case{i}.txt --n 1000 --m 600 > /dev/null")
    
    exit(0)
    # Test 2
    shutil.rmtree("2", ignore_errors=True)
    os.makedirs("2", exist_ok=True)
    for i in trange(num_testcases[1]):
        os.system(f"python IRWA_2.py ./2/TestCase_2_case{i}.txt > /dev/null")