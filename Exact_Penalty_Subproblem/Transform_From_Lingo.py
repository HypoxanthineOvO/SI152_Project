import os, sys

if __name__ == "__main__":
    with open("./Lingo_Out.log", "r") as f:
        raws = f.readlines()
    lines = []
    variable_id = 0
    for i, raw in enumerate(raws):
        if raw.strip().startswith("Variable"):
            variable_id = i
            break
        lines.append(raw.strip().replace(" ", ""))
    print("\n".join(lines))
    print("Variable")
    
    num_of_variables = 10
    variables = [0 for _ in range(num_of_variables)]
    for i in range(variable_id, len(raws)):
        # start: space space ... X{i} space space ... value space space ...
        line = raws[i].strip()
        if line == "":
            break
        if line.startswith("X"):
            line_split = line.split()
            idx = int(line_split[0][1:])
            value = float(line_split[-2])
            print(f"X[{idx}] = {value}")