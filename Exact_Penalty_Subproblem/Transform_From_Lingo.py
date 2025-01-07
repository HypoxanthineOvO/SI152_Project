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
    #print("\n".join(lines))
    #print("Variable")
    #print("\n====================\n")
    objective_value = -1
    for i in range(variable_id):
        line = raws[i].strip()
        if line.startswith("Objective value:"):
            objective_value = float(line[len("Objective value:"):].strip())
    
    num_of_variables = 0
    variables = [0 for _ in range(2000)]
    
    #print(f"Variables: {num_of_variables}")
    #print(f"Variable ID: {variable_id}, Total Lines: {len(raws)}")
    for i in range(variable_id, len(raws)):
        # start: space space ... X{i} space space ... value space space ...
        line = raws[i].strip()
        #print(f"[{line}]")
        if line == "":
            continue
        if line.startswith("X"):
            line_split = line.split()
            idx = int(line_split[0][1:])
            value = float(line_split[-2])
            variables[idx - 1] = value
            num_of_variables += 1
    variables = variables[:num_of_variables]
    
    print("ref: [", end = "")
    for i, value in enumerate(variables):
        print(f"{value}", end = "")
        if i != num_of_variables - 1:
            print(", ", end = "")
    print("]")
    print(f"ref_val: [{objective_value}]")