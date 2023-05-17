import re
import sys

filename = sys.argv[1]
print(filename)
time_dict = {}
is_first_time = {}

with open(filename, 'r') as f:
    for line in f:
        match = re.search(r'(.*)\s+=\s+(\d+\.\d+)\s+s', line)
        if match:
            name = match.group(1)
            time = float(match.group(2))
            if name not in is_first_time:
                is_first_time[name] = True
                continue
            else:
                if name in time_dict:
                    time_dict[name] += time
                else:
                    time_dict[name] = time

print("Total time table:")
print(filename)
print("=======================================")
for name, time in sorted(time_dict.items(), key=lambda x: x[1], reverse=True):
    if name == "ExecutionTime":
        continue
    elif name == "CPU Time (get turb souce) " or name == "CPU Time (copy&permutate) " or name == "GPU Time                  ":
        print(f"  {name}: {time:.6f} s")
    else:
        print(f"{name}: {time:.6f} s")
print("=======================================")
