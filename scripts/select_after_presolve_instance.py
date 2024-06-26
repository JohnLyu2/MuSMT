import os
import csv
import subprocess

BENCH_DIR = "/home/z52lu/z3alpha2/benchmarks/LassoRanker/valid"
DICT_PATH = "/home/z52lu/z3strategy_cc/neurips24/LassoRanker/z3.csv"
TARGET_DIR = "/home/z52lu/z3alpha2/benchmarks/LassoRanker/valid10"
THESHOLD = 10

if not os.path.exists(TARGET_DIR):
    # Create the directory
    os.makedirs(TARGET_DIR)

res_dict = {}
solve_in_threshold = 0

with open(DICT_PATH, 'r') as f:
  reader = csv.reader(f)
  next(reader)
  for row in reader:
    path = row[1]
    trimed_path = path.split("smt_lib/", 1)[1]
    solved = True if row[2] == "True" else False
    time = float(row[3])
    if solved and time <= THESHOLD: solve_in_threshold+=1
    res_dict[trimed_path] = (solved, time)
dict_size = len(res_dict)
print(f"solved within {THESHOLD} sec/total size: {solve_in_threshold}/{dict_size}")
print(f"retain rate: {(1-solve_in_threshold/dict_size)*100:.2f}%")


for file in os.listdir(BENCH_DIR):
    file_path = os.path.join(BENCH_DIR, file)
    if os.path.islink(file_path):
        bench_path = os.path.realpath(file_path)
        print(bench_path)
        bench_path_str = str(bench_path).split("smt_lib/", 1)[1]
        # print(bench_path_str)
        assert bench_path_str in res_dict, f"{bench_path_str} not in res_dict"
        if (not res_dict[bench_path_str][0]) or res_dict[bench_path_str][1] > THESHOLD:
            command = ["cp", "-P", file_path, TARGET_DIR]
            try:
                subprocess.run(command, check=True)
                print(f"Symlink {file_path} copied successfully.")
            except subprocess.CalledProcessError:
                print(f"Failed to copy the symlink {file_path}.")