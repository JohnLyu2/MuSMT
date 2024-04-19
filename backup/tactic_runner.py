import argparse
from z3 import *

def get_rlimit(tmpSolver):
    stats = tmpSolver.statistics()
    for i in range(len(stats)):
        if stats[i][0] == 'rlimit count':
            return stats[i][1]
    return 0

def toSMT2Benchmark(f, status="unknown", name="benchmark", logic=""):
    v = (Ast * 0)()
    return Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='smt2 file path')
    parser.add_argument('tactic', type=str, help='tactic to apply')
    parser.add_argument('timeout', type=int, help='timeout for the tactic application in seconds')
    path = parser.parse_args().path
    tactic = parser.parse_args().tactic
    timeout = parser.parse_args().timeout
    print(f"file path: {path}")
    print(f"tactic: {tactic}")
    print(f"time_out: {timeout}")
    win = False
    nochange = False
    fail = False
    tmp = z3.Solver()
    rlimit_before = get_rlimit(tmp)
    formula = z3.parse_smt2_file(path)
    pre_goal = z3.Goal()
    pre_goal.add(formula)
    t = TryFor(Tactic(tactic), timeout*1000)
    try:
        new_goals = t(pre_goal)
        assert(len(new_goals) == 1)
        new_goal = new_goals[0]
        if (str(new_goal) == "[]") or (str(new_goal) == "[False]"): win = True
        if str(pre_goal) == str(new_goal):
            nochange = True
        else:
            e = new_goal.as_expr()
            out_goal_str = toSMT2Benchmark(e, logic="QF_NIA") #do not hardcode this logic later
    except Z3Exception:
            fail = True
    rlimit_after = get_rlimit(tmp)
    rlimit = rlimit_after - rlimit_before
    print(out_goal_str)

if __name__ == "__main__":
    main()
