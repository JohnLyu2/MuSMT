from z3 import *
import numpy as np
import copy
import time


def get_rlimit(tmpSolver):
    stats = tmpSolver.statistics()
    for i in range(len(stats)):
        if stats[i][0] == 'rlimit count':
            return stats[i][1]
    return 0

def toSMT2Benchmark(f, status="unknown", name="benchmark", logic=""):
    v = (Ast * 0)()
    return Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())

class Board(): # Keep its name as Board for now; may call it goal later
    def __init__(self, ID, formulaPath, no_suc, moves_str, probes, stats, total_timeout, prior_action_embed_size, train):
        "Set up initial board configuration."
        self.train = train
        self.id = ID
        self.fPath = formulaPath
        self.no_suc = no_suc
        self.moves_str = moves_str
        self.probes = probes
        self.prior_action_embed_size = prior_action_embed_size
        self.stats = stats
        self.total_timeout = total_timeout
        with open(self.fPath, 'r') as f: #may write to a temp file later
            self.curGoal = f.read()
        self.step = 0 # number of times tactics have already been applied
        self.priorActions = [] # store list of tactic strings
        self.win = False
        self.res = None # "SAT" or "UNSAT"
        self.failed = False
        self.nochange  = False
        self.tac_timeout = False
        self.accRLimit = 0 # machine-independent time
        self.accTime = 0 # time
        self.last_time = None # time for executing the last transformNextState()
        self.last_tac_time = None # time for executing the last tactic (in Z3)
        self.rlimit = None # rlimit for executing the last tactic
        self.use_cache = False # used cache or not in the last tactic application
        self.error = False

    def __str__(self): # when printing board object
        return f"fID: {self.id}; fPath: {self.fPath};\nEmbedding\n{self.get_embedding()}\nprior actions: {self.priorActions};\nstep: {self.step}; is_win: {self.is_win()}; is_nochange: {self.is_nochange()}; is_fail: {self.is_fail()}; is_tac_timeout: {self.is_tac_timeout()}; is_error: {self.is_error()}; use_cache: {self.use_cache}; accTime: {self.accTime:.1f}\nLast Move: {self.most_recent_act_str()}"

    # mixed shallow&deep copy
    def get_copy(self):
        copied = copy.copy(self)
        copied.priorActions = copy.copy(copied.priorActions)
        copied.failed = False
        copied.nochange = False
        copied.tac_timeout = False
        copied.last_time = None
        copied.last_tac_time = None
        copied.rlimit = None
        copied.use_cache = False
        return copied

    def get_mcts_rep(self):
        return str(self.priorActions)

    def get_legal_moves(self):
        if self.is_done():
            raise Exception("Game is already over")
        valids = [1] * len(self.moves_str)
        if self.no_suc:
            most_recent_action = self.get_most_recent_act() # if none, return -1
            if most_recent_action > -1: valids[most_recent_action] = 0
        return np.array(valids)
        
    # -1 if no prior action
    def get_most_recent_act(self):
        if len(self.priorActions) == 0: return -1
        return self.moves_str.index(self.priorActions[-1])

    def most_recent_act_str(self):
        if len(self.priorActions) == 0: return "No Prior Action"
        return f"{self.priorActions[-1]} with total time {self.last_time:.1f} and tactic time {self.last_tac_time:.1f}"

    def get_embedding(self):
        measureLst = []
        formula = z3.parse_smt2_string(self.curGoal) # error sometime; catch in MCTS.py and Runner.py
        goal = z3.Goal()
        goal.add(formula)
        for pStr in self.probes:
            if pStr is None: measure = 0  # padding 0 if no probe
            elif pStr == "acc_time": measure = self.accTime/self.total_timeout
            else:
                p = z3.Probe(pStr)
                measure = p(goal)
                if pStr in self.stats:
                    measure = (measure-self.stats[pStr][0])/(self.stats[pStr][1]-self.stats[pStr][0])
            measureLst.append(measure)
        priorActionsInt = [self.moves_str.index(act)+1 for act in self.priorActions] # +1 to avoid 0 (0 is reserved for padding)
        priorActionsInt = priorActionsInt[-(self.prior_action_embed_size):] # only keep the last PREV_ACTIONS_EMBED actions
        prior_actions_padded = [0] * (self.prior_action_embed_size - len(priorActionsInt)) + priorActionsInt 
        return np.array(measureLst + prior_actions_padded)

    def get_acc_time(self):
        return self.accTime

    def is_win(self):
        return self.win

    def is_fail(self):
        return self.failed

    def is_nochange(self):
        return self.nochange

    # is the last tactic timeouted
    def is_tac_timeout(self):
        return self.tac_timeout

    # def is_giveup(self):
    #     return self.step > STEP_UPPER_BOUND

    # episode timeout
    def is_timeout(self):
        return self.accTime > self.total_timeout

    def is_error(self):
        return self.error

    # merge with getGameEnded later
    def is_done(self):
        return self.is_win() or self.is_timeout() or self.is_error()

    # with the current caching design, timeout cannot be changed for a formula
    def transformNextState(self, move, timeout, cache):
        tmp = z3.Solver()
        self.priorActions.append(self.moves_str[move])
        rlimit_before = get_rlimit(tmp)
        time_before = time.time()
        formula = z3.parse_smt2_string(self.curGoal)
        pre_goal = z3.Goal()
        pre_goal.add(formula)
        try:
            pre_goal_str = str(pre_goal)
        except:
            self.error = True
            return
        key_pair = (pre_goal_str, move)
        key_pair_str = str(key_pair)
        if self.train and (key_pair_str in cache):
            cache_info = cache[key_pair_str]
            if cache_info[4] and (timeout > cache_info[5]):
                del cache[key_pair_str]
            else:
                self.use_cache = True
                if timeout < cache_info[5]: # may change this later
                    self.tac_timeout = True
                    self.last_time = timeout # for now
                    self.last_tac_time = timeout
                    self.rlimit = 0 # the rlimit here is incorrect
                else:
                    self.curGoal, self.win, self.nochange, self.failed, self.tac_timeout, self.last_time, self.last_tac_time, self.rlimit = cache_info
        if not self.use_cache:
            t = Tactic(self.moves_str[move])
            tTimed = TryFor(t, timeout * 1000)
            try:
                tac_time_before = time.time()
                new_goals = tTimed(pre_goal)
                self.last_tac_time = time.time() - tac_time_before
                assert(len(new_goals) == 1)
                new_goal = new_goals[0]
                new_goal_str = str(new_goal)
                if (new_goal_str == "[]") or (new_goal_str == "[False]"):
                    self.win = True
                    if new_goal_str == "[]": self.res = "SAT"
                    else: self.res = "UNSAT"
                if pre_goal_str == str(new_goal):
                    self.nochange = True
                else:
                    exp = new_goal.as_expr()
                    self.curGoal = toSMT2Benchmark(exp)
            except Z3Exception as e:
                self.last_tac_time = time.time() - tac_time_before
                message = (e.args[0]).decode()
                if message == "canceled": self.tac_timeout = True
                else: self.failed = True
            time_after = time.time()
            rlimit_after = get_rlimit(tmp)
            self.last_time = time_after - time_before
            self.rlimit = rlimit_after - rlimit_before
            if self.train:
                cache[key_pair_str] = [self.curGoal, self.win, self.nochange, self.failed, self.tac_timeout, self.last_time, self.last_tac_time, self.rlimit]
        self.accTime += self.last_time
        self.accRLimit += self.rlimit
        self.step += 1

    def execute_move(self, move, timeout_lst, cache):
        """Perform the given move on the board and return the result board
        """
        assert(not self.is_done())
        result = self.get_copy()
        for timeout_pair in timeout_lst:
            if self.accTime < timeout_pair[0]:
                timeout = timeout_pair[1]
                break
        assert(timeout > 0)
        result.transformNextState(move, timeout, cache)
        return result

    # currently time, rlimit, step in presolver is not counted in class fields;
    def presolve(self, timeout):
        s = z3.Solver()
        s.set("timeout", timeout * 1000)
        formula = z3.parse_smt2_string(self.curGoal)
        s.add(formula)
        res = str(s.check())
        if res == 'sat' or res == 'unsat':
            self.win = True
            if res == 'sat': self.res = "SAT"
            else: self.res = "UNSAT"
