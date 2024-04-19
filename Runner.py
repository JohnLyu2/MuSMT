import multiprocessing
import numpy as np
import time
from z3 import *

class Runner(multiprocessing.Process):

    def __init__(self, nnet, board, total_timeout, tactic_timeout_lst, queue, z3_presolver_time): # may do something to log to a file later
        multiprocessing.Process.__init__(self)
        assert(z3_presolver_time >= 0)
        self.z3_pre_time = z3_presolver_time
        # self.no_suc = no_suc
        self.q = queue
        self.initialBoard = board # may not need this
        self.curBoard = self.initialBoard
        self.nnet = nnet
        self.total_timeout = total_timeout
        self.tactic_timeout_lst = tactic_timeout_lst
        self.action_size = len(board.moves_str)
        self.timeed_out = False
        self.nn_time = 0
        self.solver_time = 0

    # def stop(self):
    #     self._stop_event.set()
    #
    # def stopped(self):
    #     return self._stop_event.is_set()

    # no argument for game now
    def run(self):
        time_before = time.time() # think about how to accout for time better
        if self.z3_pre_time > 0:
            self.curBoard.presolve(self.z3_pre_time)
        # priorMove = -1
        while not self.curBoard.is_done():
            nn_before_time = time.time()
            try:
                s_embedding = self.curBoard.get_embedding()
            except:
                self.curBoard.error = True
            prob, v = self.nnet.predict(s_embedding)
            valids = self.curBoard.get_legal_moves()
            prob = prob * valids
            move = np.argmax(prob)
            # if self.no_suc and move == priorMove:
            #     move = np.argsort(prob)[-2]
            assert(move < self.action_size)
            # priorMove = move
            # no valid check since now all actions are valid
            nn_after_time = time.time()
            self.nn_time += nn_after_time - nn_before_time
            self.curBoard = self.curBoard.execute_move(move, self.tactic_timeout_lst, None)
            self.solver_time += time.time() - nn_after_time
            acc_time = time.time() - time_before
            if acc_time > self.total_timeout:
                self.timeed_out = True
                break
        log_text = f"{self.curBoard}\nActions: {self.curBoard.priorActions}\n"
        if self.timeed_out:
            log_text += "timed out\n\n"
            res_tuple = (self.initialBoard.id, self.initialBoard.fPath, None, None, None, None, None, log_text)
        elif self.curBoard.is_win():
            res = self.curBoard.res
            rlimit = self.curBoard.accRLimit
            total_time = time.time() - time_before
            log_text += f"Result: {res}, rlimit: {rlimit}, time: {total_time}\n\n"
            res_tuple = (self.initialBoard.id, self.initialBoard.fPath, res, rlimit, total_time, self.nn_time, self.solver_time, log_text)
        elif self.curBoard.is_error():
            log_text += "Error in parsing\n\n"
            res_tuple = (self.initialBoard.id, self.initialBoard.fPath, None, None, None, None, None, log_text)
        else:
            log_text += "WHAT???\n\n"
            res_tuple = (self.initialBoard.id, self.initialBoard.fPath, None, None, None, None, None, log_text)
        self.q.put(res_tuple)
