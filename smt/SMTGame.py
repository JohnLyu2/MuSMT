from __future__ import print_function
import sys
sys.path.append('..')
# from .SMTLogic import CacheTreeNode
from .SMTLogic import Board
import os
import pathlib
import copy
from z3 import * # may not need

import functools
print = functools.partial(print, flush=True)

"""
Game class implementation for SMT solving.
"""

ACC_TIME_IN_EMBEDDING = True

MODEL_OUT_FEATURES = 768
MIN_SOLVED_REWARD = 0.5
WIN_REWARD = 1
STEP_EXCEED_REWARD = -1
LOSE_REWARD = -1
MAX_STEP = 18

class SMTGame():
    def __init__(self, benchmarkPath, ext, no_suc, moves_str, probes, stats, total_timeout, tactic_timeout_lst, prior_action_embed_size, train, max_step = None):
        self.train = train
        self.bPath = benchmarkPath
        self.ext = ext
        self.no_suc = no_suc
        self.formulaLst = []
        self.total_timeout = total_timeout
        self.tactic_timeout_lst = tactic_timeout_lst
        self.max_step = max_step
        self.moves_str = moves_str
        self.probes = copy.deepcopy(probes)
        if len(self.probes) == 0: self.probes.append(None) # padding 0 if no probe
        if ACC_TIME_IN_EMBEDDING: self.probes.append("acc_time")
        self.prior_action_embed_size = prior_action_embed_size
        self.stats = stats
        self.action_size = len(moves_str)
        assert(self.action_size > 1)
        for f in sorted(list(pathlib.Path(self.bPath).rglob(f"*.{self.ext}"))): 
            self.formulaLst.append(str(f))
        self.fSize = len(self.formulaLst)

        self.solveLst = [False] *  self.fSize # recording whether a formula in the list has ever been solved
        if self.fSize < 1: raise Exception("No smt file in the folder")

    # def _make_representation(self): # TODO: smt
    #     return Board(self.formulaLst[self.curFmID], self.moves_str)

    def get_copy(self): # verified that a deep copy is not required for smt game
        return self
        # copy.deepcopy(self)

    def getBenchmarkSize(self):
        return self.fSize

    def is_solvable(self, id):
        return self.solveLst[id]

    # need to change the overridden function signature in Game.py?
    def getInitBoard(self, id):
        assert(id < self.fSize) # may consider reiterate from the beginning when id >= size
        tnode = None
        bd = Board(id, self.formulaLst[id], self.no_suc, self.moves_str, self.probes, self.stats, self.total_timeout, self.prior_action_embed_size, self.train)
        return bd

    def getBoardSize(self):
        return len(self.probes)

    def getActionSize(self):
        # return number of actions
        # return len(board.get_legal_moves())
        return self.action_size 

    def getGameEnded(self, board):
        if board.is_win():
            self.solveLst[board.id] = True
            if board.get_acc_time() > self.total_timeout: return MIN_SOLVED_REWARD
            return WIN_REWARD - (board.get_acc_time()/self.total_timeout)*(WIN_REWARD - MIN_SOLVED_REWARD)
        if self.train:
            if board.step > self.max_step: return STEP_EXCEED_REWARD
        if board.is_timeout() or board.is_error():
            return LOSE_REWARD
        return 0 # game not over yet
