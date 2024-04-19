import logging
import math
import time
import csv

import numpy as np

from smt.NNet import NNetWrapper as snn

from smt.SMTGame import LOSE_REWARD

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, args, cache, log_filepath, embed_logfile):
        self.log_filepath = log_filepath # current no option to not log
        self.log_embed = args.log_embed
        self.embedding_logfile = embed_logfile
        self.args = args
        self.nnet = nnet
        self.cache = cache
        self.log_to_file = self.args.log_to_file
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, game, board, temp=1, verbose=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from the input board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        with open(self.log_filepath,'a+') as f:
            f.write(f"\n\n\nGetActionProb() starts:\n")
        for i in range(self.args.numMCTSSims):
            if self.log_to_file:
                with open(self.log_filepath,'a+') as f:
                    f.write(f"\n\nStart Sim no. {i}\n")
                    f.write(f"Starting Board:\n{board}\n\n")
            self.search(game, board)

        s = board.get_mcts_rep()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(game.getActionSize())]
        if self.log_to_file:
            f = open(self.log_filepath,'a+')
            f.write(f"After search, counts: {counts}\n")
            f.close()

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        if self.log_to_file:
            f = open(self.log_filepath,'a+')
            f.write(f"After temp, counts: {counts}\n")
            f.close()
        counts_sum = float(sum(counts))
        # assert counts_sum > 0 otherwise print counts
        assert counts_sum > 0, f"counts_sum is {counts_sum}, counts is {counts}\n{board}\nBoard\n{board}\n"    
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, game, board, verbose=False, level=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        # start_time = time.time()
        #print("1: ", time.time()-start_time)

        s = board.get_mcts_rep()
        # if self.log_to_file:
            # with open(self.log_filepath,'a+') as f:
            #     f.write(f"\nStart search with {board}\n\n")

        # if verbose:
        #     log.info(f"At level {level}\n{s}")

        if s not in self.Es: # STEP 2: EXPANSION
            if verbose:
                log.info(f"Node not yet seen\n{s}")
            if self.log_to_file:
                with open(self.log_filepath,'a+') as f:
                    f.write(f"\nNode not yet seen\n")
            self.Es[s] = game.getGameEnded(board)

        #print("2: ", time.time()-start_time)

        if self.Es[s] != 0: # STEP 4: BACKPROPAGATION
            # terminal node
            if self.log_to_file:
                f = open(self.log_filepath,'a+')
                f.write(f"Search reach final board {board}\n")
                f.write(f"Actions: {board.priorActions}\n")
                f.write(f"Game over: Return {game.getGameEnded(board)}\n\n")
                f.close()
            # if verbose:
            #     log.info(f"Node is terminal node, reward is {self.Es[s]}\n{s}")
            return game.getGameEnded(board)

        #print("3: ", time.time()-start_time)

        if s not in self.Ps: # STEP 3: ROLLOUT or SIMULATION (use NN to predcit the value, i.e., the end reward to be backpropagated)
            # leaf node
            if verbose:
                log.info(f"Node is leaf node, using NN to predict value for\n{s}")
            with open(self.log_filepath,'a+') as f:
                f.write(f"Node is leaf node, using NN to predict value\n")
            try:
                s_embedding = board.get_embedding()
            except:
                self.Es[s] = LOSE_REWARD
                return self.Es[s]
            if self.log_embed:
                with open(self.embedding_logfile, 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow(s_embedding)
            self.Ps[s], v = self.nnet.predict(s_embedding) # plays a role in calculating UCB too
            valids = board.get_legal_moves()
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            with open(self.log_filepath,'a+') as f:
                f.write(f"ready to return v in leaf\n")
            return v # STEP 4: BACKPROPAGATION

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        if verbose:
            log.info(f"Pick an action")
        for a in range(game.getActionSize()): # STEP 1: SELECTION
            if valids[a]:
                if (s, a) in self.Qsa:
                    if verbose:
                        log.info(f"Exists in Qsa")
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    if verbose:
                        log.info(f"Does not exist in Qsa")
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_board = board.execute_move(a, game.tactic_timeout_lst, self.cache)

        if verbose:
            log.info(f"Non-leaf node, considering action ({a}) resulting in \n{next_board}\n")

        if self.log_to_file:
            with open(self.log_filepath,'a+') as f:
                f.write(f"Non-leaf node, considering action {a} resulting in \n{next_board}\n")
        #print("6: ", time.time()-start_time)

        v = self.search(game, next_board, level=level+1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
