import logging
import os
import sys
import copy
import json 
import ijson
from ijson.common import ObjectBuilder
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import time
import datetime
import numpy as np
from tqdm import tqdm
import multiprocessing

from Arena import PlanningArena
from MCTS import MCTS
from smt.NNet import NNetWrapper as snn

log = logging.getLogger(__name__)

import functools
print = functools.partial(print, flush=True)

EPISODE_TIMEOUT = 150 # in minutes # may delete later

def streamReadFromJson(json_path):
    out_dict = {}
    with open(json_path, "rb") as f:  
        key = "-" # must be a string type; otherwise the prefix,startswith will report error
        builder = None
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix == '' and event == 'map_key':  # found new object at the root
                if key != "-":
                    out_dict[key] = builder.value # update the output dictionary for last map key
                key = value # mark the key value
                builder = ObjectBuilder()
            elif prefix.startswith(key):  # while at this key, build the object
                builder.event(event, value)
            elif prefix == '' and event == 'end_map':  # found the end of an object at the current key, yield
                out_dict[key] = builder.value
    return out_dict

class EpisodeExecutor(multiprocessing.Process):
    def __init__(self, game, args, nnet, queue, id, json_cache, cache_folder, log_to_file, log_folder):
        multiprocessing.Process.__init__(self)
        self.game = game
        self.id = id
        self.json_cache = json_cache
        self.cache_path = cache_folder + str(id) + ".json"
        self.args = args
        self.nnet = nnet
        self.q = queue
        self.log_to_file = log_to_file
        self.log_file = log_folder + str(id) + ".log"
        self.embed_logfile = log_folder + str(id) + "embed" + ".csv"
        self.solvable = multiprocessing.Value('i',0)
        self.sample_size = multiprocessing.Value('i',0)
        self.eps_time = multiprocessing.Value('i',0)

    def run(self):
        episode_before_time = time.time()
        cache = dict()
        if self.json_cache:
            if os.path.exists(self.cache_path):
                try:
                    cache = streamReadFromJson(self.cache_path)
                except:
                    log.warning(f"Formula {self.id} fails to read the json cache")
            else:
                with open(self.cache_path, 'x') as f:
                    pass
        mcts = MCTS(self.nnet, self.args, cache, self.log_file, self.embed_logfile)
        board = self.game.getInitBoard(self.id)
        episodeStep = 0
        trainExamples = []
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            # log.info(f"Looking for next action on board\n{canonicalBoard}")
            pi = mcts.getActionProb(self.game, board, temp=temp)
            trainExamples.append([board.get_embedding(), pi, None]) # store the embedding of the board
            action = np.random.choice(len(pi), p=pi)
            # log.info(f"Taking action {action}")
            if self.log_to_file:
                with open(self.log_file,'a+') as f:
                    f.write(f"After simulations, take action {action}\n")
            board = board.execute_move(action, self.game.tactic_timeout_lst, cache)

            r = self.game.getGameEnded(board)

            if r != 0:
                # log.info(f"Final board\n{board} with reward {r}")
                train_samples = [(x[0], x[1], r) for x in trainExamples] # update the reward for the previous moves
                self.sample_size.value = len(train_samples)
                if (self.game.is_solvable(self.id)):
                    self.solvable.value = 1
                    self.q.put((self.id, train_samples))
                episode_after_time = time.time()
                self.eps_time.value = int((episode_after_time - episode_before_time)/60)
                if self.log_to_file:
                    with open(self.log_file,'a+') as f:
                        f.write(f"Final board {board}\n")
                        f.write(f"Actions: {board.priorActions}\n")
                        f.write(f"Game over: Return {r}\n")
                        f.write(f"Episode Total Time: {self.eps_time.value} minutes\n")
                break
        if self.json_cache:
            with open(self.cache_path, 'w') as f:
                try:
                    json.dump(cache, f)
                except:
                    pass

    def collect(self):
        if self.is_alive(): self.terminate()
        return self.solvable.value, self.sample_size.value, self.eps_time.value

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, game_validation, nnet, args, log_folder):
        self.game = game
        self.game_validation = game_validation
        self.args = args
        self.nnet = nnet
        # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.sample_size = self.args.sample_number_val
        self.log_to_file = self.args.log_to_file
        self.train_batch = self.args.train_batch
        self.val_batch = self.args.val_batch
        self.log_folder = log_folder
        self.json_cache = self.args.json_cache
        self.cache_folder = log_folder + "cache/"
        os.makedirs(os.path.dirname(self.cache_folder))
        self.nnet_folder = log_folder + "nnet/"
        # self.mcts = MCTS(self.nnet, self.args, self.filename)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        # self.cacheList = [dict()] * self.game.fSize
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        multiprocessing.set_start_method('spawn')
        prewards = None
        for i in range(1, self.args.numIters + 1):
            log.info(f"Iteration {i} starts")
            # bookkeeping
            # examples of the iteration
            iterLogFolder = self.log_folder + str(i) + "/"
            os.makedirs(os.path.dirname(iterLogFolder), exist_ok=True)
            if not self.skipFirstSelfPlay or i > 1:
                sample_size_queue = 0
                sLst = []
                sample_size_total = 0
                max_eps_time = 0
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                q = multiprocessing.Queue()
                for j in tqdm(range(0, self.args.numEps, self.train_batch), desc="Batch Self Play"):
                    batch_instance_ids = range(j, min(j+self.train_batch, self.args.numEps))
                    processes = []
                    for id in batch_instance_ids:
                        processes.append(EpisodeExecutor(self.game, self.args, self.nnet, q, id, self.json_cache, self.cache_folder, self.log_to_file, iterLogFolder))
                    for process in processes:
                        process.start()
                    t1 = time.time()
                    while True:
                        any_running = any(p.is_alive() for p in processes)
                        while (not q.empty()):
                            id, trainExamples = q.get()
                            self.game.solveLst[id] = True
                            iterationTrainExamples += trainExamples
                            sample_size_queue += len(trainExamples)
                        if not any_running: break
                    for process in processes:
                        t2 = time.time()
                        process.join(max(1, EPISODE_TIMEOUT * 60 - (t2 - t1))) #the timeout does not mean anything now
                    for process in processes:
                        if process.is_alive(): process.terminate()
                        solvable, sample_size, eps_time = process.collect()
                        sLst.append(solvable)
                        if solvable: sample_size_total += sample_size
                        if eps_time > max_eps_time: max_eps_time = eps_time

                log.info(f"Longest episode time: {max_eps_time} mins")

                # save the iteration examples to the history


                self.trainExamplesHistory.append(iterationTrainExamples)

                rew = [e[2] for e in iterationTrainExamples]
                # mean, min, std and max of the rewards
                log.info(f"REWARDS - Mean: {np.mean(rew):.4f}, Std: {np.std(rew):.4f}, Min: {np.min(rew):.4f}, Max: {np.max(rew):4f}")

                log.info(f"Data collected from Queue, {len([x for x in self.game.solveLst if x])} are solvable;\nlist: {self.game.solveLst}\nNum of training samples: {sample_size_queue}\n")
                log.info(f"Data collected from Processes, {len([x for x in sLst if x])} are solvable;\nlist: {sLst}\nNum of training samples: {sample_size_total}\n")

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            trainExamples = self.prepareTrainExamples()

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.nnet_folder, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.nnet_folder, filename='temp.pth.tar')

            self.nnet.train(trainExamples)

            log.info('PITTING AGAINST PREVIOUS VERSION')

            val_log_file = iterLogFolder + "val.log"

            if prewards is None:
                log.info('Previous NN results:')
                if self.log_to_file:
                    f = open(val_log_file,'a+')
                    f.write("Val using pre nnet\n")
                    f.close()
                arena = PlanningArena(self.pnet, self.game_validation, log_to_file=self.log_to_file, log_file=val_log_file, iter=self.sample_size, val_batch=self.val_batch)
                prewards = arena.playGames(self.args.arenaCompare, verbose=False)
            log.info('New NN results:')
            if self.log_to_file:
                f = open(val_log_file,'a+')
                f.write("Val using new nnet\n")
                f.close()
            arena = PlanningArena(self.nnet, self.game_validation, log_to_file=self.log_to_file, log_file=val_log_file, iter=self.sample_size, val_batch=self.val_batch)
            nrewards = arena.playGames(self.args.arenaCompare, verbose=False)

            log.info(f"NEW/PREV WINING COUNTS : {nrewards} / {prewards}")
            if (nrewards[0] < prewards[0]) or ((nrewards[0] == prewards[0]) and (nrewards[1] >= prewards[1])):
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.nnet_folder, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                prewards = nrewards
                self.nnet.save_checkpoint(folder=self.nnet_folder, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.nnet_folder, filename='best.pth.tar')

    def prepareTrainExamples(self):

        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)

        # all training examples (not only the last iteration)
        trainExamples = [(e[0], e[1], e[2]) for e in trainExamples]

        # shuffle examples before training
        shuffle(trainExamples)

        return trainExamples

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.nnet_folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed #closed?

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            # r = input("Continue? [y|n]")
            # if r != "y":
            #     sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
