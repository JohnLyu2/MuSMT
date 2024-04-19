import argparse
import json
import logging
import datetime
import numpy as np
import coloredlogs

from Coach import Coach

# from qzero_planning.NNet import NNetWrapper as pnn
# from qzero_planning.PlanningGame import PlanningGame
# from qzero_planning.PlanningLogic import DomainAction, MinSpanTimeRewardStrategy, RelativeProductRewardStrategy

from smt.NNet import NNetWrapper as snn
from smt.SMTGame import SMTGame

from utils import *

import functools
print = functools.partial(print, flush=True)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def main():
    log_folder = "experiment_results/out-{date:%Y-%m-%d_%H-%M-%S}/".format(date=datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='Json with experiment design')
    args_cl = parser.parse_args()
    config = json.load(open(args_cl.json_config, 'r'))
    train_path = config['training_dir']
    val_path = config['validation_dir']
    smt_ext = config['file_ext']
    no_suc = config['no_suc']
    mv_str = config['tactics_config']['all_tactics']
    probe_lst = config['probes']
    pa_size = config['prior_action_embed_size']
    total_timeout = config['total_timeout']
    tactic_timeout_lst = config["tactic_timeout_lst"]
    training_max_step = config["training_max_step"]
    assert(tactic_timeout_lst[-1][0] > total_timeout) # ensure tactic timeouts are all defined in the list
    stats = config['probe_stats'] 
    coach_args = dotdict(config['coach_args'])
    log.info(f'Loading {SMTGame.__name__}...')
    g = SMTGame(benchmarkPath = train_path, ext = smt_ext, no_suc = no_suc, moves_str = mv_str, probes = probe_lst, stats = stats, total_timeout = total_timeout, tactic_timeout_lst = tactic_timeout_lst, prior_action_embed_size = pa_size, train = True, max_step = training_max_step)
    g_val = SMTGame(benchmarkPath = val_path, ext = smt_ext, no_suc = no_suc, moves_str = mv_str, probes = probe_lst, stats = stats, total_timeout = total_timeout, tactic_timeout_lst = tactic_timeout_lst, prior_action_embed_size = pa_size, train = False)

    log.info('Loading %s...', snn.__name__)
    nnet = snn(g)

    if coach_args.load_model:
        log.info('Loading checkpoint "%s/%s"...', coach_args.load_folder_file[0], coach_args.load_folder_file[1])
        nnet.load_checkpoint(coach_args.load_folder_file[0], coach_args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')


    # nnet.save_checkpoint(folder=nnet_folder, filename='best.pth.tar')

    log.info('Loading the Coach...')
    c = Coach(g, g_val, nnet, coach_args, log_folder)

    if coach_args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

    # accRlimit_all = np.array(g.accRlimit_all)
    # print(f"min accRlimit_all: {np.min(accRlimit_all)}, max accRlimit_all: {np.max(accRlimit_all)}, mean accRlimit_all: {np.mean(accRlimit_all)}, std accRlimit_all: {np.std(accRlimit_all)}")

if __name__ == "__main__":
    main()
