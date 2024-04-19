import pathlib
import csv
import multiprocessing
import time
import argparse
import json

from Runner import Runner
from smt.SMTLogic import Board
from smt.NNet import NNetWrapper as snn
from smt.SMTGame import SMTGame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='Json with evaluation settings')
    args = parser.parse_args()
    config = json.load(open(args.json_config, 'r'))

    BATCH_SIZE = config["BATCH_SIZE"]
    Z3_PRE_TIME = config["Z3_PRE_TIME"]
    START_ID = config["START_ID"]
    TOTAL_TIMEOUT = config["TOTAL_TIMEOUT"]
    TACTIC_TIMEOUT_LST = config["TACTIC_TIMEOUT_LST"]
    PRIOR_ACTION_EMBED_SIZE = config["PRIOR_ACTION_EMBED_SIZE"]
    NNET_FOLDER = config["NNET_FOLDER"]
    NNET_FILE = config["NNET_FILE"]
    BENCHMARK_PATH = config["BENCHMARK_PATH"]
    MOVES = config["MOVES"]
    PROBES = config["PROBES"]
    NO_SUC = config["NO_SUC"]
    RESULT_FILE = config["RESULT_FILE"]
    STATS = config["STATS"]
    HEADER = config["HEADER"]

    multiprocessing.set_start_method('spawn')

    g = SMTGame(benchmarkPath = BENCHMARK_PATH, ext = "smt2", no_suc = NO_SUC, moves_str = MOVES, probes = PROBES, stats = STATS, total_timeout = TOTAL_TIMEOUT, tactic_timeout_lst = TACTIC_TIMEOUT_LST, prior_action_embed_size = PRIOR_ACTION_EMBED_SIZE, train = False)

    nnet = snn(g)
    nnet.load_checkpoint(NNET_FOLDER, NNET_FILE)

    # having this part is not elegant
    fLst = []
    for fm in sorted(list(pathlib.Path(BENCHMARK_PATH).rglob("*.smt2"))):
        fLst.append(str(fm))

    num = len(fLst)
    q = multiprocessing.Queue()

    with open(RESULT_FILE, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i in range(START_ID, num, BATCH_SIZE):
            batch_instance_ids = range(i,min(i+BATCH_SIZE, num))
            processes = []
            for id in batch_instance_ids:
                board = g.getInitBoard(id)
                processes.append(Runner(nnet, board, TOTAL_TIMEOUT, TACTIC_TIMEOUT_LST, q, Z3_PRE_TIME))
            for process in processes:
                process.start()
            t1 = time.time()
            while True:
                any_running = any(p.is_alive() for p in processes)
                while (not q.empty()):
                    id, path, result, rlimit, time_res, nn_time, solver_time, log_info = q.get()
                    sol_res = [id, path, result, rlimit, time_res, nn_time, solver_time]
                    print(sol_res)
                    writer.writerow(sol_res)
                    f.flush()
                if not any_running: break
            for process in processes:
                t2 = time.time()
                process.join(max(1, TOTAL_TIMEOUT + TACTIC_TIMEOUT_LST[-1][1] - (t2-t1) + 10))

if __name__ == "__main__":
    main()
