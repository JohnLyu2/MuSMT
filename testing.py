import argparse
import json
from smt.SMTGame import SMTGame
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='Json with experiment design')
    args_cl = parser.parse_args()
    config = json.load(open(args_cl.json_config, 'r'))
    train_path = config['training_dir']
    val_path = config['validation_dir']
    smt_ext = config['file_ext']
    mv_str = config['tactics_config']['all_tactics']
    probe_lst = config['probes']
    pa_size = config['prior_action_embed_size']
    train_total_timeout = config['train_total_timeout']
    val_total_timeout = config['val_total_timeout']
    train_tactic_timeout = config["train_tactic_timeout"]
    val_tactic_timeout = config["val_tactic_timeout"]
    stats = config['probe_stats'] 
    coach_args = dotdict(config['coach_args'])
    g = SMTGame(benchmarkPath = train_path, ext = smt_ext, moves_str = mv_str, probes = probe_lst, stats = stats, total_timeout = train_total_timeout, tactic_timeout = train_tactic_timeout, prior_action_embed_size = pa_size, train = True)
    
    # testing timeout dependant caching
    b3 = g.getInitBoard(3)
    cache3 = dict()
    print("Initial Formula")
    print(b3.get_embedding())
    
    smt_id = 1
    print(f"Apply {mv_str[smt_id]} for ")
    b3_1 = b3.execute_move(smt_id, 1, cache3)
    print(b3_1)

    b3_2 = b3.execute_move(smt_id, 0.5, cache3)
    print(b3_2)

    b3_3 = b3.execute_move(smt_id, 5, cache3)
    print(b3_3)

    b3_4 = b3.execute_move(smt_id, 4, cache3)
    print(b3_4)

    b3_5 = b3.execute_move(smt_id, 15, cache3)
    print(b3_5)


if __name__ == "__main__":
    main()
