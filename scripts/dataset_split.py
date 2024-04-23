import argparse
import os
import random
import shutil
import pathlib

def create_dir(target_dir, files):
    counter = 0
    for source_filepath in files:
        # file = pathlib.Path(source_filepath).name
        target_filepath = os.path.join(target_dir, f"i{counter}.smt2")
        os.symlink(source_filepath, target_filepath)
        counter += 1

def main():
    parser = argparse.ArgumentParser(description='Create split dataset from files in the folder.')
    parser.add_argument('--split_size', type=int, required=True, help='Size of formulas that should go into the first set; the rest goes into test')
    parser.add_argument('--benchmark_dir', type=str, required=True, help='Benchmark directory')
    parser.add_argument('--dataset_dir', type=str, required=True, help='The directory that stores the created dataset')
    parser.add_argument('--first_name', type=str, default='1', help='The directory name of the first set')
    parser.add_argument('--second_name', type=str, default='2', help='The directory name of the second set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    benchmark_dir = args.benchmark_dir
    assert os.path.exists(benchmark_dir), 'The specified benchmark folder does not exist!'
    
    all_files = []
    for file in sorted(list(pathlib.Path(benchmark_dir).rglob("*.smt2"))):
        all_files.append(str(file))
    all_file_size = len(all_files)
    random.shuffle(all_files)

    first_size = args.split_size
    assert first_size <= all_file_size, 'The size of the first split set is larger than the total smt2 file size'

    dataset_dir = args.dataset_dir
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    first_dir = os.path.join(dataset_dir, args.first_name)
    second_dir = os.path.join(dataset_dir, args.second_name)

    os.mkdir(first_dir)
    os.mkdir(second_dir)

    create_dir(first_dir, all_files[:first_size])
    create_dir(second_dir, all_files[first_size:])

# sample usage: python scripts/dataset_split.py --split_size 30 --benchmark_dir benchmarks/cinteger10 --dataset_dir benchmarks/cinteger10
if __name__ == '__main__':
    main()