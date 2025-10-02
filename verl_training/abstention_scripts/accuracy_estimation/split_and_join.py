import argparse
import os
import json
from vllm import LLM, SamplingParams
import torch
import pickle
import math

PARENT_DIR = 'data/model_accuracy_estimates/preds'

def split_dataset(args):
    dataset_path = os.path.join(PARENT_DIR, f'{args.dataset_name}_preds.jsonl')
    datapoints = []
    with open(dataset_path, 'r') as f:
        for line in f:
            datapoints.append(json.loads(line.strip()))
    
    chunk_size = math.ceil(len(datapoints) / args.num_splits)
    for chunk_num in range(args.num_splits):
        curr_chunk = datapoints[chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        chunk_path = os.path.join(PARENT_DIR, f'{args.dataset_name}_preds_split_{chunk_num}.jsonl')
        with open(chunk_path, 'w') as f:
            for dp in curr_chunk:
                f.write(json.dumps(dp) + '\n')

def join_dataset(args):
    datapoints = []
    for i in range(args.num_splits):
        dataset_path = os.path.join(PARENT_DIR, f'{args.dataset_name}_preds_split_{i}.jsonl')
        with open(dataset_path, 'r') as f:
            for line in f:
                datapoints.append(json.loads(line.strip()))

    print(len(datapoints))
    dataset_path = os.path.join(PARENT_DIR, f'{args.dataset_name}_preds.jsonl')
    with open(dataset_path, 'w') as f:
        for dp in datapoints:
            f.write(json.dumps(dp) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--num_splits', type=int, default=128)
    parser.add_argument('--action', type=str, default="")
    args = parser.parse_args()
    assert(args.action != "")

    if args.action == "split":
        split_dataset(args)
    else:
        join_dataset(args)


if __name__ == "__main__":
    main()


