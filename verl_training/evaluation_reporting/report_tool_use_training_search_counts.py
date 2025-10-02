# File: report_test_values.py
# ---------------------------

import os
import pickle
import numpy as np
import argparse
import json
from collections import Counter

EXPERIMENT_DIR = "experiments/agentic_verl"

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--step_limit', type=int, default=50)
    args = parser.parse_args()
    return args

def get_train_files(args, parent_dir):
    steps_files = []

    folders = [f for f in os.listdir(parent_dir) if 'global_step' in f]
    folders = sorted(folders, key=lambda x: int(x.split("_")[-1]))

    for folder in folders:
        curr_step = int(folder.split("_")[-1])
        if curr_step > args.step_limit:
            break

        curr_path = os.path.join(parent_dir, folder, 'step_outputs.pkl')
        steps_files.append((curr_step, curr_path))

    return steps_files

def main():
    args = get_config() 
    parent_dir = os.path.join(EXPERIMENT_DIR, args.experiment_name)
    steps_files = get_train_files(args, parent_dir)
    
    train_steps = []
    num_searches = []
    for train_step, train_file in steps_files:
        with open(train_file, 'rb') as f:
            step_outputs = pickle.load(f)
        average_searches = np.mean(step_outputs['num_searches'])
        train_steps.append(train_step)
        num_searches.append(average_searches)

    print(args.experiment_name)
    print(f"X Axis: ", train_steps)
    print(f"Y Axis: ", num_searches)

if __name__ == "__main__":
    main()
