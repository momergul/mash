# File: report_test_values.py
# ---------------------------

import os
import pickle
import numpy as np
import argparse

from collections import Counter

EXPERIMENT_DIR = "experiments/factscore_verl"

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--exact_step', type=int, default=-1)
    parser.add_argument('--metric_name', type=str, default='tool_productivity')
    args = parser.parse_args()
    return args

def report_test_results(metric_path):
    # Load the metrics
    with open(metric_path, 'rb') as f:
        metrics, step_outputs = pickle.load(f)
    print(f"Reporting results for: {metric_path}")

    datasets = set(step_outputs["data_sources"])
    for dataset in datasets:
        print(dataset)
        search_stats = {i : {'pct' : 0, 'total' : 0, 'correct' : 0} for i in range(3)}

        num_outputs = len(step_outputs["em_scores"])
        for i in range(num_outputs):
            num_searches = int(step_outputs["num_searches"][i])
            search_category = min(num_searches, 2)

            search_stats[search_category]['pct'] += 100 / num_outputs
            search_stats[search_category]['total'] += 1
            search_stats[search_category]['correct'] += step_outputs['neural_accs'][i]

        formatted_output = ""
        for i in range(3):
            percentage = search_stats[i]['pct']
            acc = 100 * search_stats[i]['correct'] / search_stats[i]['total'] if search_stats[i]['total'] != 0 else 0
            formatted_output += f"& ${percentage:.1f}_{{{acc:.1f}}}$ " 
        print(formatted_output)
        print()
    print("-----\n")

def get_test_files(args, parent_dir):
    test_files = []
    if args.exact_step == 0:
        folders = [os.path.join(parent_dir, 'initial_checkpoint_evals')]
    elif args.exact_step != -1:
        folders = [os.path.join(parent_dir, f'global_step_{args.exact_step}')]
    else:
        folders = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
        folders = [f for f in folders if os.path.isdir(f)]

    for folder in folders:
        folder_contents = [os.path.join(folder, f) for f in os.listdir(folder) if 'test' in f and args.metric_name in f]
        test_files += folder_contents

    return test_files

def main():
    args = get_config() 
    parent_dir = os.path.join(EXPERIMENT_DIR, args.experiment_name)
    test_files = get_test_files(args, parent_dir)
    
    for test_file in test_files:
        report_test_results(test_file)

if __name__ == "__main__":
    main()
