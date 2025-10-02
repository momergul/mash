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
    metrics_in_sequence = ["neural_acc", "num_searches", "neural_tool_productivity"]
    for dataset in datasets:
        print(dataset)

        metric_values = []
        for metric in metrics_in_sequence:
            found_value = None
            for key, value in metrics.items():
                if dataset in key and key.split("/")[-1] == metric:
                    if metric == "num_searches":
                        found_value = round(value, 2)
                    else:
                        found_value = round(value*100, 2)                        
                    break
            metric_values.append('-' if found_value is None else found_value)
        print(metric_values)

        formatted_text = ""
        for val in metric_values:
            formatted_text += f"& {val:.2f} "
        print(formatted_text)
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
