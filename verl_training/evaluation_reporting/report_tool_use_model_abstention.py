# File: report_test_values.py
# ---------------------------

import os
import pickle
import numpy as np
import argparse
import json
from collections import Counter

EXPERIMENT_DIR = "experiments/agentic_verl"
ACC_ESTIMATES_DIR = "data/model_accuracy_estimates/preds"

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--exact_step', type=int, default=-1)
    parser.add_argument('--metric_name', type=str, default="tool_productivity")
    parser.add_argument('--answerability_threshold', type=str, default=0.1)
    parser.add_argument('--model_name', type=str, default="qwen")
    args = parser.parse_args()
    return args

def abstention_metrics(args, dataset, step_outputs, answerability_threshold):
    # First load the accuracy estimates
    if dataset == "2wiki":
        dataset = "well_behaved_2wiki"
    acc_estimate_path = os.path.join(ACC_ESTIMATES_DIR, f'{dataset}_preds.jsonl')
    question_to_dp = {}
    with open(acc_estimate_path, 'r') as f:
        for line in f:
            dp = json.loads(line.strip())
            question = dp['question']
            question_to_dp[question] = dp

    # Next determine the stuff
    accs = [i/10 for i in range(11)]
    buckets = {acc : {"hit" : 0, "total" : 0} for acc in accs}
    num_outputs = len(step_outputs["em_scores"])
    for i in range(num_outputs):
        question = step_outputs["question"][0][i]
        abstained = step_outputs["num_searches"][i] > 0
        average_acc = question_to_dp[question][args.model_name]['sampling_knowns'] / 10

        buckets[average_acc]["total"] += 1
        buckets[average_acc]["hit"] += 1 if abstained else 0

    
    parametric_proportion = [round(100 * buckets[acc]["hit"] / buckets[acc]["total"], 2) for acc in accs]
    intercept = parametric_proportion[0]
    slope = parametric_proportion[0] - parametric_proportion[-1]

    return parametric_proportion, [intercept, slope]

def report_test_results(args, metric_path):
    # Load the metrics
    with open(metric_path, 'rb') as f:
        metrics, step_outputs = pickle.load(f)
    print(f"Reporting results for: {metric_path}")

    datasets = set(step_outputs["data_sources"])
    metrics_in_sequence = ["parametric_acc", "parametric_precision"]
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
        abstention_results = abstention_metrics(args, dataset, step_outputs, args.answerability_threshold)
        print(abstention_results)
        
        formatted_outputs = ""
        for val in metric_values:
            formatted_outputs += f"& {val:.2f} "
        print(formatted_outputs)

        formatted_outputs = ""
        for val in abstention_results[-1]:
            formatted_outputs += f"& {val:.2f} "
        print(formatted_outputs)
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
        report_test_results(args, test_file)

if __name__ == "__main__":
    main()
