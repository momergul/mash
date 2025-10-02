# File: report_test_values.py
# ---------------------------

import os
import pickle
import numpy as np
import argparse
import json
from collections import Counter

EXPERIMENT_DIR = "experiments/abstention_experiments"
ACC_ESTIMATES_DIR = "data/model_accuracy_estimates/preds"

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--output_subfolder', type=str, default="")
    parser.add_argument('--answerability_threshold', type=str, default=0.1)
    parser.add_argument('--model_name', type=str, default="qwen")
    parser.add_argument('--dataset', type=str, default="")
    args = parser.parse_args()
    return args

def abstention_metrics(args, dataset, model_question_to_outputs, answerability_threshold):
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

    for question, output_dict in model_question_to_outputs.items():
        average_acc = question_to_dp[question][args.model_name]['sampling_knowns'] / 10
        for abstained in output_dict["abstained"]:
            buckets[average_acc]["total"] += 1
            buckets[average_acc]["hit"] += 1 if abstained else 0

    parametric_proportion = [round(100 * buckets[acc]["hit"] / buckets[acc]["total"], 2) for acc in accs]
    intercept = parametric_proportion[0]
    slope = parametric_proportion[0] - parametric_proportion[-1]

    return parametric_proportion, [intercept, slope]

def report_accuracy_metrics(model_question_to_outputs):
    neural_accs = []
    neural_precision = []

    seen_lens = set()
    for question, output_dict in model_question_to_outputs.items():
        for curr_acc, abstained in zip(output_dict["neural_acc"], output_dict["abstained"]):
            neural_accs.append(curr_acc)
            if not abstained:
                neural_precision.append(curr_acc)

    neural_accs = round(100 * np.mean(neural_accs), 2)
    neural_precision = round(100*np.mean(neural_precision), 2) if len(neural_precision) != 0 else 0
    sqa_f1 = round(2 * neural_accs * neural_precision / (neural_accs + neural_precision), 2)
    return [neural_accs, neural_precision, sqa_f1]

def report_test_results(args, filename):
    # Load the output file
    with open(filename, 'rb') as f:
        model_question_to_outputs = pickle.load(f)
    print(f"Reporting results for: {filename} with dataset {args.dataset}")

    acc_metrics = report_accuracy_metrics(model_question_to_outputs)
    print(acc_metrics)
    abstention_metrics = abstention_metrics(args, args.dataset, model_question_to_outputs, args.answerability_threshold)
    print(abstention_metrics)

    formatted_outputs_acc = ""
    for val in acc_metrics[:2]:
        formatted_outputs_acc += f"& {val:.2f} "
    print(formatted_outputs_acc)

    formatted_outputs_abs = ""
    for val in abstention_metrics[-1]:
        formatted_outputs_abs += f"& {val:.2f} "
    print(formatted_outputs_abs)

def main():
    args = get_config()
    if args.output_subfolder == "":
        filename = os.path.join(EXPERIMENT_DIR, args.experiment_name, f'{args.dataset}_outputs.pkl')
    else:
        filename = os.path.join(EXPERIMENT_DIR, args.experiment_name, args.output_subfolder, f'{args.dataset}_outputs.pkl')
    report_test_results(args, filename)

if __name__ == "__main__":
    main()
