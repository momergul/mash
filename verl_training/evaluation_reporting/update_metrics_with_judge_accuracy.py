# File: update_metrics_with_judge_accuracy.py
# -------------------------------------------

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from openai import OpenAI

import os
import pickle
import numpy as np
import argparse
from pathlib import Path
import re
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

from verl_training.rule_based_rewards.reward_functions import extract_answer
from verl_training.abstention_scripts.accuracy_estimation.evaluate_model_responses import sample_judgement_outputs, \
    judgement_prompt, normalize_answer, extract_float

EXPERIMENT_DIR = "experiments/agentic_verl"

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--mode', type=str, default="test", choices=["val", "test"])
    parser.add_argument('--exact_step', type=int, default=-1)
    parser.add_argument('--test_metric', type=str, default="tool_productivity")
    parser.add_argument('--num_threads', type=int, default=128)
    return parser.parse_args()

def get_client():
    API_KEY = os.getenv("API_KEY")
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client

def get_files(args, parent_dir):
    files = []
    if args.mode == "val":
        steps = [i*25 for i in range(1, 17)]
        for step in steps:
            filepath = os.path.join(parent_dir, f'global_step_{step}', 'step_val_outputs.pkl')
            files.append(filepath)
    else:
        if args.exact_step == 0:
            folders = [os.path.join(parent_dir, 'initial_checkpoint_evals')]
        elif args.exact_step != -1:
            folders = [os.path.join(parent_dir, f'global_step_{args.exact_step}')]
        else:
            folders = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
            folders = [f for f in folders if os.path.isdir(f)]
        for folder in folders:
            folder_contents = [os.path.join(folder, f) for f in os.listdir(folder) if "test" in f and args.test_metric in f]
            files += folder_contents

    return files

def compute_neural_metrics(step_outputs, dataset_indices):
    neural_accs = []
    tps = []
    parametric_accs = []
    parametric_precision = []

    for i in dataset_indices:
        neural_acc = step_outputs["neural_accs"][i]
        num_searches = step_outputs["num_searches"][i]
        tp = neural_acc / (1 + num_searches)

        neural_accs.append(neural_acc)
        tps.append(tp)
        parametric_accs.append(neural_acc if num_searches == 0 else 0)
        if num_searches == 0:
            parametric_precision.append(neural_acc)

    neural_acc = np.mean(neural_accs)
    tp = np.mean(tps)
    parametric_acc = np.mean(parametric_accs)
    parametric_prec = np.mean(parametric_precision) if len(parametric_precision) != 0 else -1

    metrics = {
        'neural_acc' : neural_acc,
        'neural_tool_productivity' : tp,
        'parametric_acc' : parametric_acc,
        'parametric_precision' : parametric_prec,
    }
    return metrics

def construct_judgement_inputs(all_outputs):
    questions = []
    answers = []
    extracted_responses = []
    used_indices = {}

    for i in range(len(all_outputs['reward'])):
        # First get the question and response
        question = all_outputs['question'][0][i].strip()
        full_response = all_outputs['responses'][0][i]
        model_answer = extract_answer(full_response) 
        if model_answer is None:
            continue
        ground_truths = [answer.strip() for answer in all_outputs['golden_answers'][0][i]]

        for answer in ground_truths:
            questions.append(question)
            answers.append(answer)
            extracted_responses.append(model_answer)
        used_indices[i] = len(ground_truths)

    inputs = []
    for question, answer, response in zip(questions, answers, extracted_responses):
        inputs.append(judgement_prompt(question, answer, response))
    return inputs, used_indices

def aggregate_judgements(all_outputs, judgement_outputs, used_indices):
    neural_judgements = []
    api_rejection = []

    judgement_start_idx = 0
    for i in range(len(all_outputs['reward'])):
        if i not in used_indices:
            neural_judgements.append(0)
            api_rejection.append(False)
            continue

        num_answers = used_indices[i]
        relevant_outputs = judgement_outputs[judgement_start_idx:judgement_start_idx+num_answers]

        best_val = -1
        best_output = None
        for output in relevant_outputs:
            value = extract_float(output)
            if value is None:
                value = 0
            if value > best_val:
                best_val = value
                best_output = output

        neural_judgements.append(best_val)
        api_rejection.append(best_output == "0 due to error")
        judgement_start_idx += num_answers

    return neural_judgements, api_rejection

def update_metrics(args, client, metric_path):
    # Load the metrics
    with open(metric_path, 'rb') as f:
        metrics, step_outputs = pickle.load(f)
    if "api_rejection" in step_outputs:
        return
    print("Evaluating: ", metric_path)

    # First assign neural accuracies to each
    inputs, used_indices = construct_judgement_inputs(step_outputs)
    judgement_outputs = sample_judgement_outputs(args, inputs, client)
    neural_judgements, api_rejection = aggregate_judgements(step_outputs, judgement_outputs, used_indices)
    judgements = []
    for i, neural_judgement in enumerate(neural_judgements):
        exact_match = step_outputs['em_scores'][i]
        judgement = 1.0 if exact_match == 1 or neural_judgement > 0.7 else 0.0
        judgements.append(judgement)
    step_outputs['neural_accs'] = judgements
    step_outputs['api_rejection'] = api_rejection

    # Now get the metrics
    datasets = set(step_outputs["data_sources"])
    for dataset in datasets:
        dataset_indices = [i for i in range(len(step_outputs['data_sources'])) if step_outputs['data_sources'][i] == dataset]
        dataset_metrics = compute_neural_metrics(step_outputs, dataset_indices)
        for key, value in dataset_metrics.items():
            metrics[f"my_val_metrics_{dataset}/{key}"] = value

    # Resave
    with open(metric_path, 'wb') as f:
        pickle.dump([metrics, step_outputs], f)    

def main():
    args = get_config()
    client = get_client()
    parent_dir = os.path.join(EXPERIMENT_DIR, args.experiment_name)
    files = get_files(args, parent_dir)

    for filepath in files:
        update_metrics(args, client, filepath)
    


if __name__ == "__main__":
    main()
