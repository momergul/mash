# File: evaluate_responses.py
# ---------------------------

import argparse
import os
import json
import pickle
from vllm import LLM, SamplingParams
import torch
import re
import string
import math
from time import time
import concurrent.futures
from openai import OpenAI

from verl_training.abstention_scripts.accuracy_estimation.evaluate_model_responses import sample_extraction_outputs, \
    sample_judgement_outputs, extraction_prompt, judgement_prompt, normalize_answer, extract_float

PARENT_DIR = 'data/model_accuracy_estimates'
SAVE_DIR = 'experiments/abstention_experiments'

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen')
    parser.add_argument('--test_dataset_name', type=str, default='nq')
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--output_subfolder', type=str, default="")
    parser.add_argument('--num_threads', type=int, default=128)
    return parser.parse_args()

def get_client():
    API_KEY = os.getenv("API_KEY")
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client

def get_eval_model(args):
    hf_name = 'Qwen/Qwen2.5-32B-Instruct'
    model = LLM(model=hf_name, dtype=torch.bfloat16, gpu_memory_utilization=0.9,
                max_model_len=2048, enable_prefix_caching=True, tensor_parallel_size=args.tensor_parallel_size)
    return model
    
def load_dataset(args):
    data_path = os.path.join(PARENT_DIR, 'preds', f'{args.test_dataset_name}_preds.jsonl')
    question_to_dp = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dp = json.loads(line.strip())
            if dp["split"] != "test":
                continue
            question = dp["question"]
            question_to_dp[question] = dp

    return question_to_dp

def load_model_outputs(args):
    if args.output_subfolder == "":
        savepath = os.path.join(SAVE_DIR, args.experiment_name, f'{args.test_dataset_name}_outputs.pkl')
    else:
        savepath = os.path.join(SAVE_DIR, args.experiment_name, args.output_subfolder,
                                f'{args.test_dataset_name}_outputs.pkl')
    with open(savepath, 'rb') as f:
        model_outputs = pickle.load(f)

    question = list(model_outputs.keys())[0]
    should_evaluate = "extracted_text" not in model_outputs[question]
    return model_outputs, should_evaluate

def construct_extraction_inputs(args, question_order, model_question_to_outputs):
    questions = []
    responses = []
    for question in question_order:
        output_dict = model_question_to_outputs[question]
        for response in output_dict['outputs']:
            questions.append(question.strip())
            responses.append(response.strip())

    inputs = []
    for question, response in zip(questions, responses):
        inputs.append(extraction_prompt(question, response)) 

    return inputs

def construct_judgement_inputs(args, question_order, outputs_per_q, question_to_dp, extraction_outputs):
    questions = []
    answers = []
    extracted_responses = []

    for i, question in enumerate(question_order):
        gold_answers = [answer.strip() for answer in question_to_dp[question]['answers']]
        for j in range(outputs_per_q):
            response = extraction_outputs[i*outputs_per_q+j].strip()
            for answer in gold_answers:
                questions.append(question)
                answers.append(answer)
                extracted_responses.append(response)

    inputs = []
    for question, answer, response in zip(questions, answers, extracted_responses):
        inputs.append(judgement_prompt(question, answer, response))

    return inputs

def compute_exact_matches(args, question_order, outputs_per_q, question_to_dp, extraction_outputs):
    exact_matches = []
    for i, question in enumerate(question_order):
        datapoint = question_to_dp[question]
        gold_answers = [normalize_answer(answer) for answer in datapoint['answers']] 

        for j in range(outputs_per_q):
            output = extraction_outputs[i*outputs_per_q + j]
            response = normalize_answer(output)

            has_match = False
            for gold_answer in gold_answers:
                if response == gold_answer:
                    has_match = True
            exact_matches.append(has_match)
    return exact_matches

def collect_judgements_with_all_answers(args, question_order, outputs_per_q, question_to_dp,
                                        judgement_outputs, exact_matches):
    judgements = []
    selected_judgement_outputs = []
    judgement_start_idx = 0
    for i, exact_match in enumerate(exact_matches):
        # Get the judgements
        question_idx = i // outputs_per_q
        datapoint = question_to_dp[question_order[question_idx]]
        num_answers = len(datapoint['answers'])
        datapoint_judgements = [judgement_outputs[j] for j in range(judgement_start_idx, judgement_start_idx+num_answers)]
        judgement_start_idx += num_answers

        # Determine the highest judgement
        best_output = None
        best_output_val = -1
        for output in datapoint_judgements:
            value = extract_float(output)
            if value is None:
                value = 0

            if value > best_output_val:
                best_output_val = value
                best_output = output
        neural_judgement = best_output_val > 0.7

        judgements.append(exact_match or neural_judgement)
        selected_judgement_outputs.append(best_output)

    return selected_judgement_outputs, judgements

def process_and_predict(args, client, question_to_dp, model_question_to_outputs):
    # First extract the model responses
    question_order = list(model_question_to_outputs.keys())
    outputs_per_q = len(model_question_to_outputs[question_order[0]]['outputs'])
    
    inputs = construct_extraction_inputs(args, question_order, model_question_to_outputs)
    extraction_outputs = sample_extraction_outputs(args, inputs, client)

    # Then determine accuracies of the extracted responses
    inputs = construct_judgement_inputs(args, question_order, outputs_per_q, question_to_dp, extraction_outputs)
    judgement_outputs = sample_judgement_outputs(args, inputs, client) 
    exact_matches = compute_exact_matches(args, question_order, outputs_per_q, question_to_dp, extraction_outputs)

    selected_judgement_outputs, judgements = collect_judgements_with_all_answers(
        args, question_order, outputs_per_q, question_to_dp, judgement_outputs, exact_matches
    )

    return question_order, extraction_outputs, selected_judgement_outputs, judgements

def save_outputs(args, model_question_to_outputs, questions, extraction_outputs,
                 judgement_outputs, judgements):
    outputs_per_q = len(model_question_to_outputs[questions[0]]['outputs'])
    for i, question in enumerate(questions):
        output_dict = model_question_to_outputs[question]
        output_dict["extracted_text"] = []
        output_dict["judgement_text"] = []
        output_dict["neural_acc"] = []
        output_dict["abstained"] = []

        for j in range(outputs_per_q):
            curr_idx = i*outputs_per_q+j
            output_dict["extracted_text"].append(extraction_outputs[curr_idx])
            output_dict["neural_acc"].append(1.0 if judgements[curr_idx] else 0.0)
            output_dict["judgement_text"].append(judgement_outputs[curr_idx])
            output_dict["abstained"].append("no answer" in extraction_outputs[curr_idx])

    if args.output_subfolder == "":
        savepath = os.path.join(SAVE_DIR, args.experiment_name, f'{args.test_dataset_name}_outputs.pkl')
    else:
        savepath = os.path.join(SAVE_DIR, args.experiment_name, args.output_subfolder,
                                f'{args.test_dataset_name}_outputs.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(model_question_to_outputs, f)

def main():
    args = get_configs()
    client = get_client()

    question_to_dp = load_dataset(args)
    model_question_to_outputs, should_evaluate = load_model_outputs(args)
    if not should_evaluate:
        return

    questions, extraction_outputs, judgement_outputs, judgements = process_and_predict(
        args, client, question_to_dp, model_question_to_outputs
    )
    save_outputs(
        args, model_question_to_outputs, questions, extraction_outputs,
        judgement_outputs, judgements
    )

    

if __name__ == "__main__":
    main()
