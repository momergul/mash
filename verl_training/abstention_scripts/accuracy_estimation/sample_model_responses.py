# File: sample_model_responses.py
# -------------------------------

import argparse
import os
import json
from vllm import LLM, SamplingParams
import torch
import pickle

PARENT_DIR = 'data/model_accuracy_estimates'

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen')
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--few_shot_name', type=str, default="")
    
    parser.add_argument('--output_length', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--top_k', type=float, default=50)

    return parser.parse_args()

def get_hf_name(model_name):
    path = "Qwen/Qwen2.5-3B"        
    if 'instruct' in model_name:
        path += '-Instruct'
    return path

def load_model(args):
    # First load the vllm model
    hf_name = get_hf_name(args.model_name)
    model = LLM(model=hf_name, dtype=torch.bfloat16, gpu_memory_utilization=0.9,
                max_model_len=4096, enable_prefix_caching=True)

    # Define the sampling params
    sampling_params = SamplingParams(n=10, temperature=args.temperature, top_p=args.top_p,
                                     top_k=args.top_k, max_tokens=args.output_length)
    return model, sampling_params

def load_dataset(args):
    # Get the overall questions
    data_path = os.path.join(PARENT_DIR, 'preds', f'{args.dataset_name}_preds.jsonl')
    print("Loading: ", data_path)
    dataset_questions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset_questions.append(json.loads(line.strip()))

    # Get the few-shot prompt QA pairs
    few_shot_path = os.path.join(PARENT_DIR, 'few_shot_examples', f'{args.dataset_name}_few_shot.pkl')
    print("Loading: ", few_shot_path)
    with open(few_shot_path, 'rb') as f:
        qa_pairs = pickle.load(f)

    return dataset_questions, qa_pairs

def generate_base_model_prompt(question, few_shot_examples):
    few_shot_prompt = "You are a helpful assistant. Your task is to provide shortform answers to questions that you are given.\n\n"
    for i, (example_question, gt) in enumerate(few_shot_examples):
        few_shot_prompt += f"User: Answer the following question.\nQuestion: {example_question}\nAnswer: {gt}\n\n"
    few_shot_prompt += f"User: Answer the following question.\nQuestion: {question}\n"

    return few_shot_prompt    

def process_and_predict(args, model, sampling_params, dataset_questions, few_shot_examples):
    if "instruct" in args.model_name:
        pass
    else:
        inputs = [generate_base_model_prompt(datapoint['question'], few_shot_examples) for datapoint in dataset_questions]
        outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    return outputs

def isolate_answer(args, answer):
    # Do not make answer changes if instruct
    if "instruct" in args.model_name or "User:" not in answer:
        return answer.strip()

    user_idx = answer.find("User:")
    return answer[:user_idx].strip()

def save_outputs(args, dataset_questions, outputs):
    # Update the datapoints
    for datapoint, output in zip(dataset_questions, outputs):
        datapoint[args.model_name] = {}

        texts = [isolate_answer(args, output.outputs[i].text) for i in range(10)]
        datapoint[args.model_name]['sampling_pred_text'] = texts

    # Save/update the dataset
    data_path = os.path.join(PARENT_DIR, 'preds', f'{args.dataset_name}_preds.jsonl')
    with open(data_path, 'w', encoding='utf-8') as f:
        for datapoint in dataset_questions:
            f.write(json.dumps(datapoint) + '\n')

def main():
    # Setup
    args = get_configs() 
    model, sampling_params = load_model(args)
    dataset_questions, few_shot_examples = load_dataset(args)

    # Predict and save
    outputs = process_and_predict(args, model, sampling_params, dataset_questions, few_shot_examples)
    save_outputs(args, dataset_questions, outputs)

if __name__ == "__main__":
    main()
