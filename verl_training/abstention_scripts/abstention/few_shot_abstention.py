# File: sample_model_responses.py
# -------------------------------

import argparse
import os
import json
from vllm import LLM, SamplingParams
import torch
import pickle

PARENT_DIR = 'data/model_accuracy_estimates'
SAVE_DIR = 'experiments/abstention_experiments'

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--checkpoint_name', type=str, default="checkpoint-400")
    parser.add_argument('--model_name', type=str, default='qwen')
    parser.add_argument('--test_dataset_name', type=str, default='nq')
    parser.add_argument('--few_shot_dataset_name', type=str, default='nq')
    parser.add_argument('--num_prompts', type=int, default=4)
    
    parser.add_argument('--output_length', type=int, default=128)
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
    if args.experiment_name == "":
        # The few-shot setting
        return load_hf_model(args)
    else:
        # The instruction-tuning setting
        return load_local_model(args)

def load_hf_model(args):
    # First load the vllm model
    hf_name = get_hf_name(args.model_name)
    model = LLM(model=hf_name, dtype=torch.bfloat16, gpu_memory_utilization=0.9,
                max_model_len=4096, enable_prefix_caching=True)

    # Define the sampling params
    sampling_params = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                                     top_k=args.top_k, max_tokens=args.output_length)
    return model, sampling_params

def load_local_model(args):
    model_path = os.path.join(SAVE_DIR, args.experiment_name, args.checkpoint_name)
    model = LLM(model=model_path, dtype=torch.bfloat16, gpu_memory_utilization=0.9,
                max_model_len=1024, enable_prefix_caching=True)
    
    sampling_params = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.output_length)
    return model, sampling_params

def load_dataset(args):
    # Get the overall questions
    data_path = os.path.join(PARENT_DIR, 'preds', f'{args.test_dataset_name}_preds.jsonl')
    print("Loading: ", data_path)
    dataset_questions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dp = json.loads(line.strip())
            if dp["split"] != "test":
                continue
            dataset_questions.append(dp)

    # Get the few-shot question-answer pairs
    qa_pairs = []
    if args.experiment_name != "":
        return dataset_questions, qa_pairs
    for i in range(args.num_prompts):
        few_shot_path = os.path.join(PARENT_DIR, 'few_shot_abstention', f'{args.few_shot_dataset_name}_{args.model_name}_few_shot_{i}.pkl')
        print("Loading: ", few_shot_path)
        with open(few_shot_path, 'rb') as f:
            curr_pairs = pickle.load(f)
        qa_pairs.append(curr_pairs)

    return dataset_questions, qa_pairs

def generate_instruct_model_prompt(question):
    prompt = "Answer the given question. If you are not confident that your answer will be correct, " +\
        "you should abstain from answering by using the phrase 'I am afraid I cannot help you as I do " +\
        f"not know the answer to this question.' Question: {question}"

    conversation = [
        {
            'role' : 'user',
            'content' : prompt
        }
    ]
    return conversation

def generate_base_model_prompt(question, few_shot_examples):
    few_shot_prompt = "You are a helpful assistant. Your task is to provide shortform answers to questions that you are given. " +\
        "If you are not confident that your answer will be correct, you should abstain from answering by using the phrase 'I am afraid I " +\
        "cannot help you as I do not know the answer to this question.'\n\n"
    for i, (example_question, gt) in enumerate(few_shot_examples):
        few_shot_prompt += f"User: Answer the following question.\nQuestion: {example_question}\nAnswer: {gt}\n\n"
    few_shot_prompt += f"User: Answer the following question.\nQuestion: {question}\n"
    return few_shot_prompt    

def process_and_predict(args, model, sampling_params, dataset_questions, all_few_shot_examples):
    if args.experiment_name != "":
        inputs = []
        for datapoint in dataset_questions:
            for i in range(args.num_prompts):
                inputs.append(generate_instruct_model_prompt(datapoint['question']))
        outputs = model.chat(inputs, sampling_params=sampling_params, use_tqdm=True)
    else:
        inputs = []
        for datapoint in dataset_questions:
            for few_shot_examples in all_few_shot_examples:
                inputs.append(generate_base_model_prompt(datapoint['question'], few_shot_examples))
        outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    return outputs

def isolate_answer(args, answer):
    # Do not make answer changes if instruct
    if args.experiment_name != "":
        targets = ["user\n", "assistant\n", "User\n", "Assistant\n"]
        earliest_pos = len(answer)
        for target in targets:
            pos = answer.find(target)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        return answer[:earliest_pos].strip()
    else:
        if "User:" not in answer:
            return answer.strip()
        user_idx = answer.find("User:")
        return answer[:user_idx].strip()

def save_outputs(args, dataset_questions, outputs):
    # Create the folder
    if args.experiment_name == "":
        save_folder = os.path.join(SAVE_DIR, f'{args.few_shot_dataset_name}_{args.model_name}_few_shot')
    else:
        save_folder = os.path.join(SAVE_DIR, args.experiment_name, "outputs")
    os.makedirs(save_folder, exist_ok=True)
    savepath = os.path.join(save_folder, f'{args.test_dataset_name}_outputs.pkl')

    # Save the datapoints
    question_to_responses = {}
    num_prompts = args.num_prompts
    for i, datapoint in enumerate(dataset_questions):
        question = datapoint['question']
        curr_outputs = outputs[i*num_prompts:(i+1)*num_prompts]
        texts = [isolate_answer(args, output.outputs[0].text) for output in curr_outputs]
        question_to_responses[question] = {'outputs' : texts}

    # Save/update the dataset
    print("Saving to: ", savepath)
    with open(savepath, 'wb') as f:
        pickle.dump(question_to_responses, f)

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
