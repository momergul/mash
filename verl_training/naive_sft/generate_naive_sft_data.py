# File: generate_naive_sft_data.py
# --------------------------------
# Script to perform what was normally done inside verl with a minimal and faster python script instead.

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from transformers import AutoTokenizer
import re
import random
import requests
import gc
import torch
import argparse

import os
import pickle
import json
import pandas as pd


DATA_DIR = 'data/shortform_qa'

def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='larger_qwen')
    parser.add_argument('--maximum_length', type=int, default=6144)
    parser.add_argument('--action_response_length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)

    parser.add_argument('--target_dataset', type=str, default='nq')
    parser.add_argument('--prompt_type', type=str, default='baseline')
    parser.add_argument('--oracle_response', action='store_true')

    parser.add_argument('--url', type=str, default='http://127.0.0.1:8000/retrieve')
    parser.add_argument('--max_actions', type=int, default=2)
    parser.add_argument('--sample_n', type=int, default=4)
    parser.add_argument('--independent_sample_n', type=int, default=5)    

    parser.add_argument('--save_folder', type=str, default='data/naive_sft')
    return parser.parse_args()

## MODEL AND DATA
def get_dataset(args):
    file_suffix = 'parametric' if args.prompt_type == 'r1' else 'search' 
    dataset_path = os.path.join(DATA_DIR, f'{args.target_dataset}_naive_sft_{file_suffix}_train.parquet')
    df = pd.read_parquet(dataset_path)

    dataset = []
    for i in range(len(df)):
        dataset.append({
            'question' : df.iloc[i]['question'],
            'ground_truths' : df.iloc[i]['reward_model']['ground_truth']['target']
        })

    return dataset

def get_hf_name(args):
    if "larger_qwen" in args.model_name:
        path = "Qwen/Qwen2.5-32B"
        if 'instruct' in args.model_name:
            path += '-Instruct'
    elif 'large_qwen' in args.model_name:
        path = "Qwen/Qwen2.5-7B"
        if 'instruct' in args.model_name:
            path += '-Instruct'
    else:
        path = "Qwen/Qwen2.5-3B"
        if 'instruct' in args.model_name:
            path += '-Instruct'
    return path

def get_model(args):
    hf_name = get_hf_name(args)
    model = LLM(
        model=hf_name, tokenizer=hf_name, dtype='bfloat16',
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.maximum_length,
        enable_prefix_caching=True, tensor_parallel_size=args.tensor_parallel_size
    )

    stop_tokens = ["</think>", "</answer>", "</help>"] if args.oracle_response else ["</think>", "</answer>", "</search>"]
    sampling_params = SamplingParams(
        n=args.sample_n, temperature=args.temperature,
        top_p=args.top_p, max_tokens=args.action_response_length, 
        stop=stop_tokens,
        include_stop_str_in_output=True, logprobs=0
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    return model, tokenizer, sampling_params
    

## INFERENCE ##
def get_prompt(question, prompt_type):
    if prompt_type == 'baseline':
        prompt = "Answer the given question. You must conduct reasoning between <think> and </think> " +\
            "every time you get new information. After reasoning, if you find you lack some knowledge, you can call " +\
            "a search engine by <search> query </search> and it will return the top searched results between <document> and </document>. " +\
            "You need to make every search call count and gain helpful results. If you find no further external knowledge is needed, " +\
            "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, " +\
            f"<answer> Beijing </answer>. Question: {question}\n"
    elif prompt_type == "oracle_baseline":
        prompt = "Answer the given question. You must conduct reasoning between <think> and </think> " +\
            "every time you get new information. After reasoning, if you find you lack some knowledge, you can ask " +\
            "for help by <help> I need help </help> and it will return the answer to the original question between <helper_answer> and </helper_answer>. " +\
            "You need to ask for help only when necessary. If you find no further external knowledge is needed, " +\
            "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, " +\
            f"<answer> Beijing </answer>. Question: {question}\n"
    elif prompt_type == "decomposition":
        prompt = "Answer the given question. You must conduct reasoning between <think> and </think> " +\
            "every time you get new information. After reasoning, if you find you lack some knowledge, you can ask a question to " +\
            "a search engine by <search> query </search> and it will return the top searched results between <document> and </document>. " +\
            "A search query should be an atomic question asking about one, single piece of information.\n\n" +\
            "Example 1:\n Question: 'Who was born first, Clint Eastwood or Harrison Ford?'\n" +\
            "Valid Queries:'<search>Clint Eastwood birth date</search>' and " +\
            "'<search>Harrison Ford birth date</search>'.\nThe query '<search>Clint Eastwood and Harrison Ford birth date</search>' " +\
            "is invalid. The query '<search>Clint Eastwood birth date\n Harrison Ford birth date</search>' is also invalid. " +\
            "Do not pack in multiple questions into one query. Each query should be completely independent.\n\n" +\
            "Example 2:\n Question: 'Which is a genus of palms, Zinnia or Butia?'\n" +\
            "Valid Queries: '<search>Zinnia genus classification</search>' " +\
            "and <search>Butia genus classification</search>.\n\n" +\
            "Example 3:\n Question: 'When did the country where Piltene is located become part of the USSR?'\n" +\
            "Initial Query: '<search>Piltene location</search>'\n\n" +\
            "In each of these examples, you should conduct a search only if you lack the relevant information. " +\
            "Remember, you should decompose questions in your search queries and conduct searches for each atomic question separately. " +\
            "You need to make every search call count and gain helpful results. If you find no further external knowledge is needed, " +\
            "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, " +\
            f"<answer> Beijing </answer>.\nQuestion: {question}\n"
    elif prompt_type == "r1":
        prompt = f"Answer the given question. You should first have a reasoning process in mind and then provides the answer. " +\
            "Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example " +\
            f"<answer> Beijing </answer>. Question: {question}\n"

    return [{'role' : 'user', 'content' : prompt}]

def get_raw_prompt_ids(questions, prompt_type, tokenizer):
    raw_prompt_ids = []
    for i, question in enumerate(questions):
        prompt = get_prompt(question, prompt_type)
        templated_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        raw_prompt_ids.append(tokenizer.encode(templated_prompt, add_special_tokens=False))
    return raw_prompt_ids

def get_properly_formatted_outputs_for_naive_sft(args, output, sample_n, action):
    # First get properly formatted outputs
    if args.oracle_response:
        action_end = "</think>" if action == "<think>" else ("</help>" if action == "<help>" else "</answer>")
        start_tags = ["<think>", "<help>", "<answer>"]
        end_tags = ["</think>", "</help>", "</answer>"]
    else:
        action_end = "</think>" if action == "<think>" else ("</search>" if action == "<search>" else "</answer>")
        start_tags = ["<think>", "<search>", "<answer>"]
        end_tags = ["</think>", "</search>", "</answer>"]

    if args.oracle_response and action == "<help>":
        return ["I need help </help>"]

    proper_output_texts = []
    for i in range(sample_n):
        output_text = output.outputs[i].text
        valid_output = action_end in output_text
        for tag in start_tags+end_tags:
            if tag == action_end:
                continue
            if tag in output_text:
                valid_output = False

        if valid_output:
            proper_output_texts.append(output_text)
    if len(proper_output_texts) == 0:
        proper_output_texts = [output.outputs[i].text+action_end for i in range(sample_n)]
    return proper_output_texts

def batched_oracle_answer(args, search_indices, ground_truths):
    search_results = []
    for idx in search_indices:
        curr_answer = ground_truths[idx][0]
        doc = f"\n<helper_answer>The answer to the original question is: {curr_answer}</helper_answer>\n"
        search_results.append(doc)
    return search_results

def batched_search(args, search_indices, search_queries, entities):
    search_results = []
    unprocessed_results = server_search(args, search_queries)
    for result in unprocessed_results:
        doc = "\n"
        for doc_item in result:
            content = doc_item['document']['contents'].split("\n")
            title = content[0]
            text = "\n".join(content[1:])                    
            doc += f"<document>\n(Title: {title}) {text}\n</document>\n"
        search_results.append(doc)
    return search_results

def server_search(args, query_list):
    payload = {
        "queries" : query_list,
        "topk" : 3,
        "return_scores": True
    }

    results = requests.post(args.url, json=payload).json()
    return results['result']

def construct_partial_inputs(prompt_ids, partial_completion, tokenizer):
    if len(partial_completion) == 0:
        return tokenizer.decode(prompt_ids)
    else:
        full_text = ""
        for tokens in partial_completion:
            full_text += tokenizer.decode(tokens)
        return full_text

def perform_inference(args, input_dataset, model, tokenizer, sampling_params):
    max_actions = args.max_actions
    sample_n = args.sample_n
    independent_sample_n = args.independent_sample_n

    questions = []
    ground_truths = []
    for dp in input_dataset:
        for repetition in range(independent_sample_n):
            questions.append(dp['question'])
            ground_truths.append(dp['ground_truths'])
    
    # Processing the data
    raw_prompt_ids = get_raw_prompt_ids(questions, args.prompt_type, tokenizer) 
    num_outputs = len(raw_prompt_ids)

    # Data structures
    idx_to_samples = {}
    for i in range(num_outputs // independent_sample_n):
        if max_actions == 0:
            target_searches = 0
            action_sequence = ["<think>", "<answer>"]
        elif args.oracle_response:
            target_searches = random.randint(1, max_actions)
            action_sequence = ["<think>", "<help>"] * target_searches + ["<think>", "<answer>"]
        else:
            target_searches = random.randint(1, max_actions)
            action_sequence = ["<think>", "<search>"] * target_searches + ["<think>", "<answer>"]

        for j in range(independent_sample_n):
            idx = i*independent_sample_n + j
            idx_dict = {
                'num_searches' : target_searches,
                'action_sequence' : action_sequence,
                'action_idx' : 0,
                'partial_tokens' : [raw_prompt_ids[idx]],
            }
            idx_to_samples[idx] = idx_dict

    # Start sampling
    search_pattern = r'(.*?)</search>' if not args.oracle_response else r'(.*?)</help>'
    for i in range(2 * (max_actions+1)):
        tp_rank = vllm_ps.get_tensor_model_parallel_rank() if args.tensor_parallel_size != 1 else 0

        if tp_rank == 0:
            # Prepare inputs with action and do inference
            candidate_indices = [idx for idx, idx_dict in idx_to_samples.items() if idx_dict['action_idx'] < len(idx_dict['action_sequence'])]
            curr_indices = []
            inputs = []
            for curr_idx in candidate_indices:
                curr_dict = idx_to_samples[curr_idx]
                curr_action_start = "\n" + curr_dict['action_sequence'][curr_dict['action_idx']] + "\n"
                curr_action_tokens = list(tokenizer(curr_action_start, add_special_tokens=False)['input_ids'])
                partial_tokens = curr_dict['partial_tokens']
                partial_tokens.append(curr_action_tokens)
            
                total_length = sum([len(curr_partial_tokens) for curr_partial_tokens in partial_tokens])
                if total_length >= args.maximum_length:
                    curr_dict['action_idx'] = len(curr_dict['action_sequence'])
                    continue
                curr_indices.append(curr_idx)
                inputs.append(construct_partial_inputs(raw_prompt_ids[curr_idx], partial_tokens, tokenizer))
            broadcast_data = {'curr_indices' : curr_indices, 'inputs' : inputs, 'idx_to_samples' : idx_to_samples}
        else:
            broadcast_data = None
        broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0) if args.tensor_parallel_size != 1 else broadcast_data
        curr_indices, inputs, idx_to_samples = broadcast_data['curr_indices'], broadcast_data['inputs'], broadcast_data['idx_to_samples']
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=True)

        # First pass: Pick which output to use and process
        search_indices = []
        search_queries = []
        for curr_idx, output in zip(curr_indices, outputs):
            # Choose the output
            curr_dict = idx_to_samples[curr_idx]
            curr_action = curr_dict['action_sequence'][curr_dict['action_idx']]
            properly_formatted_outputs = get_properly_formatted_outputs_for_naive_sft(args, output, sample_n, curr_action)

            # Then, process
            if curr_action == "<answer>":
                curr_dict['answer_set'] = properly_formatted_outputs
            else:
                curr_output_text = random.sample(properly_formatted_outputs, 1)[0]
                curr_output_tokens = list(tokenizer(curr_output_text, add_special_tokens=False)['input_ids'])
                curr_dict['partial_tokens'].append(curr_output_tokens)
                search_matches = re.findall(search_pattern, curr_output_text, re.DOTALL)
                if len(search_matches) != 0:
                    search_indices.append(curr_idx)
                    search_queries.append(search_matches[0])
            curr_dict['action_idx'] += 1

        # Second pass: Incorporate search results
        if tp_rank == 0:
            if args.oracle_response:
                search_results = batched_oracle_answer(args, search_indices, ground_truths)
            else:
                search_results = batched_search(args, search_indices, search_queries, None)
            broadcast_data = {
                'search_results': search_results,
            }
        else:
            broadcast_data = None
        search_results = vllm_ps._TP.broadcast_object(broadcast_data, src=0)['search_results'] if args.tensor_parallel_size != 1 else search_results # broadcast tool call results across tp
        for curr_idx, search_result in zip(search_indices, search_results):
            doc_tokens = list(tokenizer(search_result, add_special_tokens=False)['input_ids'])
            idx_to_samples[curr_idx]['partial_tokens'].append(doc_tokens)

    # Done sampling: Now to decode outputs and return
    for i in range(len(idx_to_samples)):
        partial_tokens = idx_to_samples[i]['partial_tokens'][1:]
        curr_tokens = []
        for tokens in partial_tokens:
            curr_tokens += tokens
        completion = tokenizer.decode(curr_tokens)
        idx_to_samples[i]['completion'] = completion

    non_tensor_batch = {
        'question' : [question for question in questions],
        'ground_truth' : [gt for gt in ground_truths],
        'completion' : [idx_to_samples[i]['completion'] for i in range(len(idx_to_samples))],
        'answer_set' : [idx_to_samples[i].get('answer_set', None) for i in range(len(idx_to_samples))]
    }

    return non_tensor_batch

def postprocess_responses(all_outputs):
    from collections import defaultdict
    question_to_response = defaultdict(list)

    for i in range(len(all_outputs['question'])):
        curr_question = all_outputs['question'][i]
        partial_completion = all_outputs['completion'][i]
        answer_set = all_outputs['answer_set'][i]
        ground_truths = all_outputs['ground_truth'][i]
        question_to_response[curr_question].append((partial_completion, answer_set, ground_truths))

    return question_to_response

def save_outputs(args, question_to_responses):
    parent_dir = args.save_folder
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    model_name = args.model_name

    dataset_suffix = args.prompt_type
    dataset_name = f'{args.target_dataset}_naive_sft_{dataset_suffix}'

    filename = os.path.join(parent_dir, f'{model_name}_{dataset_name}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(question_to_responses, f)

def main():
    args = get_configs()
    input_dataset = get_dataset(args)
    model, tokenizer, sampling_params = get_model(args)
    all_outputs = perform_inference(args, input_dataset, model, tokenizer, sampling_params)
    question_to_responses = postprocess_responses(all_outputs)
    save_outputs(args, question_to_responses)
    


if __name__ == "__main__":
    main()
