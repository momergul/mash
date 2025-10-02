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

PARENT_DIR = 'data/model_accuracy_estimates'

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen')
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--num_splits', type=int, default=128)
    parser.add_argument('--num_threads', type=int, default=64)
    return parser.parse_args()

def extract_float(s):
    match = re.match(r'^[-+]?(\d+\.?\d*|\.\d+)', s)
    return float(match.group()) if match else None

def get_client():
    API_KEY = os.getenv("API_KEY")
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client
    
def load_dataset(args, split_num):
    data_path = os.path.join(PARENT_DIR, 'preds', f'{args.dataset_name}_preds_split_{split_num}.jsonl')

    dataset_questions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset_questions.append(json.loads(line.strip()))

    return dataset_questions
    
def extraction_prompt(question, response):
    conversation = [
        {
            'role' : 'system', 
             'content' : 'You are a helpful assistant. You will be given a question and a response to the question, which ' + 'does not need to be correct. If the response does not contain an answer to the question, output "no answer"; ' + 'otherwise, extract the answer from the text.'
        },
        {
            'role' : 'user',
            'content' : "Question: What was the last US state to reintroduce alcohol after prohibition?\nText: The last US state to reintroduce alcohol after prohibition was Mississippi. Mississippi legalized alcohol on August 17, 1933, making it the last state to do so. Prohibition had been enacted nationwide through the 18th Amendment to the US Constitution in 1919, but it was repealed in 1933 with the passage of the 21st Amendment."
        },
        {
            'role' : 'assistant',
            'content' : "Mississippi"
        },
        {
            'role' : 'user',
            'content' : "Question: What was the name of Michael Jackson's autobiography written in 1988?\nText: Michael Jackson did not write an autobiography in 1988."
        },
        {
            'role' : 'assistant',
            'content' : 'no answer'
        },
        {
            'role' : 'user',
            'content' : "Question: Who was the director of the CIA from 1976-81?\nText: Stansfield Turner."
        },
        {
            'role' : 'assistant',
            'content' : 'Stansfield Turner'
        },
        {
            'role' : 'user',
            'content' : "Question: Which 90s sci fi series with James Belushi was based on Bruce Wagner's comic strip of the same name?\nText: The 90s sci-fi series based on Bruce Wagner's comic strip was \"The Adventures of Buckaroo Banzai Across the 8th Dimension.\" It starred James Belushi as the titular character and aired in 1984."
        },
        {
            'role' : 'assistant',
            'content' : "The Adventures of Buckaroo Banzai Across the 8th Dimension."
        },
        {
            'role' : 'user',
            'content' : "Question: In what years did John Henry win the Arlington Million?\nText: I'm sorry, but I don't have information about John Henry winning the Arlington Million. John Henry was a thoroughbred race horse, but I don't have details about his victories in specific races, including the Arlington Million."
        },
        {
            'role' : 'assistant',
            'content' : 'no answer'
        },
        {
            'role' : 'user',
            'content' : f"Question: {question}\nText: {response}"
        }
    ]

    return conversation

# Let's then check the evaluation process
def judgement_prompt(question, answer, response):
    conversation = [
        {
            'role' : 'system',
            'content' : 'Please rate the consistency between the reference answer and the proposed answer on a scale of 0 to 1. A rating of 0 indicates inconsistency, while a rating of 1 indicates perfect consistency.'
        },
        {
            'role' : 'user',
            'content' : "Question: In which country is the Sky Train Rail bridge?\nReference Answer: Canada\nProposed Answer: Thailand"
        },
        {
            'role' : 'assistant',
            'content' : "0"
        },
        {
            'role' : 'user',
            'content' : "Question: What color is the lowest level of the Homeland Security Advisory System?\nReference Answer: Green\nProposed Answer: Blue"
        },
        {
            'role' : 'assistant',
            'content' : '0'
        },
        {
            'role' : 'user',
            'content' : "Question: After the United States and the Soviet Union, what country became the third in the world to test an atom bomb (in 1952)?\nReference Answer: Great Britain\nProposed Answer: United Kingdom"
        },
        {
            'role' : 'assistant',
            'content' : '1'
        },
        {
            'role' : 'user',
            'content' : "Question: What was Eddie Murphy's first movie?\nReference Answer: 48 Hours\nProposed Answer: 48 Hrs."
        },
        {
            'role' : 'assistant',
            'content' : '1',
        },
        {
            'role' : 'user',
            'content' : "Question: How long do NFL football teams have to get a play off (the play clock)?\nReference Answer: 40 seconds\nProposed Answer: 30 seconds"
        },
        {
            'role' : 'assistant',
            'content' : '0'
        },
        {
            'role' : 'user',
            'content' : f"Question: {question}\nReference Answer: {answer}\nProposed Answer: {response}"
        }
    ]

    return conversation

def construct_extraction_inputs(args, dataset_questions):
    questions = []
    responses = []
    for datapoint in dataset_questions:
        for response in datapoint[args.model_name]['sampling_pred_text']:
            questions.append(datapoint['question'].strip())
            responses.append(response.strip())

    inputs = []
    for question, response in zip(questions, responses):
        inputs.append(extraction_prompt(question, response))

    return inputs

def construct_judgement_inputs(args, dataset_questions, extraction_outputs):
    questions = []
    answers = []
    extracted_responses = []

    for i, datapoint in enumerate(dataset_questions):
        gold_answers = [answer.strip() for answer in datapoint['answers']]

        for j in range(10):
            question = datapoint['question'].strip()
            response = extraction_outputs[i*10+j].strip()
            for answer in gold_answers:
                questions.append(question)
                answers.append(answer)
                extracted_responses.append(response)

    inputs = []
    for question, answer, response in zip(questions, answers, extracted_responses):
        inputs.append(judgement_prompt(question, answer, response))

    return inputs

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def compute_exact_matches(args, dataset_questions, extraction_outputs):
    exact_matches = []
    for i, datapoint in enumerate(dataset_questions):
        gold_answers = [normalize_answer(answer) for answer in datapoint['answers']]

        for j in range(10):
            output = extraction_outputs[i*10 + j]
            response = normalize_answer(output)

            has_match = False
            for gold_answer in gold_answers:
                if response == gold_answer:
                    has_match = True
            exact_matches.append(has_match)
    return exact_matches

def collect_judgements_with_all_answers(args, dataset_questions, judgement_outputs, exact_matches):
    judgements = []
    selected_judgement_outputs = []
    judgement_start_idx = 0
    for i, exact_match in enumerate(exact_matches):
        # Get the judgements
        datapoint_idx = i // 10
        num_answers = len(dataset_questions[datapoint_idx]['answers'])
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

def collect_judgements_with_gold_answer(dataset_questions, judgement_outputs, exact_matches):
    judgements = []
    for i, output in enumerate(judgement_outputs):
        # Get the exact match
        exact_match = exact_matches[i]

        # Get the neural heuristic match
        value = extract_float(output.outputs[0].text)
        if value is None:
            value = 0
        neural_judgement = value > 0.7

        judgements.append(exact_match or neural_judgement)

    return judgement_outputs, judgements

def single_sample_fn(client, max_tokens, temperature, top_p, failure_message):
    def sample_fn(curr_input):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=curr_input,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )

            return response.choices[0].message.content
        except:
            return failure_message
    return sample_fn

def sample_extraction_outputs(args, inputs, client):
    print(f'There are {len(inputs)} inputs')
    start_time = time()

    extraction_function = single_sample_fn(client, 64, 0.0, 1.0, "no answer due to error")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        outputs =  list(executor.map(extraction_function, inputs))

    print(f'Took {(time() - start_time) / 60:.2f} minutes')
    return outputs

def sample_judgement_outputs(args, inputs, client):
    print(f'There are {len(inputs)} inputs')
    start_time = time()

    extraction_function = single_sample_fn(client, 5, 0.0, 1.0, "0 due to error")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        outputs =  list(executor.map(extraction_function, inputs))

    print(f'Took {(time() - start_time) / 60:.2f} minutes')
    return outputs

def process_and_predict(args, client, dataset_questions):
    # First extract the model responses
    inputs = construct_extraction_inputs(args, dataset_questions)
    extraction_outputs = sample_extraction_outputs(args, inputs, client)

    # Then determine accuracies of the extracted responses
    inputs = construct_judgement_inputs(args, dataset_questions, extraction_outputs)
    judgement_outputs = sample_judgement_outputs(args, inputs, client)
    exact_matches = compute_exact_matches(args, dataset_questions, extraction_outputs)

    selected_judgement_outputs, judgements = collect_judgements_with_all_answers(
        args, dataset_questions, judgement_outputs, exact_matches
    ) 

    return extraction_outputs, selected_judgement_outputs, judgements

def save_outputs(args, dataset_questions, extraction_outputs, judgement_outputs, judgements, split_num):
    # Update things
    for i, datapoint in enumerate(dataset_questions):
        curr_judgements = judgements[i*10:(i+1)*10]
        curr_extractions = [extraction_outputs[j] for j in range(i*10, (i+1)*10)]
        curr_judgement_outputs = [judgement_outputs[j] for j in range(i*10, (i+1)*10)]

        curr_knowns = ['known' if judgement else 'unknown' for judgement in curr_judgements]
        datapoint[args.model_name]['sampling_labels'] = curr_knowns
        datapoint[args.model_name]['sampling_knowns'] = len([val for val in curr_knowns if val == 'known'])
        datapoint[args.model_name]['sampling_extracted_text'] = curr_extractions
        datapoint[args.model_name]['sampling_judgement_text'] = curr_judgement_outputs


    data_path = os.path.join(PARENT_DIR, 'preds', f'{args.dataset_name}_preds_split_{split_num}.jsonl')
    with open(data_path, 'w', encoding='utf-8') as f:
        for datapoint in dataset_questions:
            f.write(json.dumps(datapoint) + '\n')

def main():
    args = get_configs()
    client = get_client()

    for split in range(args.num_splits):
        dataset_questions = load_dataset(args, split) 
        if "sampling_knowns" in dataset_questions[0][args.model_name]:
            continue

        print(f'Going through split {split} for {args.dataset_name}')
        extraction_outputs, judgement_outputs, judgements = process_and_predict(args, client, dataset_questions)
        save_outputs(args, dataset_questions, extraction_outputs, judgement_outputs, judgements, split)
    

if __name__ == "__main__":
    main()
