# File: filter_naive_sft_generations.py
# -------------------------------------

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

import argparse
import os
import json
import pickle
import torch
import re
import string
import random


PARENT_DIR = 'data/naive_sft'

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='larger_qwen')
    parser.add_argument('--target_dataset', type=str, default='nq')
    parser.add_argument('--parametric_suffix', type=str, default='r1')
    parser.add_argument('--search_suffix', type=str, default='decomposition')
    parser.add_argument('--save_suffix', type=str, default='')
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    return parser.parse_args()

def extract_float(s):
    match = re.match(r'^[-+]?(\d+\.?\d*|\.\d+)', s)
    return float(match.group()) if match else None

def get_eval_model(args):
    hf_name = 'Qwen/Qwen2.5-72B-Instruct'
    model = LLM(model=hf_name, dtype=torch.bfloat16, gpu_memory_utilization=0.9,
                max_model_len=2048, enable_prefix_caching=True, tensor_parallel_size=args.tensor_parallel_size)
    return model
    
def load_dataset(args):
    dataset_questions = []

    p_suffix = args.parametric_suffix
    s_suffix = args.search_suffix

    suffixes = [f'{args.target_dataset}_naive_sft_{p_suffix}', f'{args.target_dataset}_naive_sft_{s_suffix}']
    for suffix in suffixes:
        filepath = os.path.join(PARENT_DIR, f'{args.model_name}_{suffix}.pkl')
        with open(filepath, 'rb') as f:
            question_to_responses = pickle.load(f)

        for question, responses in question_to_responses.items():
            for trajectory, answer_set, ground_truths in responses:
                if answer_set is None:
                    continue
                for answer in answer_set:
                    curr_datapoint = {
                        'question' : question,
                        'trajectory' : trajectory,
                        'model_answer' : answer,
                        'ground_truths' : ground_truths
                    }
                    dataset_questions.append(curr_datapoint)
    
    print(len(dataset_questions))

    return dataset_questions
    
def extraction_prompt(question, response):
    conversation = [
        {
            'role' : 'system',
            'content' : 'Given a question and a piece of text, if the text does not contain an answer to the question, output "no answer"; otherwise, extract the answer from the text.'
        },
        {
            'role' : 'user',
            'content' : """\
Question: What was the last US state to reintroduce alcohol after prohibition?
Text: The last US state to reintroduce alcohol after prohibition was Mississippi. Mississippi legalized alcohol on August 17, 1933, making it the last state to do so. Prohibition had been enacted nationwide through the 18th Amendment to the US Constitution in 1919, but it was repealed in 1933 with the passage of the 21st Amendment, which allowed individual"""
        },
        {
            'role' : 'assistant',
            'content' : "Mississippi"
        },
        {
            'role' : 'user',
            'content' : """\
Question: What was the name of Michael Jackson's autobiography written in 1988?
Text: Michael Jackson did not write an autobiography in 1988."""
        },
        {
            'role' : 'assistant',
            'content' : 'no answer'
        },
        {
            'role' : 'user',
            'content' : """\
Question: Who was the director of the CIA from 1976-81?
Text: Stansfield Turner."""
        },
        {
            'role' : 'assistant',
            'content' : 'Stansfield Turner'
        },
        {
            'role' : 'user',
            'content' : """\
Question: Which 90s sci fi series with James Belushi was based on Bruce Wagner's comic strip of the same name?
Text: The 90s sci-fi series based on Bruce Wagner's comic strip was "The Adventures of Buckaroo Banzai Across the 8th Dimension." It starred James Belushi as the titular character and aired in 1984."""
        },
        {
            'role' : 'assistant',
            'content' : "The Adventures of Buckaroo Banzai Across the 8th Dimension."
        },
        {
            'role' : 'user',
            'content' : """\
Question: In what years did John Henry win the Arlington Million?
Text: I'm sorry, but I don't have information about John Henry winning the Arlington Million. John Henry was a thoroughbred race horse, but I don't have details about his victories in specific races, including the Arlington Million. To provide an accurate answer, I would need to check a reliable horse racing database or records. If you have any other questions about John Henry or thoroughbred racing, I'd be happy to try to help with that information."""
        },
        {
            'role' : 'assistant',
            'content' : 'no answer'
        },
        {
            'role' : 'user',
            'content' : """\
Question: Art Garfunkel trained for which profession although he didn't qualify?
Text: Art Garfunkel trained to be a pharmacist, but he did not qualify."""
        },
        {
            'role' : 'assistant',
            'content' : 'pharmacist'
        },
        {
            'role' : 'user',
            'content' : """\
Question: What was the name of Drew Barrymore's character in E.T.?
Text: Drew Barrymore played the character of Elliott in the movie E.T. The Extra-Terrestrial."""
        },
        {
            'role' : 'assistant',
            'content' : 'Elliott'
        },
        {
            'role' : 'user',
            'content' : """\
Question: Which British monarch popularized the Homgburg which came from the German town of the same name?
Text: The British monarch who popularized the Hamburg (also spelled Hamburgh) cake, which originated from the German town of Hamburgh (now known as Hamburg), was Queen Victoria. She was known for her love of German cuisine and often featured German dishes, including the Hamburg cake, in her royal banquets and at official receptions."""
        },
        {
            'role' : 'assistant',
            'content' : 'Queen Victoria'
        },
        {
            'role' : 'user',
            'content' : f"""\
Question: {question}
Text: {response}"""
        }
    ]

    return conversation

def judgement_prompt(question, answer, response):
    conversation = [
        {
            'role' : 'system',
            'content' : 'Please rate the consistency between the reference answer and the proposed answer on a scale of 0 to 1. A rating of 0 indicates inconsistency, while a rating of 1 indicates perfect consistency.'
        },
        {
            'role' : 'user',
            'content' : """\
Question: In which country is the Sky Train Rail bridge?
Reference Answer: Canada
Proposed Answer: Thailand"""
        },
        {
            'role' : 'assistant',
            'content' : "0"
        },
        {
            'role' : 'user',
            'content' : """\
Question: What color is the lowest level of the Homeland Security Advisory System?
Reference Answer: Green
Proposed Answer: Blue"""
        },
        {
            'role' : 'assistant',
            'role' : '0'
        },
        {
            'role' : 'user',
            'content' : """\
Question: After the United States and the Soviet Union, what country became the third in the world to test an atom bomb (in 1952)?
Reference Answer: Great Britain
Proposed Answer: United Kingdom"""
        },
        {
            'role' : 'assistant',
            'content' : '1'
        },
        {
            'role' : 'user',
            'content' : """\
Question: What was Eddie Murphy's first movie?
Reference Answer: 48 Hours
Proposed Answer: 48 Hrs."""
        },
        {
            'role' : 'assistant',
            'content' : '1',
        },
        {
            'role' : 'user',
            'content' : """\
Question: How long do NFL football teams have to get a play off (the play clock)?
Reference Answer: 40 seconds
Proposed Answer: 30 seconds"""
        },
        {
            'role' : 'assistant',
            'content' : '0'
        },
        {
            'role' : 'user',
            'content' : f"""\
Question: {question}
Reference Answer: {answer}
Proposed Answer: {response}"""
        }
    ]

    return conversation

def construct_extraction_inputs(args, dataset_questions):
    questions = []
    responses = []
    for datapoint in dataset_questions:
        questions.append(datapoint['question'])
        response = datapoint['model_answer'][:datapoint['model_answer'].index('</answer>')].strip()
        responses.append(response)

    inputs = []
    for question, response in zip(questions, responses):
        inputs.append(extraction_prompt(question, response))

    return inputs

def construct_judgement_inputs(args, dataset_questions, extraction_outputs):
    questions = []
    answers = []
    extracted_responses = []

    for i, datapoint in enumerate(dataset_questions):
        gold_answers = [answer.strip() for answer in datapoint['ground_truths']]
        for answer in gold_answers:
            questions.append(datapoint['question'].strip())
            answers.append(answer)
            extracted_responses.append(extraction_outputs[i].outputs[0].text.strip())

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
        gold_answers = [normalize_answer(answer) for answer in datapoint['ground_truths']]
        output = extraction_outputs[i]
        response = normalize_answer(output.outputs[0].text)

        has_match = False
        for gold_answer in gold_answers:
            if response == gold_answer:
                has_match = True
        exact_matches.append(has_match)
    return exact_matches

def construct_trajectories(dataset_questions, judgement_outputs, exact_matches):
    # Let's first get the questions with correct responses
    question_to_correct_trajectories = {}
    judgement_start_idx = 0

    for i, exact_match in enumerate(exact_matches):
        num_answers = len(dataset_questions[i]['ground_truths'])
        datapoint_judgements = [judgement_outputs[j] for j in range(judgement_start_idx, judgement_start_idx+num_answers)]
        judgement_start_idx += num_answers

        # Determine the highest judgement
        best_output_val = -1
        best_gt = None
        for output, answer in zip(datapoint_judgements, dataset_questions[i]['ground_truths']):
            value = extract_float(output.outputs[0].text)
            if value is None:
                value = 0

            if value > best_output_val:
                best_output_val = value
                best_gt = answer
        neural_judgement = best_output_val > 0.7
        correct = exact_match or neural_judgement

        if correct:
            question = dataset_questions[i]['question']
            if question not in question_to_correct_trajectories:
                question_to_correct_trajectories[question] = set()
            full_response = dataset_questions[i]['trajectory'] + best_gt + "</answer>"
            question_to_correct_trajectories[question].add(full_response)
    question_to_correct_trajectory = {question : random.sample(list(trajectories), 1)[0] for question, trajectories in question_to_correct_trajectories.items()}

    # Then handle the questions where the model flopped
    # 1) Map from question to trajectories
    question_to_incorrect_trajectories = {}
    for datapoint in dataset_questions:
        question = datapoint['question']
        if question in question_to_correct_trajectory:
            continue

        if question not in question_to_incorrect_trajectories:
            question_to_incorrect_trajectories[question] = []
        question_to_incorrect_trajectories[question].append((datapoint['trajectory'], datapoint['model_answer']))
        
    # 2) Map to trajectory with the shortest answer to be in-distribution
    question_to_incorrect_trajectory = {}
    for question, mapped_items in question_to_incorrect_trajectories.items():
        trajectory, answer = sorted(mapped_items, key=lambda x: len(x[1]))[0]
        question_to_incorrect_trajectory[question] = trajectory + answer
    num_incorrects = len(question_to_incorrect_trajectory)

    question_to_correct_trajectory.update(question_to_incorrect_trajectory)
    incorrect_rate = 100 * num_incorrects / len(question_to_correct_trajectory)
    return question_to_correct_trajectory, question_to_incorrect_trajectory, incorrect_rate

def process_and_predict(args, model, dataset_questions):
    # First extract the model responses
    tp_rank = vllm_ps.get_tensor_model_parallel_rank() if args.tensor_parallel_size != 1 else 0    
    if tp_rank == 0:
        inputs = construct_extraction_inputs(args, dataset_questions)
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=128)
        broadcast_data = {'inputs' : inputs, 'sampling_params' : sampling_params}
    else:
        broadcast_data = None
    inputs, sampling_params = broadcast_data["inputs"], broadcast_data["sampling_params"]
    extraction_outputs = model.chat(inputs, sampling_params=sampling_params, use_tqdm=True)

    # Then determine accuracies of the extracted responses
    if tp_rank == 0:
        inputs = construct_judgement_inputs(args, dataset_questions, extraction_outputs)
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=5)
        broadcast_data = {'inputs' : inputs, 'sampling_params' : sampling_params}
    else:
        broadcast_data = None
    inputs, sampling_params = broadcast_data["inputs"], broadcast_data["sampling_params"]
    judgement_outputs = model.chat(inputs, sampling_params=sampling_params, use_tqdm=True) 
    exact_matches = compute_exact_matches(args, dataset_questions, extraction_outputs)

    return construct_trajectories(dataset_questions, judgement_outputs, exact_matches)

def save_outputs(args, question_to_response, question_to_incorrect_responses):
    # First save the correct responses in the desired format
    dataset_list = []
    for question, response in question_to_response.items():
        dataset_list.append({'question' : question, 'response' : response})

    search_suffix = args.search_suffix
    dataset_path = os.path.join(PARENT_DIR, f'{args.model_name}_{args.target_dataset}_{search_suffix}_naive_sft.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset_list, f)

    # Then save the incorrect outputs for later inspection
    incorrect_path = os.path.join(PARENT_DIR, f'{args.model_name}_{args.target_dataset}_{search_suffix}_incorrect_outputs_in_naive_sft.pkl')
    with open(incorrect_path, 'wb') as f:
        pickle.dump(question_to_incorrect_responses, f)

def main():
    args = get_configs()
    model = get_eval_model(args)
    dataset_questions = load_dataset(args)
    question_to_response, question_to_incorrect_responses, incorrect_rate = process_and_predict(args, model, dataset_questions) 
    print(f'{incorrect_rate}% of outputs are incorrect, alas')
    save_outputs(args, question_to_response, question_to_incorrect_responses)
    

if __name__ == "__main__":
    main()
