import json
import pickle
import pandas as pd
from openai import OpenAI
import random
from collections import Counter, defaultdict
import os
import argparse

def get_train_questions(args):
    filepath = f"data/model_accuracy_estimates/preds/{args.dataset_name}_preds.jsonl"
    em_to_questions = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            dp = json.loads(line.strip())
            if dp["split"] != "train":
                continue
            
            # Ensure that the assessments are valid
            if "no answer due to error" in dp[args.model_name]["sampling_extracted_text"]:
                continue
            if "0 due to error" in dp[args.model_name]["sampling_judgement_text"]:
                continue

            average_acc = dp[args.model_name]["sampling_knowns"] / 10
            em_to_questions[average_acc].append(dp)

    return em_to_questions

def get_client():
    API_KEY = os.getenv("API_KEY")
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client

def sample_questions(em_to_questions, i):
    num_1s = 3 if i % 2 == 0 else 2
    num_0s = 5 - num_1s

    while True:
        sampled_1s = random.sample(em_to_questions[1.0], num_1s)
        sampled_0s = random.sample(em_to_questions[0.0], num_0s)
        sampled_questions = [0 for _ in range(5)]
        for j in range(num_1s):
            sampled_questions[j*2 + (i%2)] = sampled_1s[j]
        for j in range(num_0s):
            sampled_questions[j*2 + ((i+1)%2)] = sampled_0s[j]
        
        actual_questions = [dp["question"] for dp in sampled_questions]
        for question in actual_questions:
            print(question)
        happy = input("Are you happy with these questions? Y/N: ") == "Y"
        if happy:
            return sampled_questions

def construct_prompt(question):
    conversation = [
        {
            'role' : 'system', 
            'content' : "You are a helpful assistant. The user will ask you a knowledge-intensive question. You should answer them concisely, providing reasoning when appropriate."
        },
        {
            'role' : 'user',
            'content' : f"Answer the following question: {question}"
        }
    ]

    return conversation

def get_outputs(args, sampled_questions, client):
    temperature = 1.0
    top_p = 0.8
    model = "deepseek-chat"
    max_tokens = 64

    outputs = []
    for dp in sampled_questions:
        average_acc = dp[args.model_name]["sampling_knowns"] / 10
        if average_acc == 0:
            outputs.append("I am afraid I cannot help you as I do not know the answer to this question.")
            continue
        
        curr_input = construct_prompt(dp["question"])
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=curr_input,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )
        outputs.append(response.choices[0].message.content)

    return outputs

def accept_outputs(args, sampled_questions, outputs):
    for dp, output in zip(sampled_questions, outputs):
        question = dp["question"]
        answer = dp["answers"]
        average_acc = dp[args.model_name]["sampling_knowns"] / 10

        print(question)
        print(answer, average_acc)
        print(output)
        print("-"*20)
        print()
    happy = input("Are you happy with these outputs? Y/N: ") == "Y"
    return happy

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--model_name', type=str, default='qwen')
    parser.add_argument('--num_prompts', type=int, default=4)
    args = parser.parse_args()

    em_to_questions = get_train_questions(args)
    client = get_client()

    # Get the question-accuracy-answer tuples
    for i in range(args.num_prompts):
        while True:
            sampled_questions = sample_questions(em_to_questions, i)
            outputs = get_outputs(args, sampled_questions, client)
            if accept_outputs(args, sampled_questions, outputs):
                break

        parent_folder = f"data/model_accuracy_estimates/few_shot_abstention/{args.dataset_name}_{args.model_name}_few_shot_{i}.pkl"
        with open(parent_folder, 'wb') as f:
            pairs = []
            for dp, output in zip(sampled_questions, outputs):
                question = dp['question']
                pairs.append((question, output))
            pickle.dump(pairs, f)

if __name__ == "__main__":
    main()
