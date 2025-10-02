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
    questions = []
    answers = []
    with open(filepath, 'r') as f:
        for line in f:
            dp = json.loads(line.strip())
            if dp["split"] == "train":
                questions.append(dp['question'])
                answers.append(dp['answers'])
    return questions, answers

def get_client():
    API_KEY = os.getenv("API_KEY")
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client

def sample_questions(train_questions, answers):
    while True:
        sampled_questions = random.sample(train_questions, 5)
        associated_answers = [answers[train_questions.index(question)] for question in sampled_questions]

        for question in sampled_questions:
            print(question)
        happy = input("Are you happy with these questions? Y/N: ") == "Y"
        if happy:
            return sampled_questions, associated_answers

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

def get_outputs(sampled_questions, client):
    temperature = 1.0
    top_p = 0.8
    model = "deepseek-chat"
    max_tokens = 64

    inputs = [construct_prompt(question) for question in sampled_questions]
    outputs = []
    for curr_input in inputs:
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

def accept_outputs(sampled_questions, associated_answers, outputs):
    for question, answer, output in zip(sampled_questions, associated_answers, outputs):
        print(question)
        print(output)
        print(answer)
        print("-"*20)
        print()
    happy = input("Are you happy with these outputs? Y/N: ") == "Y"
    return happy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='nq')
    args = parser.parse_args()

    train_questions, answers = get_train_questions(args) 
    client = get_client()
    
    while True:
        sampled_questions, associated_answers = sample_questions(train_questions, answers)
        outputs = get_outputs(sampled_questions, client)
        if accept_outputs(sampled_questions, associated_answers, outputs):
            break

    parent_folder = f"data/model_accuracy_estimates/few_shot_examples/{args.dataset_name}_few_shot.pkl"
    with open(parent_folder, 'wb') as f:
        pairs = []
        for i in range(5):
            pairs.append((sampled_questions[i], outputs[i]))
        pickle.dump(pairs, f)

if __name__ == "__main__":
    main()
