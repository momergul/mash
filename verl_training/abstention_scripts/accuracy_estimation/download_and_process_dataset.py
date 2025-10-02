# File: download_and_process_overall_triviaqa.py
# ----------------------------------------------

import pandas as pd
import os
import json
import pickle
from datasets import load_dataset
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='nq')
    args = parser.parse_args()
    construct_dataset_data(args)

def construct_dataset_data(args):
    root_path = 'data/shortform_qa'
    splits = ["train", "test"]
    #splits = ["test"]

    questions = []
    for split in splits:
        parquet_path = os.path.join(root_path, f'{args.dataset_name}_{split}.parquet')
        df = pd.read_parquet(parquet_path)

        if split == "train":
            question_path = os.path.join(f"experiments/agentic_verl/{args.dataset_name}_datapoints/questions.pkl")
            with open(question_path, 'rb') as f:
                training_questions = set(pickle.load(f))

            valid_indices = []
            for i in range(len(df)):
                if df.iloc[i]['question'] in training_questions:
                    valid_indices.append(i)
            df = df.iloc[valid_indices]
            print(len(df))

        for i in range(len(df)):
            question_dict = {
                'question' : df.iloc[i]['question'],
                'question_id' : df.iloc[i]['id'],
                'answers' : [answer for answer in df.iloc[i]['golden_answers']],
                'gold_answer' : df.iloc[i]['golden_answers'][0],
                'dataset' : df.iloc[i]['extra_info']['data_source'],
                'split' : split
            }
            questions.append(question_dict)

    # Save json
    filename = f'{args.dataset_name}_preds.jsonl'
    save_path = os.path.join('data', 'model_accuracy_estimates', 'preds', filename)
    with open(save_path, 'w', encoding='utf-8') as f:
        for datapoint in questions:
            f.write(json.dumps(datapoint) + '\n')

if __name__ == "__main__":
    main()
