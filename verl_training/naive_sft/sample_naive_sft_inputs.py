# File: sample_naive_sft_data.py
# ------------------------------

import argparse
import os
import pandas as pd
import random


PARENT_DIR = 'data/shortform_qa'

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='nq')
    parser.add_argument('--num_searches', type=int, default=2)
    return parser.parse_args()

def subsample_dataset_naive_sft_inputs(args, train_df, dataset, curr_parametric_size, curr_search_size):
    # Subsample dataset specific indices
    dataset_indices = [i for i in range(len(train_df)) if train_df.iloc[i]['extra_info']['data_source'] == dataset]
    subsampled_indices = random.sample(dataset_indices, curr_parametric_size+curr_search_size)

    # Get the parametric and search indices
    parametric_indices = subsampled_indices[:curr_parametric_size]
    search_indices = subsampled_indices[curr_parametric_size:]

    # Construct the respective dataframes
    parametric_df = train_df.iloc[parametric_indices]
    search_df = train_df.iloc[search_indices]
    return parametric_df, search_df

def get_dataset_to_sizes(args, datasets):
    # First determine a number of parametric datapoints divisible by 4
    parametric_estimate = 1000 // (args.num_searches+1)
    parametric_size = (parametric_estimate // 4)*4
    search_size = 1000 - parametric_size

    # Allocation for parametric
    dataset_parametric_sizes = []
    num_remaining = parametric_size % len(datasets)
    for i in range(len(datasets)):
        curr_size = parametric_size // len(datasets)
        if i < num_remaining:
            curr_size += 1
        dataset_parametric_sizes.append(curr_size)

    # Allocation for search
    dataset_search_sizes = []
    num_remaining = search_size % len(datasets)
    for i in range(len(datasets)):
        curr_size = search_size // len(datasets)
        if i < num_remaining:
            curr_size += 1
        dataset_search_sizes.append(curr_size)

    dataset_to_sizes = {datasets[i] : (dataset_parametric_sizes[i], dataset_search_sizes[i]) for i in range(len(datasets))}
    return dataset_to_sizes
            
def construct_naive_sft_inputs(args):
    # Load the train set and get the datasets there
    train_dataset_path = os.path.join(PARENT_DIR, f'{args.target_dataset}_train.parquet')
    train_df = pd.read_parquet(train_dataset_path)
    datasets = list({train_df.iloc[i]['extra_info']['data_source'] for i in range(len(train_df))})
    dataset_to_sizes = get_dataset_to_sizes(args, datasets)
    print(dataset_to_sizes)

    # Then, construct the datapoints for each
    parametric_dfs = []
    search_dfs = []
    for dataset in datasets:
        curr_parametric_size, curr_search_size = dataset_to_sizes[dataset]
        curr_parametric, curr_search = subsample_dataset_naive_sft_inputs(args, train_df, dataset, curr_parametric_size, curr_search_size)
        parametric_dfs.append(curr_parametric)
        search_dfs.append(curr_search)

    # Then, we save
    for df_list, file_suffix in zip([parametric_dfs, search_dfs], ['parametric', 'search']):
        joined_df = pd.concat(df_list, ignore_index=True)
        for data_split in ['train', 'fast_dev', 'test']:
            savepath = os.path.join(PARENT_DIR, f'{args.target_dataset}_naive_sft_{file_suffix}_{data_split}.parquet')
            joined_df.to_parquet(savepath, index=False)

def main():
    args = get_configs()
    construct_naive_sft_inputs(args)

if __name__ == "__main__":
    main()
