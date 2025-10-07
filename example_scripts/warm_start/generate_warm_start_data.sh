#!/bin/bash

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki}

python -m verl_training.naive_sft.sample_naive_sft_inputs --target_dataset=${DATASET_NAME} --num_searches=2
python -m verl_training.naive_sft.generate_naive_sft_data --model_name=larger_qwen --target_dataset=${DATASET_NAME} --prompt_type=r1 --max_actions=0
python -m verl_training.naive_sft.generate_naive_sft_data --model_name=larger_qwen --target_dataset=${DATASET_NAME} --prompt_type=decomposition --max_actions=2
