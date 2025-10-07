#!/bin/bash

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki}

VLLM_ATTENTION_BACKEND=XFORMERS python -m verl_training.naive_sft.filter_naive_sft_generations --model_name=larger_qwen --target_dataset=${DATASET_NAME} --tensor_parallel_size=2 --search_suffix=oracle_baseline
CUDA_VISIBLE_DEVICES=0 python -m verl_training.naive_sft.warm_start_training --model_name=qwen --experiment_name="mash_${DATASET_NAME}_warm_start_oracle_helper" --gradient_accumulation_steps=16 --train_batch_size=1 --distillation_filename="larger_qwen_${DATASET_NAME}_oracle_baseline_naive_sft.pkl"


