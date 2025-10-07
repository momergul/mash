#!/bin/bash

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki}
RETRIEVER_PORT=${2:-8000}

HYDRA_FULL_ERROR=1 python -m main_verl_trainer actor_rollout_ref.model.load_from_hf=True actor_rollout_ref.model.hf_model=qwen trainer.experiment_name="${DATASET_NAME}_search_r1" reward_model.rule_based_reward.retrieval_penalty_type=none trainer.only_keep_best_and_recent=True data.shortform_dataset_prefix=$DATASET_NAME actor_rollout_ref.actor.ppo_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=2 actor_rollout_ref.rollout.log_prob_micro_batch_size=2 actor_rollout_ref.rollout.retrieval_kwargs.shortform_retrieval.url="http://127.0.0.1:${RETRIEVER_PORT}/retrieve" actor_rollout_ref.rollout.retrieval_kwargs.max_searches=3
