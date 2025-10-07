#!/bin/bash

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki}

HYDRA_FULL_ERROR=1 python -m main_verl_trainer actor_rollout_ref.model.load_from_hf=True actor_rollout_ref.model.hf_model=qwen trainer.experiment_name="${DATASET_NAME}_r1" reward_model.rule_based_reward.retrieval_penalty_type=none trainer.only_keep_best_and_recent=True data.shortform_dataset_prefix=$DATASET_NAME actor_rollout_ref.actor.ppo_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=2 actor_rollout_ref.rollout.log_prob_micro_batch_size=2 data.shortform_prompt_type=r1 actor_rollout_ref.rollout.is_agentic=False actor_rollout_ref.rollout.response_length=512 data.max_response_length=512
