#!/bin/bash

MODEL_NAME=$1 # {nq, hotpotqa, 2wiki}
TEST_DATASET_NAME=$2
API_KEY=$3
RETRIEVER_PORT=${4:-8000}



# Inference
HYDRA_FULL_ERROR=1 python -m main_verl_trainer actor_rollout_ref.model.load_from_hf=True actor_rollout_ref.model.hf_model="momergul/${MODEL_NAME}" trainer.experiment_name=$MODEL_NAME reward_model.rule_based_reward.retrieval_penalty_type=none trainer.only_keep_best_and_recent=True data.shortform_dataset_prefix=$TEST_DATASET_NAME actor_rollout_ref.actor.ppo_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=2 actor_rollout_ref.rollout.log_prob_micro_batch_size=2 actor_rollout_ref.rollout.retrieval_kwargs.shortform_retrieval.url="http://127.0.0.1:${RETRIEVER_PORT}/retrieve" trainer.pure_test=True trainer.test_load_metric=tool_productivity trainer.pure_test_suffix=$TEST_DATASET_NAME trainer.test_checkpoint_step=0

# Processing with API
API_KEY=$API_KEY python -m verl_training.evaluation_reporting.update_metrics_with_judge_accuracy --mode=test --test_metric=tool_productivity --experiment_name=$MODEL_NAME
