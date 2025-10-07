#!/bin/bash

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki}
REWARD_PENALTY=$2 # {exponential, otc, otc_strict}
if [$DATASET_NAME = "nq"]; then
    EXPONENTIAL_LAMBDA=0.5
else
    EXPONENTIAL_LAMBDA=0.8
fi
TEST_DATASET_NAME=$3
API_KEY=$4
LOAD_METRIC=${5:tool_productivity} # {tool_productivity, em_scores (for R1 and Search-R1)}
RETRIEVER_PORT=${6:-8000}



# Inference
HYDRA_FULL_ERROR=1 python -m main_verl_trainer actor_rollout_ref.model.load_from_hf=True actor_rollout_ref.model.hf_model="mash_${DATASET_NAME}_warm_start" trainer.experiment_name="${DATASET_NAME}_mash_${REWARD_PENALTY}" reward_model.rule_based_reward.retrieval_penalty_type=$REWARD_PENALTY trainer.only_keep_best_and_recent=True data.shortform_dataset_prefix=$TEST_DATASET_NAME actor_rollout_ref.actor.ppo_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=2 actor_rollout_ref.rollout.log_prob_micro_batch_size=2 actor_rollout_ref.rollout.retrieval_kwargs.shortform_retrieval.url="http://127.0.0.1:${RETRIEVER_PORT}/retrieve" reward_model.rule_based_reward.exponential_lambda=$EXPONENTIAL_LAMBDA trainer.pure_test=True trainer.test_load_metric=$LOAD_METRIC trainer.pure_test_suffix=$TEST_DATASET_NAME

# Processing with API
API_KEY=$API_KEY python -m verl_training.evaluation_reporting.update_metrics_with_judge_accuracy --mode=test --test_metric=$LOAD_METRIC --experiment_name="${DATASET_NAME}_mash_${REWARD_PENALTY}"
