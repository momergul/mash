#!/bash/bin

FEW_SHOT_DATASET=$1
TEST_DATASET_NAME=$2
API_KEY=$3

python -m verl_training.abstention_scripts.abstention.few_shot_abstention --test_dataset_name=$TEST_DATASET_NAME --few_shot_dataset_name=$FEW_SHOT_DATASET
API_KEY=$API_KEY python -m verl_training.abstention_scripts.abstention.evaluate_abstention --test_dataset_name=$TEST_DATASET_NAME --experiment_name="${FEW_SHOT_DATASET}_qwen_few_shot"
