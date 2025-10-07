#!/bash/bin

EXPERIMENT_NAME=$1
TEST_DATASET_NAME=$2
CHECKPOINT_NAME=$3
API_KEY=$4

python -m verl_training.abstention_scripts.abstention.few_shot_abstention --test_dataset_name=$TEST_DATASET_NAME --experiment_name=$EXPERIMENT_NAME --checkpoint_name=$CHECKPOINT_NAME
API_KEY=$API_KEY python -m verl_training.abstention_scripts.abstention.evaluate_abstention --test_dataset_name=$TEST_DATASET_NAME --experiment_name=$EXPERIMENT_NAME --output_subfolder=outputs
