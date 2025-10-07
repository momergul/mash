#!/bash/bin

EXPERIMENT_NAME=$1
TEST_DATASET_NAME=$2

python -m verl_training.evaluation_reporting.report_abstention_model_all_metrics --experiment_name=$EXPERIMENT_NAME --dataset=$TEST_DATASET_NAME
