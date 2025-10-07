#!/bash/bin

EXPERIMENT_NAME=$1
LOAD_METRIC=${2:-"tool_productivity"}

python -m verl_training.evaluation_reporting.report_tool_use_model_abstention --experiment_name=$EXPERIMENT_NAME --metric_name=$LOAD_METRIC
