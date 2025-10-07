#!/bash/bin

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki, triviaqa}
API_KEY=$2

python -m verl_training.abstention_scripts.accuracy_estimation.download_and_process_dataset --dataset_name=$DATASET_NAME
python -m verl_training.abstention_scripts.sample_model_responses --dataset_name=$DATASET_NAME
python -m verl_training.abstention_scripts.accuracy_estimation.split_and_join --dataset_name=$DATASET_NAME --action=split
API_KEY=$API_KEY python -m verl_training.abstention_scripts.accuracy_estimation.evaluate_model_responses --dataset_name=$DATASET_NAME --num_threads=128
python -m verl_training.abstention_scripts.accuracy_estimation.split_and_join --dataset_name=$DATASET_NAME --action=join
