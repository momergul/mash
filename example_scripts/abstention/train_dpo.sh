#!/bash/bin

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki}

python -m verl_training.abstention_scripts.abstention.dpo_training --train_batch_size=4 --gradient_accumulation_steps=16 --logging_steps=10 --model_name=qwen --dataset=$DATASET_NAME --experiment_name="${DATASET_NAME}_dpo" --sft_weight=1.0 --answerability_threshold=0.1
