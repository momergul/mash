#!/bash/bin

DATASET_NAME=$1 # {nq, hotpotqa, 2wiki}
AFH_METHOD=$2 # {absolute, multisample}

if [[ "$2" = "absolute" ]]; then
    GRADIENT_ACCUMULATION_STEPS=16
else
    GRADIENT_ACCUMULATION_STEPS=160
fi

python -m verl_training.abstention_scripts.abstention.alignment_for_honesty_training --train_batch_size=4 --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS --logging_steps=10 --model_name=qwen --dataset=$DATASET_NAME --processing_type=$AFH_METHOD --absolute_threshold=0.1 --experiment_name="${DATASET_NAME}_afh_${AFH_METHOD}"
