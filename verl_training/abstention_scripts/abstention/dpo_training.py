# File: retrieval_distillation_training.py
# ----------------------------------------
# Script for finetuning models on reasoning traces with interleaved retrieval and thoughts

import argparse
import os
import wandb
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from trl import get_peft_config, DPOConfig, DPOTrainer

import torch
import gc
from copy import deepcopy
import json
from collections import defaultdict

DATA_DIR = 'data/model_accuracy_estimates/preds'

## UTILITIES ##

def get_config():
    parser = argparse.ArgumentParser()

    # General training arguments
    parser.add_argument('--max_epochs', type=int, default=1,
                        help='How many gradient updates will we perform?')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='How many training steps before we evaluate the model/log results?')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help="How many distinct prompts will we train on per step (independent of processes)?")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)    
    parser.add_argument('--max_seq_length', type=int, default=2048)

    # Optimization
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help="Set if you will use gradient checkpointing")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_beta_1', type=float, default=0.9)
    parser.add_argument('--adam_beta_2', type=float, default=0.95)
    parser.add_argument('--gradient_clip_norm', type=float, default=1)
    parser.add_argument('--lr_scheduler_type', type=str, choices=['cosine', 'linear'],
                        default='linear')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)

    # DPO-specific arguments
    parser.add_argument('--sft_weight', type=float, default=0.2)
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--answerability_threshold', type=float, default=0.1)

    # PEFT, if used
    parser.add_argument('--use_peft', action='store_true')
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # Model and data arguments
    parser.add_argument('--model_name', type=str, default='qwen', help='What model to use?')
    parser.add_argument('--dataset_name', type=str, default='nq', help='What dataset to train for?')
    parser.add_argument('--processing_type', type=str, default='naive')

    # Experiment arguments
    parser.add_argument('--base_folder', type=str,
                        default='experiments/abstention_experiments')
    parser.add_argument('--project_name', type=str, default='verl_sft')    
    parser.add_argument('--experiment_name', type=str, default='testing')

    args = parser.parse_args()
    return args

def get_checkpoint(args):
    output_dir = os.path.join(args.base_folder, args.experiment_name)
    last_checkpoint = get_last_checkpoint(output_dir)
    print("Last checkpoint: ", last_checkpoint)
    return last_checkpoint

def get_dpo_config(args):
    output_dir = os.path.join(args.base_folder, args.experiment_name)
    loss_type = ["sigmoid", "sft"]
    loss_weights = [1.0, args.sft_weight]

    return DPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.max_epochs,
        logging_strategy='steps',
        save_strategy='epoch',
        logging_steps=args.logging_steps,
        report_to='wandb',
        
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta_1,
        adam_beta2=args.adam_beta_2,
        max_grad_norm=args.gradient_clip_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        precompute_ref_log_probs=True,
        loss_type=loss_type,
        loss_weights=loss_weights,
        beta=args.dpo_beta,

        max_length=args.max_seq_length,
        push_to_hub=False,
        use_liger_kernel=True
    )

## MODELS AND DATA ##

def get_model_and_tokenizer(args):
    # Get the model string
    model_name = "Qwen/Qwen2.5-3B"
    if 'instruct' in args.model_name:
        model_name += '-Instruct'

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Initialize PEFT
    if args.use_peft:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules='all-linear',
            init_lora_weights='gaussian'
        )
        model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:    
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_conversation(example):
    question = example['question']
    prompt = "Answer the given question. If you are not confident that your answer will be correct, " +\
        "you should abstain from answering by using the phrase 'I am afraid I cannot help you as I do " +\
        f"not know the answer to this question.' Question: {question}"

    chosen = example['chosen']
    rejected = example['rejected']

    return {
        "prompt" : [{'role' : 'user', 'content' : prompt}],
        "chosen" : [{'role' : 'assistant', 'content' : chosen}],
        "rejected" : [{'role' : 'assistant', 'content' : rejected}],
    }


def get_templates(args, example, tokenizer):
    # First get the response template
    conversation = create_conversation(example)
    prompt = conversation['messages'][:-1]
    tokens_without_prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=False)
    tokens_with_prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    response_template = tokens_with_prefix[len(tokens_without_prefix):]
    document_template = ("<document>", "</document>")

    return response_template, document_template

def process_datapoints(args, raw_datapoints):
    new_datapoints = []
    unknown_response = "I am afraid I cannot help you as I do not know the answer to this question."

    for datapoint in raw_datapoints:
        answerable = (datapoint[args.model_name]['sampling_knowns'] / 10) >= args.answerability_threshold
        response = pick_first_response(args, datapoint, answerable) 
        response = remove_answer_prefix(response)
        if answerable:
            chosen = response
            rejected = unknown_response
        else:
            chosen = unknown_response
            rejected = response

        new_datapoint = {'question' : datapoint['question'], 'chosen' : chosen, 'rejected' : rejected}
        new_datapoints.append(new_datapoint)

    return new_datapoints

def remove_answer_prefix(response):
    if response is not None and response.startswith("Answer:"):
        response = response[len("Answer:"):].strip()
    return response

def pick_first_response(args, datapoint, answerable):
    for sampling_pred_text, sampling_label in zip(
            datapoint[args.model_name]['sampling_pred_text'],
            datapoint[args.model_name]['sampling_labels']
    ):
        if answerable and sampling_label == 'known':
            return sampling_pred_text
        if not answerable and sampling_label != 'known':
            return sampling_pred_text            

def get_raw_datapoints(args):
    # First get all datapoints
    data_path = os.path.join(DATA_DIR, f'{args.dataset_name}_preds.jsonl')
    raw_datapoints = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dp = json.loads(line.strip())
            if dp["split"] != "train":
                continue
            # Very rare (<3 per dataset)
            if "no answer due to error" in dp[args.model_name]["sampling_extracted_text"]:
                continue
            raw_datapoints.append(dp)
    return raw_datapoints

def get_dataset(args, tokenizer):
    # Load the dataset
    raw_datapoints = get_raw_datapoints(args)
    datapoints = process_datapoints(args, raw_datapoints)
    dataset = Dataset.from_list(datapoints)

    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    return dataset

## MAIN ##

def main():
    args = get_config()

    # Get SFT config
    dpo_args = get_dpo_config(args)

    # Get model
    model, tokenizer = get_model_and_tokenizer(args)
    train_dataset = get_dataset(args, tokenizer)

    # Do your training
    wandb.init(project=args.project_name, name=args.experiment_name)
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    ) 

    # Resume from checkpoint if needed
    last_checkpoint = get_checkpoint(args)
    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    main()
