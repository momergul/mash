# File: warm_start_training.py
# ----------------------------------------
# Script for warm-starting models on reasoning traces with interleaved retrieval and thoughts

import argparse
import os
import wandb
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from trl import get_peft_config, SFTConfig, SFTTrainer
from trl_trainers.utils import DataCollatorForRetrievalInterleavedLM

import torch
import gc

## UTILITIES ##

def get_config():
    parser = argparse.ArgumentParser()

    # General training arguments
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='How many gradient updates will we perform?')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='How many training steps before we evaluate the model/log results?')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help="How many distinct prompts will we train on per step (independent of processes)?")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)    
    parser.add_argument('--max_seq_length', type=int, default=8192)

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

    # PEFT, if used
    parser.add_argument('--use_peft', action='store_true')
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # Model and data arguments
    parser.add_argument('--model_name', type=str, default='qwen_instruct', help='What model to use?')
    parser.add_argument('--warm_start_folder', type=str, default='naive_sft',
                        help="Which folder is the warm_start data in?")
    parser.add_argument('--warm_start_filename', type=str, default='',
                        help="What is the filename?")

    # Experiment arguments
    parser.add_argument('--base_folder', type=str,
                        default='experiments/pre_verl_sft')
    parser.add_argument('--project_name', type=str, default='verl_sft')    
    parser.add_argument('--experiment_name', type=str, default='testing')

    args = parser.parse_args()
    return args

def get_checkpoint(args):
    output_dir = os.path.join(args.base_folder, args.experiment_name)
    last_checkpoint = get_last_checkpoint(output_dir)
    print("Last checkpoint: ", last_checkpoint)
    return last_checkpoint

def get_sft_config(args):
    output_dir = os.path.join(args.base_folder, args.experiment_name)

    return SFTConfig(
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
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta_1,
        adam_beta2=args.adam_beta_2,
        max_grad_norm=args.gradient_clip_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        max_seq_length=args.max_seq_length,
        packing=False,
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
    prompt = "Answer the given question. You must conduct reasoning between <think> and </think> " +\
        "every time you get new information. After reasoning, if you find you lack some knowledge, you can call " +\
        "a search engine by <search> query </search> and it will return the top searched results between <document> and </document>. " +\
        "You need to make every search call count and gain helpful results. If you find no further external knowledge is needed, " +\
        "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, " +\
        f"<answer> Beijing </answer>. Question: {question}\n"
    response = example['response']

    return {
        "messages" : [
            {"role" : "user", "content" : prompt},
            {"role" : "assistant", "content" : response.strip()},
        ]
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

def get_dataset(args, tokenizer):
    # Load the dataset
    filepath = os.path.join('data', args.warm_start_folder, args.warm_start_filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    dataset = Dataset.from_list(data)

    # Get the response template
    response_template, document_template = get_templates(args, dataset[0], tokenizer)
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

    return dataset, response_template, document_template

## MAIN ##

def main():
    args = get_config()

    # Get SFT config
    sft_args = get_sft_config(args)

    # Get model
    model, tokenizer = get_model_and_tokenizer(args)
    train_dataset, response_template, document_template = get_dataset(args, tokenizer)
    collator = DataCollatorForRetrievalInterleavedLM(
        document_template,
        response_template,
        tokenizer=tokenizer
    )

    # Do your training
    wandb.init(project=args.project_name, name=args.experiment_name)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator
    ) 

    # Resume from checkpoint if needed
    last_checkpoint = get_checkpoint(args)
    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    main()
