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
from trl import get_peft_config, SFTConfig, SFTTrainer
from trl_trainers.utils import DataCollatorForRetrievalInterleavedLM

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

    # PEFT, if used
    parser.add_argument('--use_peft', action='store_true')
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # Model and data arguments
    parser.add_argument('--model_name', type=str, default='qwen', help='What model to use?')
    parser.add_argument('--dataset_name', type=str, default='nq', help='What dataset to train for?')
    parser.add_argument('--processing_type', type=str, default='multisample',
                        choices=['multisample', 'confidence_verb', 'confidence_numeric', 'absolute'],
                        help='Which alignment for honesty dataset construction method to follow')
    parser.add_argument('--absolute_threshold', type=float, default=0.1)

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
    prompt = "Answer the given question. If you are not confident that your answer will be correct, " +\
        "you should abstain from answering by using the phrase 'I am afraid I cannot help you as I do " +\
        f"not know the answer to this question.' Question: {question}"
    response = example['response']

    return {
        "messages" : [
            {"role" : "user", "content" : prompt},
            {"role" : "assistant", "content" : response},
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

def process_datapoints(args, raw_datapoints):
    if args.processing_type == "multisample":
        return multisample_datapoints(args, raw_datapoints)
    elif args.processing_type == "confidence_verb":
        return confidence_verb_datapoints(args, raw_datapoints)
    elif args.processing_type == "confidence_numeric":
        return confidence_numeric_datapoints(args, raw_datapoints) 
    elif args.processing_type == "absolute":
        return absolute_datapoints(args, raw_datapoints) 

def remove_answer_prefix(response):
    if response is not None and response.startswith("Answer:"):
        response = response[len("Answer:"):].strip()
    return response

def multisample_datapoints(args, raw_datapoints):
    new_datapoints = []
    unknown_response = "I am afraid I cannot help you as I do not know the answer to this question."
    for datapoint in raw_datapoints:
        for response, response_known in zip(
                datapoint[args.model_name]['sampling_pred_text'],
                datapoint[args.model_name]['sampling_labels']
        ):
            new_datapoint = {}
            new_datapoint['question'] = datapoint['question']
            output = remove_answer_prefix(response) if response_known == 'known' else unknown_response
            new_datapoint['response'] = output
            new_datapoints.append(new_datapoint)
    return new_datapoints

def pick_first_response(args, datapoint):
    response = None
    for sampling_pred_text, sampling_label in zip(
            datapoint[args.model_name]['sampling_pred_text'],
            datapoint[args.model_name]['sampling_labels']
    ):
        if sampling_label == 'known':
            response = sampling_pred_text
            break
    return response

def confidence_verb_datapoints(args, raw_datapoints):
    templated_responses = [
        "I am afraid I cannot help you as I do not know the answer to this question.",
        "I'm really not sure about this, but here is my answer: ",
        "I'm not completely sure about this, but here is my answer: ",
        "I don't have strong feelings either way, but here is my answer: ",
        "I'm fairly confident about this: ",
        "I'm absolutely certain about this: ",
    ]

    new_datapoints = []
    for datapoint in raw_datapoints:
        new_datapoint = {'question' : datapoint['question']}

        # First determine the response
        response = pick_first_response(args, datapoint)
        response = remove_answer_prefix(response)
        if datapoint[args.model_name]['sampling_knowns'] == 0:
            output = templated_responses[0]
        elif datapoint[args.model_name]['sampling_knowns'] <= 2:
            output = templated_responses[1] + response
        elif datapoint[args.model_name]['sampling_knowns'] <= 4:
            output = templated_responses[2] + response
        elif datapoint[args.model_name]['sampling_knowns'] <= 6:
            output = templated_responses[3] + response
        elif datapoint[args.model_name]['sampling_knowns'] <= 8:
            output = templated_responses[4] + response
        else:
            output = templated_responses[5] + response

        new_datapoint['response'] = output
        new_datapoints.append(new_datapoint)

    return new_datapoints

def confidence_numeric_datapoints(args, raw_datapoints):
    new_datapoints = []
    for datapoint in raw_datapoints:
        new_datapoint = {'question' : datapoint['question']}
        certainty = datapoint[args.model_name]['sampling_knowns'] * 10

        # First determine the response
        response = pick_first_response(args, datapoint)
        response = remove_answer_prefix(response)
        if certainty == 0:
            output = "I am afraid I cannot help you as I do not know the answer to this question."
        elif certainty < 50:
            output = f"I am only about {certainty}% confident to answer the question correctly, " +\
                "but based on my understanding and knowledge, here’s what I think is correct: " + response
        else:
            output = f"I am about {certainty}% confident to answer the question correctly, " +\
                "and based on my understanding and knowledge, here’s what I think is correct: " + response

        new_datapoint['response'] = output
        new_datapoints.append(new_datapoint)

    return new_datapoints

def absolute_datapoints(args, raw_datapoints):
    new_datapoints = []
    for datapoint in raw_datapoints:
        new_datapoint = {'question' : datapoint['question']}
        average_acc = datapoint[args.model_name]['sampling_knowns'] / 10

        # First determine the response
        response = pick_first_response(args, datapoint)
        response = remove_answer_prefix(response)
        if average_acc < args.absolute_threshold:
            output = "I am afraid I cannot help you as I do not know the answer to this question."
        else:
            output = response

        new_datapoint['response'] = output
        new_datapoints.append(new_datapoint)

    return new_datapoints

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
