# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from tqdm import tqdm

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl_training.trainer import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl_training.data_utils.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import gc
from time import time
import pickle
import re
from collections import Counter

from verl.trainer.ppo.ray_trainer import Role, AdvantageEstimator, ResourcePoolManager, apply_kl_penalty, _timer, compute_response_mask
from verl_training.rule_based_rewards.reward_functions import FORMAT_ERROR_TYPES
import verl_training.trainer.core_algos as core_algos

WorkerType = Type[Worker]

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                      group_level_mean=True, group_level_std=True): 
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            eos_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            eos_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            group_level_mean=group_level_mean,
            group_level_std=group_level_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], eos_mask=data.batch['response_mask'], gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            eos_mask=data.batch['response_mask'])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            eos_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def compute_dedicated_metrics(batch):
    # Regular metrics
    rewards = batch.batch['token_level_rewards'].sum(-1)
    format_rewards = batch.batch['format_rewards']
    em_scores = batch.batch['em_scores']
    f1_scores = batch.batch['f1_scores']    
    num_searches = batch.batch['num_searches']
    truncated = batch.batch['truncated']
    output_lengths = [len(full_response.split(" ")) for full_response in batch.non_tensor_batch['completions']]
    distinct_search_rates = compute_distinct_search_rates(batch)
    tool_productivity = torch.mean(em_scores / (1 + num_searches)).item()

    metrics = {
        'my_metrics/reward' : torch.mean(rewards).detach().item(),
        'my_metrics/format_reward' : torch.mean(format_rewards).detach().item(),
        'my_metrics/em_scores' : torch.mean(em_scores).detach().item(),
        'my_metrics/f1_scores' : torch.mean(f1_scores).detach().item(),
        'my_metrics/num_searches' : torch.mean(num_searches).detach().item(),
        'my_metrics/truncation_rate' : torch.mean(truncated).detach().item(),
        'my_metrics/output_lengths' : np.mean(output_lengths),
        'my_metrics/distinct_search_rate' : np.mean(distinct_search_rates),
        'my_metrics/tool_productivity' : tool_productivity,
    }

    # Error type logging
    error_counter = {format_error : 0 for format_error in FORMAT_ERROR_TYPES}
    for format_errors in batch.non_tensor_batch['format_errors']:
        for format_error in format_errors:
            error_counter[format_error] += 1 / rewards.shape[0]
    for format_error, proportion in error_counter.items():
        metrics[f'finegrained_error_metrics/{format_error}'] = proportion

    return metrics

def compute_distinct_search_rates(batch):
    has_distinct_searches = [0 for _ in range(len(batch))]

    all_questions = batch.non_tensor_batch['question'].tolist()
    question_to_idx = {}
    for question_idx, question in enumerate(all_questions):
        if question not in question_to_idx:
            question_to_idx[question] = []
        question_to_idx[question].append(question_idx)

    questions = set(all_questions)
    for question in questions:
        indices = question_to_idx[question]
        num_searches = {batch[idx].batch['num_searches'].item() for idx in indices}
        has_distinct_search = 1 if len(num_searches) > 1 else 0

        for idx in indices:
            has_distinct_searches[idx] = has_distinct_search

    return has_distinct_searches

def update_val_metrics(metric_dict, batch):
    # Reward information
    rewards = batch.batch['token_level_scores'].sum(-1)
    format_rewards = batch.batch['format_rewards']
    em_scores = batch.batch['em_scores']
    f1_scores = batch.batch['f1_scores']    
    num_searches = batch.batch['num_searches']
    truncated = batch.batch['truncated']
    output_lengths = [len(full_response.split(" ")) for full_response in batch.non_tensor_batch['completions']]
    distinct_search_rates = compute_distinct_search_rates(batch)
    tool_productivity = em_scores / (1 + num_searches)
    
    curr_metric_dict = {
        'reward' : rewards.detach().tolist(),
        'format_reward' : format_rewards.detach().tolist(),
        'em_scores' : em_scores.detach().tolist(),
        'f1_scores' : f1_scores.detach().tolist(),
        'num_searches' : num_searches.detach().tolist(),
        'truncation_rate' : truncated.detach().tolist(),
        'output_lengths' : output_lengths,
        'distinct_search_rate' : distinct_search_rates,
        'tool_productivity' : tool_productivity.detach().tolist(),

        'format_errors' : [batch.non_tensor_batch['format_errors']],
        'question' : [batch.non_tensor_batch['question']],
        'responses' : [batch.non_tensor_batch['completions']],
        'golden_answers' : [batch.non_tensor_batch['golden_answers']],
        'data_sources' : [info['data_source'] for info in batch.non_tensor_batch['extra_info']],
        'reward_info' : [rm_dict for rm_dict in batch.non_tensor_batch['reward_model']]
    }

    for key, value in curr_metric_dict.items():
        if key not in metric_dict:
            metric_dict[key] = value
        else:
            metric_dict[key] = metric_dict[key] + value

def process_val_metrics(metric_dict):
    # First fill up a metrics dict with separate metrics for each data source
    data_sources = list(set(metric_dict['data_sources']))
    reward_keys = ['reward', 'format_reward', 'em_scores', 'f1_scores',
                   'num_searches', 'output_lengths', 'tool_productivity', 'truncation_rate', 'distinct_search_rate']

    collected_format_errors = []
    for format_errors in metric_dict['format_errors']:
        collected_format_errors += format_errors.tolist()

    metrics = {}
    for data_source in data_sources:
        # Regular metrics
        ds_indices = [idx for idx, ds in enumerate(metric_dict['data_sources']) if ds == data_source]
        for key in reward_keys:
            values = [metric_dict[key][idx] for idx in ds_indices]
            metrics[f'my_val_metrics_{data_source}/{key}'] = np.mean(values) if len(values) > 0 else 0

        # Error mode metrics
        error_counter = {format_error : 0 for format_error in FORMAT_ERROR_TYPES}
        for idx in ds_indices:
            format_errors = collected_format_errors[idx]
            for format_error in format_errors:
                error_counter[format_error] += 1 / len(ds_indices)
        for format_error, proportion in error_counter.items():
            metrics[f'finegrained_error_metrics_val_{data_source}/{format_error}'] = proportion

    return metrics, metric_dict

def extract_summary_metric(summary_metrics, save_metrics):
    metric_dict = {}
    for save_metric in save_metrics:
        relevant_values = []
        for key, val in summary_metrics.items():
            if key.split("/")[-1] == save_metric:
                relevant_values.append(val)
        if len(relevant_values) > 0:
            metric_dict[save_metric] = sum(relevant_values) / len(relevant_values)
    return metric_dict

class AgenticGRPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.use_peft = config.actor_rollout_ref.peft.use_peft
        if self.use_peft:
            assert Role.ActorRolloutRef in role_worker_mapping, f'{role_worker_mapping.keys()=}'
        else:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'            

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping
        self.use_rm = config.reward_model.use_neural_reward
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger() # TODO (can remove calls)

        # define KL control
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self.use_critic = False

        self._validate_config() 
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        drop_last_for_train = True

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         prompt_type=self.config.data.shortform_prompt_type,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        train_batch_size = self.config.data.train_batch_size
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=train_batch_size,
                                                   num_workers=8,
                                                   drop_last=drop_last_for_train,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        # Create val set
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       prompt_type=self.config.data.shortform_prompt_type,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        # Create test set
        self.test_dataset = RLHFDataset(parquet_files=self.config.data.test_files,
                                        prompt_type=self.config.data.shortform_prompt_type,
                                        tokenizer=self.tokenizer,
                                        processor=self.processor,
                                        prompt_key=self.config.data.prompt_key,
                                        image_key=self.config.data.get('image_key', 'images'),
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation=self.config.data.get('truncation', 'error'),
                                        filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.test_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.test_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.test_dataloader = StatefulDataLoader(
            dataset=self.test_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.test_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        #assert len(
        #    self.val_dataloader
        #) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self, is_test=False, dataloader=None):
        metrics = {}
        dataloader = self.val_dataloader if dataloader is None else dataloader
        inference_split = "test" if is_test else "val"
        batch_n = self.config.actor_rollout_ref.rollout.test_kwargs.n if is_test else self.config.actor_rollout_ref.rollout.val_kwargs.n

        for test_data in dataloader:
            batch = DataProto.from_single_dict(test_data)
            batch.meta_info['batch_n'] = batch_n

            # pop those keys for generation
            gen_batch = batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=['raw_prompt_ids', 'reward_model'],
            )
            gen_batch.meta_info = {"inference_split" : inference_split}

            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                     dtype=object)
            batch = batch.repeat(repeat_times=batch_n, interleave=True)
            batch = batch.union(gen_batch_output)

            batch.batch['response_mask'] = compute_response_mask(batch)
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)
            else:
                reward_tensor = self.val_reward_fn(batch, self.global_steps, batch_n)
                batch = batch.union(reward_tensor)

            update_val_metrics(metrics, batch)

        summary_metrics, full_dict = process_val_metrics(metrics) 
        summary_metric = extract_summary_metric(summary_metrics, self.config.trainer.save_metrics) 

        return summary_metrics, full_dict, summary_metric

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        rollout_role = Role.ActorRolloutRef if self.use_peft else Role.ActorRollout
        rollout_role_string = 'actor_rollout_ref' if self.use_peft else 'actor_rollout'
        resource_pool = self.resource_pool_manager.get_resource_pool(rollout_role)
        actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[rollout_role],
                                                 config=self.config.actor_rollout_ref,
                                                 role=rollout_role_string)
        self.resource_pool_to_cls[resource_pool][rollout_role_string] = actor_rollout_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and not self.use_peft:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model.neural_reward)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        #if self.use_reference_policy and not self.use_peft:
        #    self.ref_policy_wg = all_wg['ref']
        #    self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[rollout_role_string]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self, summary_metric_dict):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

        # update best checkpoint if necessary
        self._update_best_metric_dict(summary_metric_dict) 
        if self.config.trainer.only_keep_best_and_recent:
            self._clean_unnecessary_checkpoints()
        if self.config.trainer.delete_non_params:
            self._delete_non_params()

    def _update_best_metric_dict(self, summary_metric_dict):
        # Load the old metric dict
        best_file_path = os.path.join(self.config.trainer.default_local_dir, 'best_metrics.pkl')
        if os.path.exists(best_file_path):
            with open(best_file_path, 'rb') as f:
                best_summary_metric_dict = pickle.load(f)
        else:
            best_summary_metric_dict = {
                key : {'value' : -float("inf"), "step" : 0} for key in summary_metric_dict
            }

        # Update the values
        has_improvement = False
        for metric, value in summary_metric_dict.items():
            if value > best_summary_metric_dict[metric]['value']:
                best_summary_metric_dict[metric]['value'] = value
                best_summary_metric_dict[metric]['step'] = self.global_steps
                has_improvement = True

        # Update the patience criterion
        if has_improvement:
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1       
        best_summary_metric_dict['epochs_since_improvement'] = self.epochs_since_improvement

        # Save the dictionary
        with open(best_file_path, 'wb') as f:
            pickle.dump(best_summary_metric_dict, f)

    def _clean_unnecessary_checkpoints(self):
        import shutil
        valid_checkpoints = {self.global_steps}

        best_metrics_path = os.path.join(self.config.trainer.default_local_dir, 'best_metrics.pkl')
        with open(best_metrics_path, 'rb') as f:
            best_metrics = pickle.load(f)
            for metric, metric_dict in best_metrics.items():
                if metric == "epochs_since_improvement":
                    continue
                valid_checkpoints.add(metric_dict["step"])

        # Next iterate over all relevant folders
        folders = [f for f in os.listdir(self.config.trainer.default_local_dir) if f.startswith('global_step')]
        for folder in folders:
            folder_step = int(folder.split("_")[-1])

            actor_path = os.path.join(self.config.trainer.default_local_dir, folder, 'actor')
            if folder_step not in valid_checkpoints and os.path.exists(actor_path):
                shutil.rmtree(actor_path)

            critic_path = os.path.join(self.config.trainer.default_local_dir, folder, 'critic')
            if folder_step not in valid_checkpoints and os.path.exists(critic_path):
                shutil.rmtree(critic_path)

    def _delete_non_params(self):
        import shutil        

        # Iterate over each folder; if there are model parameters, delete the optimizer and scheduler
        folders = [f for f in os.listdir(self.config.trainer.default_local_dir) if f.startswith('global_step')]
        for folder in folders:
            folder_step = int(folder.split("_")[-1])
            if folder_step == self.global_steps:
                continue

            # Delete non-parameter checkpoints from the actor
            actor_path = os.path.join(self.config.trainer.default_local_dir, folder, 'actor')
            if os.path.exists(actor_path):
                non_param_files = [filename for filename in os.listdir(actor_path) if "model_world_size" not in filename]
                for non_param_file in non_param_files:
                    os.remove(os.path.join(actor_path, non_param_file))
               
            # Delete non-parameter checkpoints from the critic
            critic_path = os.path.join(self.config.trainer.default_local_dir, folder, 'critic')
            if os.path.exists(critic_path):
                non_param_files = [filename for filename in os.listdir(critic_path) if "model_world_size" not in filename]
                for non_param_file in non_param_files:
                    os.remove(os.path.join(critic_path, non_param_file))

    def _save_outputs(self, batch):
        local_global_step_folder = os.path.join(self.config.trainer.default_intermediate_output_dir,
                                                f'global_step_{self.global_steps}')
        if not os.path.exists(local_global_step_folder):
            os.makedirs(local_global_step_folder)

        rewards = batch.batch['token_level_scores'].sum(-1)
        advantages = batch.batch['advantages'][:, 0]
        format_rewards = batch.batch['format_rewards']
        em_scores = batch.batch['em_scores']
        f1_scores = batch.batch['f1_scores']    
        num_searches = batch.batch['num_searches']
        truncated = batch.batch['truncated']
        output_lengths = [len(full_response.split(" ")) for full_response in batch.non_tensor_batch['completions']]
        distinct_search_rates = compute_distinct_search_rates(batch)
        tool_productivity = em_scores / (1 + num_searches)

        # Save input/outputs for timestep
        outputs = {
            'reward' : rewards.detach().tolist(),
            'advantages' : advantages.detach().tolist(),
            'format_reward' : format_rewards.detach().tolist(),
            'em_scores' : em_scores.detach().tolist(),
            'f1_scores' : f1_scores.detach().tolist(),
            'num_searches' : num_searches.detach().tolist(),
            'truncation_rate' : truncated.detach().tolist(),
            'output_lengths' : output_lengths,
            'distinct_search_rate' : distinct_search_rates,
            'tool_productivity' : tool_productivity.detach().tolist(),

            'format_errors' : [batch.non_tensor_batch['format_errors']],
            'question' : [batch.non_tensor_batch['question']],
            'responses' : [batch.non_tensor_batch['completions']],
            'golden_answers' : [batch.non_tensor_batch['golden_answers']],
            'data_sources' : [info['data_source'] for info in batch.non_tensor_batch['extra_info']],
            'reward_info' : [rm_dict for rm_dict in batch.non_tensor_batch['reward_model']]
        }

        output_path = os.path.join(local_global_step_folder, 'step_outputs.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(outputs, f)

    def _save_val_outputs(self, val_metrics, full_dict):
        local_global_step_folder = os.path.join(self.config.trainer.default_intermediate_output_dir,
                                                f'global_step_{self.global_steps}')
        if not os.path.exists(local_global_step_folder):
            os.makedirs(local_global_step_folder)

        output_path = os.path.join(local_global_step_folder, 'step_val_outputs.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump([val_metrics, full_dict], f)

    def _load_checkpoint(self, global_step_folder=None, only_load_params=False, load_data_state_dict=True):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        checkpoint_folder = self.config.trainer.default_local_dir 
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        if global_step_folder is None:
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        # load best metrics for patience criterion
        best_metrics_path = os.path.join(checkpoint_folder, 'best_metrics.pkl')
        if os.path.exists(best_metrics_path) and os.path.isfile(best_metrics_path):
            with open(best_metrics_path, 'rb') as f:
                best_metrics = pickle.load(f)
                self.epochs_since_improvement = best_metrics['epochs_since_improvement']

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                                              only_load_params=only_load_params)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                                           only_load_params=only_load_params)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path) and load_data_state_dict:
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def get_checkpoints(self):
        checkpoint_folder = self.config.trainer.default_local_dir 
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

        checkpoints = []

        candidates = [folder for folder in os.listdir(checkpoint_folder) if folder.startswith("global_step")]
        for candidate in candidates:
            curr_folder = os.path.join(checkpoint_folder, candidate)
            if os.path.exists(os.path.join(curr_folder, "actor")):
                checkpoints.append(curr_folder)

        return checkpoints

    def pure_test(self, suffix, dataloader):
        """
        Evaluate model on the test set
        """
        # Get the relevant checkpoint
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

        # Determine the step
        if self.config.trainer.test_checkpoint_step is None:
            # Load best checkpoint
            best_metrics_path = os.path.join(checkpoint_folder, 'best_metrics.pkl')
            load_metric = self.config.trainer.test_load_metric
            with open(best_metrics_path, 'rb') as f:
                best_metrics = pickle.load(f)
            checkpoint_step = best_metrics[load_metric]['step']
        else:
            checkpoint_step = self.config.trainer.test_checkpoint_step

        checkpoint = os.path.join(checkpoint_folder, f'global_step_{checkpoint_step}')

        # Load the model from the checkpoint
        if checkpoint_step != 0:
            self._load_checkpoint(checkpoint, only_load_params=True, load_data_state_dict=False)
        else:
            self.global_steps = 0
            
        # Validate
        summary_metrics, full_dict, summary_metric = self._validate(is_test=True, dataloader=dataloader)

        # Save to the checkpoint
        if checkpoint_step != 0:
            save_path = os.path.join(
                self.config.trainer.default_intermediate_output_dir,
                f'global_step_{checkpoint_step}',
                f'step_{suffix}.pkl'
            )
        else:
            parent_folder = os.path.join(
                self.config.trainer.default_intermediate_output_dir,
                'initial_checkpoint_evals'
            )
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            save_path = os.path.join(parent_folder, f'step_{suffix}.pkl')
            
        with open(save_path, 'wb') as f:
            pickle.dump([summary_metrics, full_dict], f)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        self.epochs_since_improvement = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                start_time = time()

                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info['batch_n'] = self.config.actor_rollout_ref.rollout.n 

                # pop those keys for generation
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'reward_model'],
                )
                gen_batch.meta_info = {"inference_split" : "train"}

                if self.config.trainer.early_stop != -1:
                    is_last_step = self.global_steps >= self.config.trainer.early_stop
                else:
                    is_last_step = self.global_steps >= self.total_training_steps                    

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    batch.batch['response_mask'] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics) 

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    #if self.use_peft:
                    #    # compute reference log_prob
                    #    with _timer('ref', timing_raw):
                    #        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch) # TODO: PEFT
                    #        batch = batch.union(ref_log_prob)
                    #else:
                    #    # compute reference log_prob
                    #    with _timer('ref', timing_raw):
                    #        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    #        batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.

                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        else:
                            reward_tensor = self.reward_fn(batch, self.global_steps, self.config.actor_rollout_ref.rollout.n)
                            batch = batch.union(reward_tensor)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  group_level_mean=self.config.algorithm.group_level_mean,
                                                  group_level_std=self.config.algorithm.group_level_std)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.save_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics, full_dict, summary_metric = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)


                        # Save the model outputs for validation
                        self._save_val_outputs(val_metrics, full_dict) 

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint(summary_metric)
                            if self.config.trainer.patience_epochs != -1:
                                is_last_step = self.epochs_since_improvement >= self.config.trainer.patience_epochs

                    if self.config.trainer.output_save_freq > 0 and ( is_last_step or \
                                                                      self.global_steps <= self.config.trainer.output_save_freq or \
                                                                      self.global_steps % self.config.trainer.output_save_freq == 0):
                        self._save_outputs(batch) 

                    print(f"Step took {time() - start_time}")

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_dedicated_metrics(batch=batch))

                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1

                if self.config.trainer.early_stop != -1 and self.global_steps > self.config.trainer.early_stop:
                    print("Stopping training due to early stopping")
                    return
