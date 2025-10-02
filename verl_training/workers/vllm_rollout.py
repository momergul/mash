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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import vllm_version

if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
    from verl.third_party.vllm import LLM
    from verl.third_party.vllm import parallel_state as vllm_ps
else:
    from vllm.distributed import parallel_state as vllm_ps
    from vllm import LLM
from vllm import SamplingParams
from torch.distributed import get_rank, broadcast_object_list

from verl.utils.debug import log_gpu_memory_usage
import requests

DB_FOLDER = 'knowledge_sources'
RETRIEVAL_CACHE_FOLDER =  'retrieval_cache'
import re
import random
from verl_training.rule_based_rewards.reward_functions import extract_answer, em_check
import pickle

# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        # vLLM side
        torch.cuda.empty_cache()
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine = LLM(
                actor_module,
                tokenizer=tokenizer,
                model_hf_config=model_hf_config,
                tensor_parallel_size=tensor_parallel_size,
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                skip_tokenizer_init=False,
                max_model_len=config.prompt_length + config.response_length,
                load_format=config.load_format,
                disable_log_stats=config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=config.enable_chunked_prefill,
            )

            # Offload vllm model to reduce peak memory usage
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine = LLM(
                model=model_path,
                tokenizer=model_path,
                enable_sleep_mode=True,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="external_launcher",
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                disable_mm_preprocessor_cache=False,
                skip_tokenizer_init=False,
                max_model_len=config.prompt_length + config.response_length,
                max_num_batched_tokens=max_num_batched_tokens,
                disable_log_stats=config.disable_log_stats,
                enable_chunked_prefill=config.enable_chunked_prefill,
                enable_prefix_caching=True,
                trust_remote_code=kwargs.get('trust_remote_code', False)
            )

            # Offload vllm model to reduce peak memory usage
            self.inference_engine.sleep(level=1)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        self.is_agentic = config.is_agentic
        if self.is_agentic:
            self.initialize_agentic_rollout(config)
        else:
            self.initialize_regular_rollout(config)

    def initialize_agentic_rollout(self, config):
        self.course_correction = config.course_correction
        self.oracle_answer = config.retrieval_kwargs.shortform_retrieval.oracle_answer

        if self.oracle_answer:
            stop_tokens = ["</help>", "</answer>"]
        else:
            stop_tokens = ["</search>", "</answer>"]

        # Define train sampling arguments
        kwargs = dict(
            n=1,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.action_response_length,
            stop=stop_tokens, include_stop_str_in_output=True,
            logprobs=0,
        )
        self.sampling_params = SamplingParams(**kwargs)
        self.training_n = config.n

        # Define validation sampling arguments
        val_kwargs = dict(
            n=1,
            temperature=config.val_kwargs.temperature,
            top_p=config.val_kwargs.top_p,
            max_tokens=config.action_response_length,
            stop=stop_tokens, include_stop_str_in_output=True,
            logprobs=0,
        )
        self.val_sampling_params = SamplingParams(**val_kwargs)
        self.val_n = config.val_kwargs.n

        test_kwargs = dict(
            n=1,
            temperature=config.test_kwargs.temperature,
            top_p=config.test_kwargs.top_p,
            max_tokens=config.action_response_length,
            stop=stop_tokens, include_stop_str_in_output=True,
            logprobs=0,
        )
        self.test_sampling_params = SamplingParams(**test_kwargs)
        self.test_n = config.test_kwargs.n

        # Retrieval side
        self.max_searches = config.retrieval_kwargs.max_searches

    def initialize_regular_rollout(self, config):
        # Define arguments
        kwargs = dict(
            n=1,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.response_length,
            stop=["</answer>"], include_stop_str_in_output=True,
            logprobs=0,
        )
        self.sampling_params = SamplingParams(**kwargs)
        self.training_n = config.n

        # Define validation arguments
        val_kwargs = dict(
            n=1,
            temperature=config.val_kwargs.temperature,
            top_p=config.val_kwargs.top_p,
            max_tokens=config.response_length,
            stop=["</answer>"], include_stop_str_in_output=True,
            logprobs=0,
        )
        self.val_sampling_params = SamplingParams(**val_kwargs)
        self.val_n = config.val_kwargs.n

        test_kwargs = dict(
            n=1,
            temperature=config.test_kwargs.temperature,
            top_p=config.test_kwargs.top_p,
            max_tokens=config.response_length,
            stop=["</answer>"], include_stop_str_in_output=True,
            logprobs=0,
        )
        self.test_sampling_params = SamplingParams(**test_kwargs)
        self.test_n = config.test_kwargs.n

    def construct_partial_inputs(self, prompt_ids, partial_completion):
        if len(partial_completion) == 0:
            return self.tokenizer.decode(prompt_ids)
        else:
            full_text = ""
            for tokens in partial_completion:
                full_text += self.tokenizer.decode(tokens)
            return full_text

    def construct_partial_tokens(self, prompt_ids, partial_completion):
        if len(partial_completion) == 0:
            return prompt_ids
        else:
            full_tokens = []
            for tokens in partial_completion:
                full_tokens += tokens
            return full_tokens

    @torch.no_grad()
    def generate_sequences(self, prompts, **kwargs) -> DataProto:
        if self.is_agentic:
            return self.agentic_generate_sequences(prompts, **kwargs)
        else:
            return self.regular_generate_sequences(prompts, **kwargs)

    def regular_generate_sequences(self, prompts, **kwargs):
        inference_split = prompts.meta_info.get("inference_split", "train")
        if inference_split == "train":
            sample_n = self.training_n
            sampling_params = self.sampling_params
        elif inference_split == "val":
            sample_n = self.val_n
            sampling_params = self.val_sampling_params
        else:
            sample_n = self.test_n
            sampling_params = self.test_sampling_params

        # Carry over from veRL
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        raw_prompt_ids = _repeat_interleave(prompts.non_tensor_batch["raw_prompt_ids"], sample_n)
        prompts.non_tensor_batch["raw_prompt_ids"] = raw_prompt_ids
        reward_model = _repeat_interleave(prompts.non_tensor_batch["reward_model"], sample_n)
        prompts.non_tensor_batch["reward_model"] = reward_model

        inputs = [self.tokenizer.decode(raw_prompt_ids[idx]) for idx in range(len(raw_prompt_ids))]
        outputs = self.inference_engine.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
        response = [output.outputs[0].token_ids for output in outputs]
        response = pad_2d_list_to_length(response, self.pad_token_id).to(idx.device)
        completions = np.array([output.outputs[0].text for output in outputs], dtype='object')
        prompts.non_tensor_batch["completions"] = completions

        idx = _repeat_interleave(idx, sample_n)
        attention_mask = _repeat_interleave(attention_mask, sample_n)
        position_ids = _repeat_interleave(position_ids, sample_n)
        batch_size = batch_size * sample_n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': full_attention_mask,
                'document_mask' : full_attention_mask,
                'position_ids': position_ids,
                'num_searches': torch.tensor([0.0 for _ in range(response.shape[0])], device=response.device),
                'truncated': torch.tensor([0.0 for _ in range(response.shape[0])], device=response.device)
            },
            batch_size=batch_size)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def agentic_generate_sequences(self, prompts, **kwargs):
        inference_split = prompts.meta_info.get("inference_split", "train")
        if inference_split == "train":
            sample_n = self.training_n
            sampling_params = self.sampling_params
        elif inference_split == "val":
            print("Inference over val")
            sample_n = self.val_n
            sampling_params = self.val_sampling_params
        else:
            print("Inference over test")
            sample_n = self.test_n
            sampling_params = self.test_sampling_params

        # Carry over from veRL
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        idx_to_partials, idx_to_masks, idx_to_num_searches, idx_to_truncation = self.agentic_loop(prompts, sampling_params, sample_n)
        response, document_mask = self.process_agentic_outputs(prompts, idx_to_partials, idx_to_masks, idx_to_truncation, idx.device)

        idx = _repeat_interleave(idx, sample_n)
        attention_mask = _repeat_interleave(attention_mask, sample_n)
        position_ids = _repeat_interleave(position_ids, sample_n)
        batch_size = batch_size * sample_n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Now construct the document mask
        attention_mask_with_documents = torch.cat([attention_mask, document_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': full_attention_mask,
                'document_mask': attention_mask_with_documents,
                'position_ids': position_ids,
                'num_searches': torch.tensor(idx_to_num_searches, device=response.device),
                'truncated' : torch.tensor(idx_to_truncation, device=response.device),
            },
            batch_size=batch_size)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def agentic_loop(self, prompts, sampling_params, sample_n):
        # Expansion for generation
        entities = None
        raw_prompt_ids = _repeat_interleave(prompts.non_tensor_batch["raw_prompt_ids"], sample_n)
        prompts.non_tensor_batch["raw_prompt_ids"] = raw_prompt_ids

        reward_model = _repeat_interleave(prompts.non_tensor_batch["reward_model"], sample_n)
        prompts.non_tensor_batch["reward_model"] = reward_model
        all_answers = [inner_dict['ground_truth']['target'] for inner_dict in reward_model.tolist()]


        num_outputs = len(raw_prompt_ids)
        idx_to_partials = [[] for _ in range(num_outputs)]
        idx_to_masks = [[] for _ in range(num_outputs)]
        idx_to_num_searches = [0.0 for _ in range(num_outputs)]
        idx_to_truncation = [0.0 for _ in range(num_outputs)]
        completed_indices = [False for _ in range(num_outputs)]
        search_pattern = r'<help>(.*?)</help>' if self.oracle_answer else r'<search>(.*?)</search>'
        answer_pattern = r'<answer>(.*?)</answer>'

        # Generation itself
        for attempt in range(self.max_searches + 2):
            tp_rank = vllm_ps.get_tensor_model_parallel_rank()
            if tp_rank == 0:
                curr_indices = [idx for idx, completed in enumerate(completed_indices) if not completed]
                inputs = [self.construct_partial_inputs(raw_prompt_ids[idx], idx_to_partials[idx]) for idx in curr_indices]
                broadcast_data = {'curr_indices' : curr_indices, 'inputs' : inputs}
            else:
                broadcast_data = None
            broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
            curr_indices, inputs = broadcast_data['curr_indices'], broadcast_data['inputs']

            if torch.distributed.is_initialized():
                print(f"Rank {torch.distributed.get_rank()}: Pre-generation barrier")
                torch.distributed.barrier()
                print(f"Rank {torch.distributed.get_rank()}: Post-generation barrier")

            outputs = self.inference_engine.generate(inputs, sampling_params=sampling_params, use_tqdm=True)

            # First pass: Process the generated text
            search_indices = []
            search_queries = []
            for curr_idx, output in zip(curr_indices, outputs):
                partial_tokens = idx_to_partials[curr_idx]
                partial_masks = idx_to_masks[curr_idx]

                # First append what we have already
                if len(partial_tokens) == 0:
                    partial_tokens.append(output.prompt_token_ids)
                output_text = output.outputs[0].text
                partial_tokens.append(output.outputs[0].token_ids)
                partial_masks.append([1] * len(output.outputs[0].token_ids))

                search_matches = re.findall(search_pattern, output_text, re.DOTALL)
                answer_matches = re.findall(answer_pattern, output_text, re.DOTALL)
                if len(search_matches) != 0:
                    idx_to_num_searches[curr_idx] += 1
                    
                    # Case 1: The model searches
                    if idx_to_num_searches[curr_idx] > self.max_searches:
                        # Add the failure message
                        if self.oracle_answer:
                            search_result = "\n<warning>HELP LIMIT REACHED</warning>\n"
                        else:
                            search_result = "\n<warning>SEARCH LIMIT REACHED</warning>\n"
                        doc_tokens = list(self.tokenizer(search_result, add_special_tokens=False)['input_ids'])
                        partial_tokens.append(doc_tokens)
                        partial_masks.append([0] * len(doc_tokens))

                        # Early truncation
                        if self.exceeds_vllm_length(partial_tokens):
                            completed_indices[curr_idx] = True
                            idx_to_truncation[curr_idx] = 1.0
                    else:
                        search_indices.append(curr_idx)
                        search_queries.append(search_matches[0])
                elif len(answer_matches) != 0:
                    completed_indices[curr_idx] = True
                elif not self.course_correction and partial_tokens[-1][-1] == self.tokenizer.eos_token_id:
                    completed_indices[curr_idx] = True
                elif not self.course_correction: 
                    end_search_tag = "</help>" if self.oracle_answer else "</search>"
                    incorrect_tag_closure = end_search_tag in output_text or "</answer>" in output_text
                    if not incorrect_tag_closure:
                        completed_indices[curr_idx] = True
                        idx_to_truncation[curr_idx] = 1.0
                else:
                    # Case 3
                    # Remove eos token 
                    if partial_tokens[-1][-1] == self.tokenizer.eos_token_id:
                        partial_tokens[-1] = partial_tokens[-1][:-1]
                        partial_masks[-1] = partial_masks[-1][:-1]

                    # Append course correction message
                    correction_message = "\nMy previous action is invalid. If I want to search, I should put the " +\
                        "query between <search> and </search>. If I want to give the final answer, I should put the " +\
                        "answer between <answer> and </answer>. Let me try again.\n"

                    correction_tokens = list(self.tokenizer(correction_message, add_special_tokens=False)['input_ids'])
                    partial_tokens.append(correction_tokens)
                    course_correction_mask_val = 0
                    partial_masks.append([course_correction_mask_val] * len(correction_tokens))

                    # Early truncation
                    if self.exceeds_vllm_length(partial_tokens):
                        completed_indices[curr_idx] = True
                        idx_to_truncation[curr_idx] = 1.0

            # Second pass: Add the retrieved documents
            if tp_rank == 0:
                if self.oracle_answer:
                    search_results = self.batched_oracle_answers(search_indices, all_answers)
                else:
                    search_results = self.batched_search(search_indices, search_queries, entities)
                broadcast_data = {
                    'search_results': search_results,
                }
            else:
                broadcast_data = None
            search_results = vllm_ps._TP.broadcast_object(broadcast_data, src=0)['search_results'] # broadcast tool call results across tp

            for curr_idx, search_result in zip(search_indices, search_results):
                # Add the search results
                partial_tokens = idx_to_partials[curr_idx]
                partial_masks = idx_to_masks[curr_idx]
                doc_tokens = list(self.tokenizer(search_result, add_special_tokens=False)['input_ids'])

                partial_tokens.append(doc_tokens)
                partial_masks.append([0] * len(doc_tokens))

                # Early truncation
                if self.exceeds_vllm_length(partial_tokens):
                    completed_indices[curr_idx] = True
                    idx_to_truncation[curr_idx] = 1.0

        return idx_to_partials, idx_to_masks, idx_to_num_searches, idx_to_truncation

    def exceeds_vllm_length(self, partial_tokens):
        full_text = ""
        for tokens in partial_tokens:
            full_text += self.tokenizer.decode(tokens)
        retokenized_tokens = list(self.tokenizer(full_text, add_special_tokens=False)['input_ids'])
        return len(retokenized_tokens) >= (self.config.prompt_length + self.config.response_length)

    def batched_oracle_answers(self, search_indices, all_answers):
        search_results = []
        for idx in search_indices:
            curr_answer = all_answers[idx][0]
            doc = f"\n<helper_answer>The answer to the original question is: {curr_answer}</helper_answer>\n"
            search_results.append(doc)
        return search_results

    def batched_search(self, search_indices, search_queries, entities):
        search_results = []

        unprocessed_results = self.server_search(query_list=search_queries)
        for result in unprocessed_results:
            doc = "\n"
            for doc_item in result:
                content = doc_item['document']['contents'].split("\n")
                title = content[0]
                text = "\n".join(content[1:])                    
                doc += f"<document>\n(Title: {title}) {text}\n</document>\n"
            search_results.append(doc)
        return search_results

    def server_search(self, query_list):
        payload = {
            "queries" : query_list,
            "topk" : self.config.retrieval_kwargs.shortform_retrieval.topk,
            "return_scores": True
        }

        results = requests.post(self.config.retrieval_kwargs.shortform_retrieval.url, json=payload).json()
        return results['result']

    def process_agentic_outputs(self, prompts, idx_to_partials, idx_to_masks, idx_to_truncation, device):
        # Get the vllm response and document mask tokens; pad them
        response = []
        document_mask = []
        decoded_responses = []
        for idx, (partial_tokens, partial_masks) in enumerate(zip(idx_to_partials, idx_to_masks)):
            curr_tokens = []
            curr_masks = []
            for tokens, masks in zip(partial_tokens[1:], partial_masks):
                curr_tokens += tokens
                curr_masks += masks

            num_tokens = len(curr_tokens)
            response.append(curr_tokens[:self.config.response_length])
            document_mask.append(curr_masks[:self.config.response_length])
            decoded_responses.append(self.tokenizer.decode(curr_tokens[:self.config.response_length]))
            if num_tokens > self.config.response_length:
                idx_to_truncation[idx] = 1
            
        response = pad_2d_list_to_length(response, self.pad_token_id).to(device)
        document_mask = pad_2d_list_to_length(document_mask, 0).to(device)
        prompts.non_tensor_batch["completions"] = np.array(decoded_responses, dtype='object')

        return response, document_mask

