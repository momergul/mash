# File: reward_manager.py
# -----------------------
# Reward manager classes for rule-based reward processing

import torch
from verl import DataProto
from verl_training.rule_based_rewards.reward_functions import compute_correctness, compute_format_reward, \
    get_minimum_tool_calls, otc_grpo_penalty, compute_linear_penalty, exponential_penalty

class QARewardManager:

    def __init__(self, reward_config):
        self.reward_config = reward_config
        self.optimal_tool_call_tracker = {}

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores.to(token_level_scores.device)

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def __call__(self, data, global_step, group_size):
        # First compute correctness and format rewards
        em_scores, f1_scores, format_rewards, format_errors = self.compute_correctness_and_format_rewards(data)
        correctness_and_format = self.combine_correctness_and_format(em_scores, f1_scores, format_rewards, global_step)
        penalties, rewards = self.apply_penalties(data, correctness_and_format, group_size)

        token_level_scores = self._expand_to_token_level(data, torch.tensor(rewards))
        output = DataProto.from_dict(
            tensors={
                "token_level_scores" : token_level_scores,
                "rewards" : torch.tensor(rewards, device=token_level_scores.device),
                "format_rewards" : torch.tensor(format_rewards, device=token_level_scores.device),
                "em_scores" : torch.tensor(em_scores, device=token_level_scores.device),
                "f1_scores" : torch.tensor(f1_scores, device=token_level_scores.device),
            },
            non_tensors={
                'format_errors': format_errors
            }
        )

        output = output.to('cpu')
        return output

    def compute_correctness_and_format_rewards(self, data):
        em_scores = []
        f1_scores = []
        format_rewards = []
        all_format_errors = []

        for i in range(len(data)):
            data_item = data[i]
            completion = data_item.non_tensor_batch["completions"]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]['target']
            em_score, f1_score = compute_correctness(self.reward_config, completion, ground_truth) 
            format_reward, format_errors = compute_format_reward(self.reward_config, completion) 

            em_scores.append(em_score)
            f1_scores.append(f1_score)
            format_rewards.append(format_reward)
            all_format_errors.append(format_errors)

        return em_scores, f1_scores, format_rewards, all_format_errors

    def combine_correctness_and_format(self, em_scores, f1_scores, format_rewards, global_step):
        combined_scores = []
        for em_score, f1_score, format_reward in zip(em_scores, f1_scores, format_rewards):
            correctness_score = em_score if self.reward_config.use_em else f1_score
            if self.reward_config.format_combination_strategy == "multiply":
                combined = correctness_score * format_reward
                if combined == 0 and global_step <= self.reward_config.formatting_reward_cutoff and format_reward > 0:
                    combined = format_reward * self.reward_config.format_reward_coeff
            elif self.reward_config.format_combination_strategy == "add":
                combined = correctness_score + format_reward * self.reward_config.format_reward_coeff
            else:
                combined = correctness_score
            combined_scores.append(combined)
        return combined_scores

    def apply_penalties(self, data, correctness_and_format, group_size):
        if self.reward_config.retrieval_penalty_type == "none":
            return [1 for _ in correctness_and_format], correctness_and_format
        elif self.reward_config.retrieval_penalty_type == "linear":
            penalties = []
            rewards = []
            for data_item, unprocessed_reward in zip(data, correctness_and_format):
                completion = data_item.non_tensor_batch["completions"]
                num_searches = data_item.batch['num_searches'].item()
                penalty = 0 if unprocessed_reward == 0 else compute_linear_penalty(self.reward_config, num_searches) 
                
                penalties.append(penalty)
                rewards.append(unprocessed_reward * penalty)

            return penalties, rewards
        elif self.reward_config.retrieval_penalty_type in ["otc_grpo", "otc_grpo_strict", "exponential"]:
            penalties = [0 for _ in range(len(correctness_and_format))]
            rewards = [0 for _ in range(len(correctness_and_format))]

            # Get mapping from questions to indices
            all_questions = data.non_tensor_batch['question'].tolist()
            question_to_idx = {}
            for question_idx, question in enumerate(all_questions):
                if question not in question_to_idx:
                    question_to_idx[question] = []
                question_to_idx[question].append(question_idx)

            for question, question_indices in question_to_idx.items():
                self.group_penalty_loop(penalties, rewards, correctness_and_format, question_indices, data,
                              self.reward_config.retrieval_penalty_type)

            return penalties, rewards

    def group_penalty_loop(self, penalties, rewards, correctness_and_format, question_indices, data, group_penalty_type):
        # Get the min estimate
        curr_inputs = [data[idx] for idx in question_indices]
        num_searches = [data_item.batch['num_searches'].item() for data_item in curr_inputs]
        curr_unprocessed = [correctness_and_format[idx] for idx in question_indices]
        estimated_min = get_minimum_tool_calls(num_searches, curr_unprocessed)

        # Replace estimate with historic minimum if present
        dataset = curr_inputs[0].non_tensor_batch['extra_info']['data_source']
        question = curr_inputs[0].non_tensor_batch['question']
        if dataset not in self.optimal_tool_call_tracker:
            self.optimal_tool_call_tracker[dataset] = {}
        if question not in self.optimal_tool_call_tracker[dataset]:
            self.optimal_tool_call_tracker[dataset][question] = float("inf")

        # Update the min estimate
        estimated_min = min(estimated_min, self.optimal_tool_call_tracker[dataset][question])
        self.optimal_tool_call_tracker[dataset][question] = estimated_min

        for question_idx, searches, unprocessed_reward in zip(question_indices, num_searches, curr_unprocessed):
            if "otc" in group_penalty_type:
                penalty = 0 if unprocessed_reward == 0 else otc_grpo_penalty(self.reward_config, searches, estimated_min)
            else:
                penalty = 0 if unprocessed_reward == 0 else exponential_penalty(self.reward_config, searches, estimated_min) 
            penalties[question_idx] = penalty
            reward = unprocessed_reward * penalty
            if group_penalty_type in ["otc_grpo_strict"] and estimated_min == 0:
                reward = reward if searches == 0 else 0
            rewards[question_idx] = reward


        
                


