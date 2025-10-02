# File: rule_based_rewards.py
# ---------------------------
# File that contains rule-based rewards and helper functions for dealing with them

import re
import string
import random
from collections import Counter
import math

FORMAT_ERROR_TYPES = [
    "mismatched_think", "no_think_tag", "mismatched_search",
    "hallucinate_docs", "mismatched_answer", "no_answer_tag",
]

## SEARCH FUNCTIONS ##

def compute_correctness(reward_config, output, golden_answers):
    answer = extract_answer(output)
    if answer is None:
        return 0.0, 0.0

    em_score = em_check(answer, golden_answers)
    f1_score = f1_check(answer, golden_answers)
    thresholded_f1_score = f1_score if f1_score >= reward_config.f1_threshold else 0.0
    return em_score, thresholded_f1_score

def compute_format_reward(reward_config, output):
    format_scorer = get_format_scorer_fn(reward_config)
    format_score, format_errors = format_scorer(output, reward_config.topk)
    return format_score, format_errors

def get_minimum_tool_calls(num_searches, rewards):
    valid_searches = [searches for searches, reward in zip(num_searches, rewards) if reward > 0]
    return float("inf") if len(valid_searches) == 0 else min(valid_searches)

def otc_grpo_penalty(reward_config, searches, estimated_min):
    # First get f(m, n)
    if searches == 0 and estimated_min == 0:
        f_m_n = 0.0
    elif estimated_min == 0:
        f_m_n = searches
    else:
        f_m_n = (2 * searches * estimated_min) / (searches + estimated_min)

    # Return the penalty
    if f_m_n == 0:
        return 1.0
    elif estimated_min == 0:
        return math.cos( (searches * math.pi) / (2 * searches + reward_config.max_searches) )
    else:
        return math.sin( (f_m_n * math.pi) / (2 * estimated_min) )

def exponential_penalty(reward_config, searches, estimated_min):
    exponential_lambda = reward_config.exponential_lambda
    return exponential_lambda ** (searches - estimated_min)

def compute_linear_penalty(reward_config, searches):
    max_searches = reward_config.max_searches
    return max(1.0 - searches / (max_searches+1), 0.0)

def get_format_scorer_fn(reward_config):
    if reward_config.formatting_reward_type == "strict":
        return strict_format_score
    else:
        return relaxed_format_score

def strict_format_score(output, topk):
    error_types = []

    # Check for proper thinking tags
    start_thoughts = output.count("<think>")
    end_thoughts = output.count("</think>")
    if start_thoughts != end_thoughts:
        error_types.append("mismatched_think")
    elif start_thoughts == 0:
        error_types.append("no_think_tag")

    # Check for proper search tags
    start_searches = output.count("<search>")
    end_searches = output.count("</search>")
    if start_searches != end_searches:
        error_types.append("mismatched_search")

    # Check for proper document tags
    start_docs = output.count("<document>")
    end_docs = output.count("</document>")
    if start_docs != end_docs or (start_searches * topk) != start_docs:
        error_types.append("hallucinate_docs")

    # Did I output answer tags?
    start_answer = output.count("<answer>")
    end_answer = output.count("</answer>")
    if start_answer != end_answer:
        error_types.append("mismatched_answer")
    elif start_answer != 1:
        error_types.append("no_answer_tag")

    format_score = 1.0 if len(error_types) == 0 else 0.0
    return format_score, error_types

def relaxed_format_score(output, topk):
    error_types = []

    # Check for proper search tags
    start_searches = output.count("<search>")
    end_searches = output.count("</search>")
    if start_searches != end_searches:
        error_types.append("mismatched_search")

    # Did I output answer tags?
    start_answer = output.count("<answer>")
    end_answer = output.count("</answer>")
    if start_answer != end_answer:
        error_types.append("mismatched_answer")
    elif start_answer != 1:
        error_types.append("no_answer_tag")

    format_score = 1.0 if len(error_types) == 0 else 0.0
    return format_score, error_types

def extract_answer(output):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, output, re.DOTALL)
    matches = list(match)
    
    # If there is no match, return None
    if len(matches) == 0:
        return None

    return matches[-1].group(1).strip()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(answer, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(answer)
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if normalized_prediction == golden_answer:
            return 1.0
    return 0.0

def f1_check(answer, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    f1_scores = [f1_score(answer, golden_answer) for golden_answer in golden_answers]
    return max(f1_scores)

def f1_score(prediction, ground_truth):
    # Taken from HotPotQA
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = 0.0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    

## OVERALL CONSTRUCTOR ##

def get_reward_fn(reward_config):
    if reward_config.name == "search_em_and_format":
        return search_em_and_format_constructor(reward_config)
    else:
        assert False, "Rule-based reward not implemented"
