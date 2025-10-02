## File: utils.py
# ---------------
# Utilities for training

from trl.trainer.utils import DataCollatorForCompletionOnlyLM
import torch
import numpy as np
from dataclasses import dataclass
import re


class DataCollatorForRetrievalInterleavedLM(DataCollatorForCompletionOnlyLM):

    def __init__(
            self,
            document_template,
            response_template,
            *args,
            **kwargs,
    ):
        super().__init__(response_template, *args, **kwargs)
        self.document_template = document_template
        self.use_string_templates = type(self.document_template[0]) == str

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        if self.use_string_templates:
            self.mask_with_strings(batch)
        else:
            self.mask_with_ids(batch)

        return batch

    def find_document_token_indices(self, text, start_tag, end_tag):
        # First, find all raw character positions
        char_positions = []
    
        # Use regex to find all non-overlapping pairs
        pattern = re.compile(f"{start_tag}(.*?){end_tag}", re.DOTALL)
        for match in pattern.finditer(text):
            start_char_idx = match.start()
            end_char_idx = match.end()
            char_positions.append((start_char_idx, end_char_idx))
    
        # Convert character positions to token positions
        token_positions = []
    
        for start_char, end_char in char_positions:
            # Encode the text up to the start tag
            tokens_before_start = self.tokenizer.encode(text[:start_char], add_special_tokens=False)
            start_token_idx = len(tokens_before_start)
            
            # Encode the text up to the end tag
            tokens_before_end = self.tokenizer.encode(text[:end_char], add_special_tokens=False)
            end_token_idx = len(tokens_before_end)
        
            token_positions.append((start_token_idx, end_token_idx))
    
        return token_positions

    def mask_with_strings(self, batch):
        document_start = self.document_template[0]
        document_end = self.document_template[1]
        bsz = batch['input_ids'].shape[0]

        for i in range(bsz):
            full_text = self.tokenizer.decode(batch['input_ids'][i])
            token_positions = self.find_document_token_indices(full_text, document_start, document_end)
            for start_idx, end_idx in token_positions:
                batch["labels"][i, start_idx:end_idx] = self.ignore_index

    def mask_with_ids(self, batch):
        document_start = self.document_template[0]
        document_end = self.document_template[1]
        bsz = batch['input_ids'].shape[0]

        for i in range(bsz):
            start_indices = self.get_indices(batch['labels'][i], document_start)
            end_indices = self.get_indices(batch['labels'][i], document_end)
            for start_idx, end_idx in zip(start_indices, end_indices):
                batch["labels"][i, start_idx:end_idx+len(document_end)] = self.ignore_index

    def get_indices(self, labels, pattern):
        indices = []

        for idx in np.where(labels == pattern[0])[0]:
            if pattern == labels[idx : idx + len(pattern)].tolist():
                indices.append(idx)

        return indices

