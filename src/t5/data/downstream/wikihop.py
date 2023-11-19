# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from transformers import AutoTokenizer
import random

class WikiHopCollator:

    def __init__(
            self,
            tokenizer,
            max_encoder_length,
            max_decoder_length,
            shuffle_candidates = False,
            shuffle_supports = False,
        ):

        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 1e9
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.shuffle_candidates = shuffle_candidates
        self.shuffle_supports = shuffle_supports

        self.tokenizer.add_tokens(['[question]', '[candidate]', '[support]'])

    def process_one_instance(self, instance):

        if self.shuffle_candidates:
            random.shuffle(instance['candidates'])
        if self.shuffle_supports:
            random.shuffle(instance['supports'])

        query = '[question]' + instance['query'] + '[candidate]' + '[candidate]'.join(instance['candidates'])
        supports = '[support]' + '[support]'.join(instance['supports'])
        answer = instance['answer']

        encoder_input = self.tokenizer(
            query + "\n\n\n" + supports,
            padding = "max_length",
            max_length = self.max_encoder_length,
            truncation = True,
            return_tensors = "pt",
        )

        query_tokens = self.tokenizer(
            query,
            padding = "max_length",
            max_length = self.max_encoder_length,
            truncation = True,
            return_tensors = "pt",
        )

        decoder_output = self.tokenizer(
            answer,
            padding = "max_length",
            max_length = self.max_decoder_length,
            truncation = True,
            return_tensors = "pt",
        )
        labels = decoder_output.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        non_truncated_decoder_output = self.tokenizer(
            answer,
            padding = "max_length",
            max_length = self.max_decoder_length * 32,
            truncation = True,
            return_tensors = "pt",
        )
        non_truncated_labels = non_truncated_decoder_output.input_ids
        non_truncated_labels[non_truncated_labels == self.tokenizer.pad_token_id] = -100

        instance = {
            "input_ids":encoder_input.input_ids,
            "attention_mask":encoder_input.attention_mask,
            "labels":labels,
            "non_truncated_labels":non_truncated_labels,
            "encoder_input_important_mask":query_tokens.attention_mask,
        }

        return instance

    def __call__(self, instances):
        batch = {}
        for inst in instances:
            inst = self.process_one_instance(inst)
            for key in inst:
                if key not in batch:
                    batch[key] = []
                batch[key].append(inst[key])

        batch = {key:torch.cat(batch[key], dim = 0) for key in batch}

        return batch
