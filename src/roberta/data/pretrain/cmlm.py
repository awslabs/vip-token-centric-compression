# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
import random
import numpy as np
import torch
from .mlm import MLMCollator

def sample_gumbel(vals_to_add, eps = 1e-10):
    u = torch.rand_like(vals_to_add)
    gs = -torch.log(-torch.log(u + eps) + eps)
    return gs

class ClusterMLMCollator(MLMCollator):

    def __init__(
            self,
            tokenizer,
            num_masked_tokens,
            max_sequence_length,
            num_masked_clusters_range,
            masked_cluster_sigma_range,
        ):
        super().__init__(tokenizer, num_masked_tokens, max_sequence_length)
        self.num_masked_clusters_range = num_masked_clusters_range
        self.masked_cluster_sigma_range = masked_cluster_sigma_range

    def mlm_masking(self, sequence, special_tokens_mask):
        tokenizer = self.tokenizer
        assert len(sequence.shape) == 2
        batch_size, sequence_length = sequence.shape

        inputs = sequence.clone()
        labels = sequence.clone()
        batch_indices = torch.arange(batch_size)[:, None]

        masking_probs = self.sample_masking_probs(sequence)

        masked_token_noise = (masking_probs + 1e-6).log() + sample_gumbel(masking_probs) - 1000 * special_tokens_mask.float()
        masked_token_ranking = torch.argsort(masked_token_noise, descending = True, dim = -1)
        masked_token_mask = torch.zeros_like(masked_token_ranking)
        masked_token_mask[batch_indices, masked_token_ranking[:, :self.num_masked_tokens]] = 1

        masked_token_indices = torch.sort(masked_token_mask, descending = True, stable = True, dim = -1).indices
        masked_token_indices = masked_token_indices[:, :self.num_masked_tokens]

        masked_token_mask = masked_token_mask.bool()

        labels[~masked_token_mask] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_token_mask
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_token_mask & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype = torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels, masked_token_indices

    def sample_masking_probs(self, sequence):
        masking_probs = []
        for b in range(sequence.shape[0]):
            num_masked_clusters = random.randint(self.num_masked_clusters_range[0], self.num_masked_clusters_range[1])
            masked_cluster_sigma = random.uniform(self.masked_cluster_sigma_range[0], self.masked_cluster_sigma_range[1])
            sequence_length = sequence.shape[-1]
            clusters = torch.randint(sequence_length, size = (num_masked_clusters, ), dtype = torch.float)
            positions = torch.arange(sequence_length, dtype = torch.float)
            logits = - (positions[:, None] - clusters[None, :]) ** 2 / (2 * masked_cluster_sigma ** 2)
            masking_probs.append(torch.exp(logits).mean(dim = -1))
        return torch.stack(masking_probs, dim = 0)
