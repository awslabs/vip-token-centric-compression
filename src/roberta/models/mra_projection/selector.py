# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import random

class Selector(nn.Module):
    def __init__(self, config, attention):
        super().__init__()

        self.num_fine_blocks = config.num_fine_blocks
        self.hidden_size = config.hidden_size
        self.selector_type = config.selector_type

        self.explore_prob = config.explore_prob
        self.attention = attention

    def extra_repr(self):
        repr = [
            f"num_fine_blocks={self.num_fine_blocks}",
            f"explore_prob={self.explore_prob}"
        ]
        return ", ".join(repr)

    def forward(self, mixed_states):
        # The selector module accepts a mixed_states that contains
        # {mask, coarse_token_states, coarse_token_mask, difference_cache, cache_indice_table}
        # outputs a mixed_states that contains
        # {mask, coarse_token_states, coarse_token_mask, difference_cache, cache_indice_table} old
        # + {fine_block_indices, coarse_block_indices} new

        if self.selector_type == "attention_based_selector":

            important_token_states = mixed_states["important_token_states"]
            importance_mask = mixed_states["importance_mask"]
            coarse_token_states = mixed_states["coarse_token_states"]
            coarse_token_mask = mixed_states["coarse_token_mask"]

            batch_size, num_blocks = coarse_token_mask.shape
            assert self.num_fine_blocks <= num_blocks

            if "important_token_positions" in mixed_states and "coarse_token_positions" in mixed_states:
                important_token_positions = mixed_states["important_token_positions"]
                coarse_token_positions = mixed_states["coarse_token_positions"]
                probs = self.attention(important_token_states, important_token_positions, coarse_token_states, coarse_token_positions, coarse_token_mask)
            else:
                probs = self.attention(important_token_states, coarse_token_states, coarse_token_mask)

            probs = probs.mean(dim = 1) * importance_mask[:, :, None].to(probs.dtype)
            average_prob_logits = torch.log(probs.mean(dim = 1) + 1e-5)

            if self.training:
                block_indices_rand = torch.argsort(torch.rand_like(average_prob_logits), dim = 1, descending = True)
                block_indices_prob = torch.argsort(average_prob_logits, dim = 1, descending = True)
                rand_mask = (torch.rand(batch_size, device = block_indices_rand.device) < self.explore_prob).long()
                block_indices = block_indices_rand * rand_mask[:, None] + block_indices_prob * (1 - rand_mask[:, None])
            else:
                block_indices = torch.argsort(average_prob_logits, dim = 1, descending = True)

            batch_indices = torch.arange(batch_size, device = block_indices.device)[:, None]

            probs = F.softmax(average_prob_logits, dim = -1)
            probs = probs[batch_indices, block_indices]

            mixed_states["fine_block_indices"] = block_indices[:, :self.num_fine_blocks]
            mixed_states["coarse_block_indices"] = block_indices[:, self.num_fine_blocks:]

            fine_block_scores = probs[:, :self.num_fine_blocks]
            coarse_block_scores = 1 - probs[:, self.num_fine_blocks:]

            mixed_states["fine_block_scores"] = 1 + fine_block_scores - fine_block_scores.detach()
            mixed_states["coarse_block_scores"] = 1 + coarse_block_scores - coarse_block_scores.detach()

        elif self.selector_type == "random":

            coarse_token_states = mixed_states["coarse_token_states"]
            coarse_token_mask = mixed_states["coarse_token_mask"]

            batch_size, num_blocks = coarse_token_mask.shape
            scores = torch.rand(batch_size, num_blocks, device = coarse_token_states.device).float()
            scores = scores - 1000.0 * (1.0 - coarse_token_mask.float())

            block_indices = torch.argsort(scores, dim = 1, descending = True)

            mixed_states["fine_block_indices"] = block_indices[:, :self.num_fine_blocks]
            mixed_states["coarse_block_indices"] = block_indices[:, self.num_fine_blocks:]

            mixed_states["fine_block_scores"] = torch.ones_like(scores[:, :self.num_fine_blocks])
            mixed_states["coarse_block_scores"] = torch.ones_like(scores[:, self.num_fine_blocks:])

        else:
            raise Exception()

        return mixed_states
