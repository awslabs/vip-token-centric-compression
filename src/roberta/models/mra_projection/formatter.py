# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import random
from .coarser import Coarser
from .finer import Finer

class Formatter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_important_tokens = config.num_important_tokens
        self.block_size = config.block_size
        self.include_positions_in_mixed_states = config.include_positions_in_mixed_states
        self.coarser = Coarser(config)
        self.finer = Finer(config)

    def extra_repr(self):
        options = [
            f"num_important_tokens={self.num_important_tokens}",
            f"block_size={self.block_size}",
        ]
        return ", ".join(options)

    def to_vcc_input(self, mixed_states):
        hidden_states = mixed_states["hidden_states"]
        mask = mixed_states["mask"]
        importance_mask = mixed_states["importance_mask"]
        positions = mixed_states["positions"]
        cached = {"mask":mask, "importance_mask":importance_mask, "positions":positions}

        hidden_states = hidden_states * mask[:, :, None]

        sorted_order = torch.sort(importance_mask, descending = True, stable = True).indices
        batch_indices = torch.arange(hidden_states.shape[0], device = hidden_states.device)[:, None]

        hidden_states = hidden_states[batch_indices, sorted_order, :]
        importance_mask = importance_mask[batch_indices, sorted_order]
        mask = mask[batch_indices, sorted_order]

        important_token_states = hidden_states[:, :self.num_important_tokens, :]
        important_token_mask = mask[:, :self.num_important_tokens]
        importance_mask = importance_mask[:, :self.num_important_tokens]

        fine_token_states = hidden_states[:, self.num_important_tokens:, :]
        fine_token_mask = mask[:, self.num_important_tokens:]

        assert fine_token_states.shape[1] % self.block_size == 0, f"length={fine_token_states.shape[1]}, block_size={self.block_size}"

        mixed_states = {
            "sorted_order":sorted_order,
            "important_token_states":important_token_states,
            "important_token_mask":important_token_mask,
            "importance_mask":importance_mask,
            "fine_token_states":fine_token_states,
            "fine_token_mask":fine_token_mask,
            "extra_cached":cached
        }
        if self.include_positions_in_mixed_states:
            positions = positions[batch_indices, sorted_order]
            important_token_positions = positions[:, :self.num_important_tokens]
            fine_token_positions = positions[:, self.num_important_tokens:]
            mixed_states["important_token_positions"] = important_token_positions
            mixed_states["fine_token_positions"] = fine_token_positions

        return mixed_states

    def from_vcc_input(self, mixed_states):
        mixed_states = self.finer(self.coarser(mixed_states))

        hidden_states = torch.cat([mixed_states["important_token_states"], mixed_states["fine_token_states"]], dim = 1)

        reverse_order = torch.argsort(mixed_states["sorted_order"], dim = -1)
        batch_indices = torch.arange(hidden_states.shape[0], device = hidden_states.device)[:, None]
        hidden_states = hidden_states[batch_indices, reverse_order, :]
        cached = mixed_states["extra_cached"]

        mixed_states = {
            "hidden_states":hidden_states,
            "mask":cached["mask"],
            "importance_mask":cached["importance_mask"],
            "positions":cached["positions"]
        }
        return mixed_states
