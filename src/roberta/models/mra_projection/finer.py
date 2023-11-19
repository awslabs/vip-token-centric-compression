# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import random

def cache_select(cache, select_indices):
    batch_size, select_size = select_indices.shape
    if cache["version"] == 0:
        batch_indices = torch.arange(batch_size, device = select_indices.device)[:, None]
        new_indices = cache["indice_table"][batch_indices, select_indices]
        select = cache["bank"][batch_indices, new_indices, :, :]
    else:
        raise Exception()
    return select

class Finer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

    def extra_repr(self):
        return f"block_size={self.block_size}"

    def forward(self, mixed_states):
        # The finer module accepts a mixed_states that contains
        # {mask, coarse_token_states, coarse_token_mask, difference_cache, cache_indice_table, fine_block_indices, coarse_block_indices}
        # in layers except for the last layer
        # outputs a mixed_states that contains
        # {mask, difference_cache, cache_indice_table, fine_block_indices, coarse_block_indices} old
        # + {partial_fine_token_states, partial_fine_token_mask, partial_coarse_token_states, partial_coarse_token_mask} new
        # In the last layer, the coarser module accepts a mixed_states that contains
        # {mask, coarse_token_states, coarse_token_mask, difference_cache, cache_indice_table}
        # outputs a mixed_states that contains
        # {mask, fine_token_states, fine_token_mask}

        if "fine_block_indices" in mixed_states and "coarse_block_indices" in mixed_states:

            fine_token_mask = mixed_states["fine_token_mask"]
            coarse_token_states = mixed_states["coarse_token_states"]
            coarse_token_mask = mixed_states["coarse_token_mask"]
            del mixed_states["coarse_token_states"]

            fine_block_indices = mixed_states["fine_block_indices"]
            fine_block_scores = mixed_states["fine_block_scores"]
            coarse_block_indices = mixed_states["coarse_block_indices"]
            coarse_block_scores = mixed_states["coarse_block_scores"]
            del mixed_states["fine_block_scores"]
            del mixed_states["coarse_block_scores"]

            batch_size, num_blocks, dim = coarse_token_states.shape
            num_fine_blocks = fine_block_indices.shape[1]
            batch_indices = torch.arange(batch_size, device = fine_block_indices.device)[:, None]

            partial_coarse_token_scores = coarse_block_scores
            partial_coarse_token_states = coarse_token_states[batch_indices, coarse_block_indices, :]
            partial_coarse_token_mask = coarse_token_mask[batch_indices, coarse_block_indices]

            partial_fine_token_scores = fine_block_scores[:, :, None].repeat(1, 1, self.block_size)
            partial_fine_token_scores = partial_fine_token_scores.reshape(batch_size, num_fine_blocks * self.block_size)

            difference = cache_select(mixed_states["difference_cache"], fine_block_indices)
            to_fine_token_states = coarse_token_states[batch_indices, fine_block_indices, :]
            partial_fine_token_states = to_fine_token_states[:, :, None, :] - difference
            partial_fine_token_states = partial_fine_token_states.reshape(batch_size, num_fine_blocks * self.block_size, dim)

            fine_token_mask = fine_token_mask.reshape(batch_size, num_blocks, self.block_size)
            partial_fine_token_mask = fine_token_mask[batch_indices, fine_block_indices, :]
            partial_fine_token_mask = partial_fine_token_mask.reshape(batch_size, num_fine_blocks * self.block_size)

            mixed_states["partial_fine_token_states"] = partial_fine_token_states
            mixed_states["partial_fine_token_mask"] = partial_fine_token_mask
            mixed_states["partial_fine_token_scores"] = partial_fine_token_scores
            mixed_states["partial_fine_token_num_dups"] = torch.ones_like(partial_fine_token_scores)
            mixed_states["partial_coarse_token_states"] = partial_coarse_token_states
            mixed_states["partial_coarse_token_mask"] = partial_coarse_token_mask
            mixed_states["partial_coarse_token_scores"] = partial_coarse_token_scores
            mixed_states["partial_coarse_token_num_dups"] = self.block_size * torch.ones_like(partial_coarse_token_scores)

            if "coarse_token_positions" in mixed_states and "fine_token_positions" in mixed_states:
                coarse_token_positions = mixed_states["coarse_token_positions"]
                fine_token_positions = mixed_states["fine_token_positions"]
                partial_coarse_token_positions = coarse_token_positions[batch_indices, coarse_block_indices]
                partial_fine_token_positions = fine_token_positions.reshape(batch_size, num_blocks, self.block_size)[batch_indices, fine_block_indices, :]
                partial_fine_token_positions = partial_fine_token_positions.reshape(batch_size, num_fine_blocks * self.block_size)
                mixed_states["partial_fine_token_positions"] = partial_fine_token_positions
                mixed_states["partial_coarse_token_positions"] = partial_coarse_token_positions

        elif "coarse_token_states" in mixed_states and "coarse_token_mask" in mixed_states:

            coarse_token_states = mixed_states["coarse_token_states"]
            coarse_token_mask = mixed_states["coarse_token_mask"]
            del mixed_states["coarse_token_states"]
            del mixed_states["coarse_token_mask"]

            batch_size, num_blocks, dim = coarse_token_states.shape

            fine_block_indices = torch.arange(num_blocks, device = coarse_token_states.device)[None, :].repeat(batch_size, 1)
            difference = cache_select(mixed_states["difference_cache"], fine_block_indices)
            del mixed_states["difference_cache"]

            fine_token_states = coarse_token_states[:, :, None, :] - difference
            fine_token_states = fine_token_states.reshape(batch_size, num_blocks * self.block_size, dim)

            mixed_states["fine_token_states"] = fine_token_states * mixed_states["fine_token_mask"][:, :, None]
        else:
            raise Exception()

        return mixed_states
