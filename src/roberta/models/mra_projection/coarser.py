# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import random

def cache_update(cache, update, update_indices):
    if cache["version"] == 0:
        if "bank" not in cache:
            cache["indice_table"] = update_indices
            cache["bank"] = update
        else:
            batch_size, update_size, _, _ = update.shape
            cache_offset = cache["bank"].shape[1]
            batch_indices = torch.arange(batch_size, device = update_indices.device)[:, None]
            new_indices = torch.arange(cache_offset, cache_offset + update_size, device = cache["bank"].device)
            cache["indice_table"][batch_indices, update_indices] = new_indices[None, :]
            cache["bank"] = torch.cat([cache["bank"], update], dim = 1)
    else:
        raise Exception()

class Coarser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.coarser_type = config.coarser_type
        self.cache_version = config.cache_version if hasattr(config, "cache_version") else 0

    def extra_repr(self):
        return f"block_size={self.block_size}, coarser_type={self.coarser_type}, cache_version={self.cache_version}"

    def to_coarse(self, fine_token_states, fine_token_mask):
        batch_size, fine_token_states_size, dim = fine_token_states.shape
        assert fine_token_states_size % self.block_size == 0
        num_fine_blocks = fine_token_states_size // self.block_size
        fine_token_states = fine_token_states * fine_token_mask[:, :, None]

        if self.coarser_type == "mean":
            fine_token_states = fine_token_states.reshape(batch_size, num_fine_blocks, self.block_size, dim)
            fine_token_mask = fine_token_mask.reshape(batch_size, num_fine_blocks, self.block_size)
            to_coarse_count = fine_token_mask.float().sum(dim = 2).to(fine_token_states.dtype)
            to_coarse_token_states = fine_token_states.float().sum(dim = 2)
            to_coarse_token_states = (to_coarse_token_states / (to_coarse_count[:, :, None] + 1e-4)).to(fine_token_states.dtype)
            to_coarse_token_mask = (to_coarse_count > 0).to(fine_token_states.dtype)
            difference = to_coarse_token_states[:, :, None, :] - fine_token_states
        else:
            raise Exception()

        return to_coarse_token_states, to_coarse_token_mask, difference

    def forward(self, mixed_states):
        # The coarser module accepts a mixed_states that contains
        # {mask, fine_token_states, fine_token_mask}
        # in first layer or
        # {mask, fine_block_indices, partial_fine_token_states, partial_fine_token_mask,
        # coarse_block_indices, partial_coarse_token_states, partial_coarse_token_mask, difference_cache, cache_indice_table}
        # in the remaining layers
        # outputs a mixed_states that contains
        # {mask} old
        # + {coarse_token_states, coarse_token_mask, difference_cache, cache_indice_table} new

        if "fine_token_states" in mixed_states and "fine_token_mask" in mixed_states:

            fine_token_states = mixed_states["fine_token_states"]
            fine_token_mask = mixed_states["fine_token_mask"]
            del mixed_states["fine_token_states"]

            to_coarse_token_states, to_coarse_token_mask, difference = self.to_coarse(fine_token_states, fine_token_mask)
            batch_size, num_blocks, _, _ = difference.shape
            cache_indice_table = torch.arange(num_blocks, device = difference.device)[None, :].repeat(batch_size, 1)

            mixed_states["coarse_token_states"] = to_coarse_token_states
            mixed_states["coarse_token_mask"] = to_coarse_token_mask
            mixed_states["difference_cache"] = {"version":self.cache_version}

            if "fine_token_positions" in mixed_states:
                fine_token_positions = mixed_states["fine_token_positions"]
                coarse_token_positions = torch.round(fine_token_positions.reshape(batch_size, num_blocks, self.block_size).float().mean(dim = -1)).long()
                mixed_states["coarse_token_positions"] = coarse_token_positions

            cache_update(mixed_states["difference_cache"], difference, cache_indice_table)

        elif "fine_block_indices" in mixed_states and "coarse_block_indices" in mixed_states:

            fine_block_indices = mixed_states["fine_block_indices"]
            partial_fine_token_states = mixed_states["partial_fine_token_states"]
            partial_fine_token_mask = mixed_states["partial_fine_token_mask"]

            coarse_block_indices = mixed_states["coarse_block_indices"]
            partial_coarse_token_states = mixed_states["partial_coarse_token_states"]
            partial_coarse_token_mask = mixed_states["partial_coarse_token_mask"]

            del mixed_states["fine_block_indices"]
            del mixed_states["partial_fine_token_states"]
            del mixed_states["partial_fine_token_mask"]
            del mixed_states["coarse_block_indices"]
            del mixed_states["partial_coarse_token_states"]
            del mixed_states["partial_coarse_token_mask"]

            batch_size = fine_block_indices.shape[0]

            block_indices = torch.cat([fine_block_indices, coarse_block_indices], dim = 1)
            reorder_block_indices = torch.argsort(block_indices, dim = 1)
            batch_indices = torch.arange(batch_size, device = block_indices.device)[:, None]

            to_coarse_token_states, to_coarse_token_mask, difference = self.to_coarse(partial_fine_token_states, partial_fine_token_mask)

            coarse_token_states = torch.cat([to_coarse_token_states, partial_coarse_token_states], dim = 1)
            coarse_token_mask = torch.cat([to_coarse_token_mask, partial_coarse_token_mask], dim = 1)

            coarse_token_states = coarse_token_states[batch_indices, reorder_block_indices, :]
            coarse_token_mask = coarse_token_mask[batch_indices, reorder_block_indices]

            mixed_states["coarse_token_states"] = coarse_token_states
            assert torch.allclose(mixed_states["coarse_token_mask"].half(), coarse_token_mask.half())

            cache_update(mixed_states["difference_cache"], difference, fine_block_indices)

        else:
            raise Exception()

        return mixed_states
