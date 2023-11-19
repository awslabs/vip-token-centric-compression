# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5LayerNorm
from typing import Optional, Tuple, Union
from transformers import T5Tokenizer
import copy
from src.roberta.models.mra_projection import Coarser, Finer, Selector
from .layers import T5Attention, T5LayerFF

class VccT5Attention(T5Attention):
    def __init__(self, config, relative_attention_bias):
        super().__init__(config, relative_attention_bias)
        self.scale_correction = config.scale_correction

    def forward(self, hidden_states, positions, mask, state_scores, num_dups):
        attn_weights = self.compute_attention_matrix(hidden_states, positions, hidden_states, positions, mask, state_scores, num_dups)
        attn_output = self.unshape(torch.matmul(attn_weights, self.shape(self.v(hidden_states)).float())).to(hidden_states.dtype)
        attn_output = self.o(attn_output)
        return attn_output

    def compute_attention_matrix(self, query_states, query_positions, key_states, key_positions, key_mask, state_scores, num_dups):
        position_bias = self.compute_bias(query_positions, key_positions)
        query_states = self.shape(self.q(query_states)).float()
        key_states = self.shape(self.k(key_states)).float()
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        bias = - 1000.0 * (1.0 - key_mask[:, None, None, :].float())
        if state_scores is not None:
            bias = bias + torch.log(state_scores[:, None, None, :].float() + 1e-4)
        if num_dups is not None and self.scale_correction:
            bias = bias + torch.log(num_dups[:, None, None, :].float() + 1e-4)

        position_bias = position_bias.float() + bias

        scores = scores + position_bias
        attn_weights = F.softmax(scores, dim = -1)
        return attn_weights

class VccT5LayerSelfAttention(nn.Module):
    def __init__(self, config, relative_attention_bias):
        super().__init__()
        self.SelfAttention = VccT5Attention(config, relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps = config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, positions, attention_mask, state_scores, num_dups):
        normed_hidden_states = self.layer_norm(hidden_states)
        attn_output = self.SelfAttention(normed_hidden_states, positions, attention_mask, state_scores, num_dups)
        attn_output = self.dropout(attn_output)
        return attn_output

    def compute_attention_matrix(self, query_states, query_positions, key_states, key_positions, key_mask):
        query_states = self.layer_norm(query_states)
        key_states = self.layer_norm(key_states)
        attn_weights = self.SelfAttention.compute_attention_matrix(query_states, query_positions, key_states, key_positions, key_mask, None, None)
        return attn_weights

class VccT5Block(nn.Module):
    def __init__(self, config, relative_attention_bias):
        super().__init__()

        self.checkpoint_attention = config.checkpoint_attention if hasattr(config, "checkpoint_attention") else False
        self.checkpoint_ffn = config.checkpoint_ffn if hasattr(config, "checkpoint_ffn") else False

        self.layer = nn.ModuleList()
        self.layer.append(VccT5LayerSelfAttention(config, relative_attention_bias))
        self.layer.append(T5LayerFF(config))

        self.coarser = Coarser(config)
        self.finer = Finer(config)
        self.selector = Selector(config, attention = self.layer[0].compute_attention_matrix)

    def encode(self, hidden_states, positions, attention_mask, state_scores, num_dups):
        if self.checkpoint_attention:
            attention_output = checkpoint(self.layer[0], hidden_states, positions, attention_mask, state_scores, num_dups, use_reentrant = False)
        else:
            attention_output = self.layer[0](hidden_states, positions, attention_mask, state_scores, num_dups)

        if self.checkpoint_ffn:
            ffn_output = checkpoint(self.layer[1], attention_output + hidden_states, use_reentrant = False)
        else:
            ffn_output = self.layer[1](attention_output + hidden_states)

        hidden_states = (ffn_output + attention_output) * state_scores[:, :, None].to(hidden_states.dtype) + hidden_states
        return hidden_states

    def forward(self, mixed_states):
        mixed_states = self.finer(self.selector(self.coarser(mixed_states)))

        shape = mixed_states["important_token_mask"].shape
        dtype = mixed_states["partial_coarse_token_scores"].dtype
        device = mixed_states["partial_coarse_token_scores"].device

        hidden_states = [
            mixed_states["partial_fine_token_states"],
            mixed_states["partial_coarse_token_states"],
            mixed_states["important_token_states"]
        ]
        positions = [
            mixed_states["partial_fine_token_positions"],
            mixed_states["partial_coarse_token_positions"],
            mixed_states["important_token_positions"]
        ]
        mask = [
            mixed_states["partial_fine_token_mask"],
            mixed_states["partial_coarse_token_mask"],
            mixed_states["important_token_mask"]
        ]
        state_scores = [
            mixed_states["partial_fine_token_scores"],
            mixed_states["partial_coarse_token_scores"],
            torch.ones(shape, dtype = dtype, device = device)
        ]
        num_dups = [
            mixed_states["partial_fine_token_num_dups"],
            mixed_states["partial_coarse_token_num_dups"],
            torch.ones(shape, dtype = dtype, device = device)
        ]

        hidden_states = torch.cat(hidden_states, dim = 1)
        mask = torch.cat(mask, dim = 1)
        positions = torch.cat(positions, dim = 1)
        state_scores = torch.cat(state_scores, dim = 1)
        num_dups = torch.cat(num_dups, dim = 1)

        hidden_states = self.encode(hidden_states, positions, mask, state_scores, num_dups)

        breakpoint = mixed_states["partial_fine_token_states"].shape[1]
        partial_fine_token_states = hidden_states[:, :breakpoint, :]
        mixed_states["partial_fine_token_states"] = partial_fine_token_states

        next_breakpoint = breakpoint + mixed_states["partial_coarse_token_states"].shape[1]
        partial_coarse_token_states = hidden_states[:, breakpoint:next_breakpoint, :]
        mixed_states["partial_coarse_token_states"] = partial_coarse_token_states
        breakpoint = next_breakpoint

        important_token_states = hidden_states[:, breakpoint:, :]
        mixed_states["important_token_states"] = important_token_states

        return mixed_states
