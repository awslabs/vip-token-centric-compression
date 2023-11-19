# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5DenseGatedActDense, T5DenseActDense
from typing import Optional, Tuple, Union
from transformers import T5Tokenizer
import copy
from .small_embedding.kernel import autograd_vector_index_select

class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        forwarded_states = self.dropout(forwarded_states)
        return forwarded_states

class T5Attention(nn.Module):
    def __init__(self, config, relative_attention_bias):
        super().__init__()
        self.relative_attention_bias = relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.custom_relative_attention = config.custom_relative_attention if hasattr(config, "custom_relative_attention") else False
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nn.Linear(self.d_model, self.inner_dim, bias = False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias = False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias = False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias = False)

    def _relative_position_bucket(self, relative_position):
        num_buckets = self.relative_attention_num_buckets // 2
        max_distance = self.relative_attention_max_distance
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, context_position, memory_position):
        context_position = context_position[:, :, None]
        memory_position = memory_position[:, None, :]
        relative_position = memory_position - context_position  # shape (batch_size, query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(relative_position)
        if self.custom_relative_attention:
            values = autograd_vector_index_select(relative_position_bucket, self.relative_attention_bias.weight, custom = True)
        else:
            values = self.relative_attention_bias(relative_position_bucket)  # shape (batch_size, query_length, key_length, num_heads)
        values = values.permute([0, 3, 1, 2])  # shape (batch_size, num_heads, query_length, key_length)
        return values

    def shape(self, states):
        return states.view(states.shape[0], -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(self, states):
        return states.transpose(1, 2).contiguous().view(states.shape[0], -1, self.inner_dim)

    def forward(self, hidden_states, positions, mask):
        attn_weights = self.compute_attention_matrix(hidden_states, positions, hidden_states, positions, mask)
        attn_output = self.unshape(torch.matmul(attn_weights, self.shape(self.v(hidden_states)).float())).to(hidden_states.dtype)
        attn_output = self.o(attn_output)
        return attn_output

    def compute_attention_matrix(self, query_states, query_positions, key_states, key_positions, key_mask):
        position_bias = self.compute_bias(query_positions, key_positions)
        query_states = self.shape(self.q(query_states)).float()
        key_states = self.shape(self.k(key_states)).float()
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        position_bias = position_bias.float() - 1000.0 * (1.0 - key_mask[:, None, None, :].float())

        scores = scores + position_bias
        attn_weights = F.softmax(scores, dim = -1)
        attn_weights = F.dropout(attn_weights, p = self.dropout, training = self.training)
        return attn_weights

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, relative_attention_bias):
        super().__init__()
        self.SelfAttention = T5Attention(config, relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps = config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, positions, attention_mask):
        normed_hidden_states = self.layer_norm(hidden_states)
        attn_output = self.SelfAttention(normed_hidden_states, positions, attention_mask)
        attn_output = self.dropout(attn_output)
        return attn_output

class T5Block(nn.Module):
    def __init__(self, config, relative_attention_bias):
        super().__init__()
        self.short_segment_size = config.short_segment_size

        self.checkpoint_attention = config.checkpoint_attention if hasattr(config, "checkpoint_attention") else False
        self.checkpoint_ffn = config.checkpoint_ffn if hasattr(config, "checkpoint_ffn") else False
        self.skip_standard_layers = config.skip_standard_layers if hasattr(config, "skip_standard_layers") else False

        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(self, mixed_states):
        if self.skip_standard_layers:
            return mixed_states

        hidden_states, positions, attention_mask = mixed_states["hidden_states"], mixed_states["positions"], mixed_states["mask"]

        batch_size, sequence_length, dim = hidden_states.shape
        assert sequence_length % self.short_segment_size == 0
        num_segments = sequence_length // self.short_segment_size

        hidden_states = hidden_states.reshape(batch_size * num_segments, self.short_segment_size, dim)
        positions = positions.reshape(batch_size * num_segments, self.short_segment_size)
        attention_mask = attention_mask.reshape(batch_size * num_segments, self.short_segment_size)

        if self.checkpoint_attention:
            hidden_states = checkpoint(self.layer[0], hidden_states, positions, attention_mask, use_reentrant = False) + hidden_states
        else:
            hidden_states = self.layer[0](hidden_states, positions, attention_mask) + hidden_states

        if self.checkpoint_ffn:
            hidden_states = checkpoint(self.layer[1], hidden_states, use_reentrant = False) + hidden_states
        else:
            hidden_states = self.layer[1](hidden_states) + hidden_states

        mixed_states["hidden_states"] = hidden_states.reshape(batch_size, sequence_length, dim)
        return mixed_states
