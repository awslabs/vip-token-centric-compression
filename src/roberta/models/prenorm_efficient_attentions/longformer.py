# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention, LongformerConfig
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class LongformerAttention(LongformerSelfAttention):
    def __init__(self, config):

        self.attention_window_size = config.attention_window_size
        self.attention_num_global_tokens = config.attention_num_global_tokens
        longformer_config = LongformerConfig()
        longformer_config.num_attention_heads = config.num_attention_heads
        longformer_config.hidden_size = config.hidden_size
        longformer_config.attention_window = [config.attention_window_size]

        super().__init__(longformer_config, 0)

        self.query_global = self.query
        self.key_global = self.key
        self.value_global = self.value

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, X, attention_mask):
        attention_mask = (attention_mask - 1) * 10000
        if self.attention_num_global_tokens > 0:
            attention_mask[:, :self.attention_num_global_tokens] = 10000
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        context_layer = super().forward(
            hidden_states = self.LayerNorm(X),
            attention_mask = attention_mask,
            is_index_masked = is_index_masked,
            is_index_global_attn = is_index_global_attn,
            is_global_attn = is_global_attn,
        )[0]

        attention_output = self.dense(context_layer)
        return attention_output

    def extra_repr(self):
        return f'window_size={self.attention_window_size}, num_global_tokens={self.attention_num_global_tokens}'
