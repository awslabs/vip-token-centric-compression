# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from typing import Optional, Tuple, Union
import copy
from .layers import T5Block
from .vcc_layers import VccT5Block
from src.roberta.models.mra_projection import Formatter
from src.args import import_from_string

class VccT5Encoder(T5Stack):
    def __init__(self, config, embed_tokens = None):
        super().__init__(config, embed_tokens = embed_tokens)
        self.formatter = Formatter(config)
        relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, config.num_heads)
        self.block = nn.ModuleList()
        for i in range(config.num_layers):
            self.block.append(import_from_string(config.encoder_layers[i])(config, relative_attention_bias))

    def forward(self, input_ids, attention_mask, encoder_input_important_mask, **kwargs):

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        inputs_embeds = self.embed_tokens(input_ids)

        position_ids = torch.arange(seq_length, device = input_ids.device)[None, :].repeat(batch_size, 1)

        mixed_states = {
            "hidden_states":self.dropout(inputs_embeds),
            "mask":attention_mask,
            "importance_mask":encoder_input_important_mask,
            "positions":position_ids,
        }

        previous_layer_type = T5Block
        for layer_idx, layer in enumerate(self.block):
            if previous_layer_type == T5Block and isinstance(layer, T5Block):
                mixed_states = layer(mixed_states)
            elif previous_layer_type == VccT5Block and isinstance(layer, T5Block):
                mixed_states = layer(self.formatter.from_vcc_input(mixed_states))
            elif previous_layer_type == T5Block and isinstance(layer, VccT5Block):
                mixed_states = layer(self.formatter.to_vcc_input(mixed_states))
            elif previous_layer_type == VccT5Block and isinstance(layer, VccT5Block):
                mixed_states = layer(mixed_states)
            else:
                raise Exception()
            previous_layer_type = type(layer)

        if previous_layer_type == VccT5Block:
            mixed_states = self.formatter.from_vcc_input(mixed_states)

        hidden_states = self.final_layer_norm(mixed_states["hidden_states"])
        hidden_states = self.dropout(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = hidden_states)
