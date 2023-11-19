# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from src.args import import_from_string
from .postnorm import RobertaEmbeddings
from .prenorm import EncoderLayer
from .mra_projection.coarser import Coarser
from .mra_projection.finer import Finer
from .mra_projection.selector import Selector
from .mra_projection.formatter import Formatter

class WeightedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.scale_correction = config.scale_correction
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        assert config.hidden_size == self.all_head_size
        if config.attention_probs_dropout_prob != 0:
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        else:
            self.dropout = nn.Identity()

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def extra_repr(self):
        return f"scale_correction={self.scale_correction}"

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, state_scores, num_dups):

        hidden_states = self.LayerNorm(hidden_states)

        attention_probs = self.compute_attention_matrix(hidden_states, hidden_states, attention_mask, state_scores, num_dups)

        value_layer = self.transpose_for_scores(self.value(hidden_states)).float()
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.to(hidden_states.dtype)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        attention_output = self.dense(context_layer)
        return attention_output

    def compute_attention_matrix(self, query_states, key_states, attention_mask, state_scores = None, num_dups = None):

        query_layer = self.transpose_for_scores(self.query(query_states)).float()
        key_layer = self.transpose_for_scores(self.key(key_states)).float()

        scale = math.sqrt(math.sqrt(self.attention_head_size))
        query_layer = query_layer / scale
        key_layer = key_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        bias = -1000.0 * (1.0 - attention_mask[:, None, None, :].float())
        if state_scores is not None:
            bias = bias + torch.log(state_scores[:, None, None, :].float() + 1e-4)
        if num_dups is not None and self.scale_correction:
            bias = bias + torch.log(num_dups[:, None, None, :].float() + 1e-4)

        attention_scores = attention_scores + bias

        attention_probs = nn.functional.softmax(attention_scores, dim = -1)
        if state_scores is not None and num_dups is not None:
            attention_probs = self.dropout(attention_probs)

        return attention_probs

class VccEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = WeightedAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.checkpoint_attention = config.checkpoint_attention if hasattr(config, "checkpoint_attention") else False
        self.checkpoint_ffn = config.checkpoint_ffn if hasattr(config, "checkpoint_ffn") else False

        self.FFN = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.coarser = Coarser(config)
        self.finer = Finer(config)
        get_attention_matrix = lambda Q, K, Kmask:self.attention.compute_attention_matrix(self.attention.LayerNorm(Q), self.attention.LayerNorm(K), Kmask)
        self.selector = Selector(config, attention = get_attention_matrix)

    def encode(self, hidden_states, mask, state_scores, num_dups):
        if self.checkpoint_attention:
            attention_output = checkpoint(self.attention, hidden_states, mask, state_scores, num_dups, use_reentrant = False)
        else:
            attention_output = self.attention(hidden_states, mask, state_scores, num_dups)

        attention_output = self.dropout(attention_output)

        if self.checkpoint_ffn:
            ffn_output = checkpoint(self.FFN, attention_output + hidden_states, use_reentrant = False)
        else:
            ffn_output = self.FFN(attention_output + hidden_states)

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
        state_scores = torch.cat(state_scores, dim = 1)
        num_dups = torch.cat(num_dups, dim = 1)

        hidden_states = self.encode(hidden_states, mask, state_scores, num_dups)

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

class VccEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.short_segment_size = config.short_segment_size

        self.layer = nn.ModuleList([import_from_string(layer_type)(config) for layer_type in config.encoder_layer_types])

        self.formatter = Formatter(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

    def extra_repr(self):
        options = [
            f"short_segment_size={self.short_segment_size}",
        ]
        return ", ".join(options)

    def short_segment_forward(self, layer, mixed_states):
        assert isinstance(layer, EncoderLayer)

        hidden_states = mixed_states["hidden_states"]
        mask = mixed_states["mask"]
        batch_size, sequence_length, dim = hidden_states.shape

        assert sequence_length % self.short_segment_size == 0
        num_segments = sequence_length // self.short_segment_size

        hidden_states = hidden_states.reshape(batch_size * num_segments, self.short_segment_size, dim)
        mask = mask.reshape(batch_size * num_segments, self.short_segment_size)
        hidden_states = layer(hidden_states = hidden_states, attention_mask = mask)
        mixed_states["hidden_states"] = hidden_states.reshape(batch_size, sequence_length, dim)

        return mixed_states

    def forward(self, mixed_states):
        previous_layer_type = EncoderLayer
        for layer_idx, layer in enumerate(self.layer):
            if previous_layer_type == EncoderLayer and isinstance(layer, EncoderLayer):
                mixed_states = self.short_segment_forward(layer, mixed_states)
            elif previous_layer_type == VccEncoderLayer and isinstance(layer, EncoderLayer):
                mixed_states = self.short_segment_forward(layer, self.formatter.from_Vcc_input(mixed_states))
            elif previous_layer_type == EncoderLayer and isinstance(layer, VccEncoderLayer):
                mixed_states = layer(self.formatter.to_vcc_input(mixed_states))
            elif previous_layer_type == VccEncoderLayer and isinstance(layer, VccEncoderLayer):
                mixed_states = layer(mixed_states)
            else:
                raise Exception()
            previous_layer_type = type(layer)

        if previous_layer_type == VccEncoderLayer:
            mixed_states = self.formatter.from_vcc_input(mixed_states)

        mixed_states["hidden_states"] = self.LayerNorm(mixed_states["hidden_states"])
        return mixed_states

class VccRobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = VccEncoder(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, importance_mask, **kwargs):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        mixed_states = {
            "hidden_states":embedding_output,
            "mask":attention_mask,
            "importance_mask":importance_mask,
            "positions":position_ids,
        }
        mixed_states = self.encoder(mixed_states)
        sequence_output = mixed_states["hidden_states"]
        assert torch.allclose(mixed_states["mask"].half(), attention_mask.half())
        assert torch.allclose(mixed_states["importance_mask"].half(), importance_mask.half())
        assert torch.allclose(mixed_states["positions"].half(), position_ids.half())
        return sequence_output,

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
