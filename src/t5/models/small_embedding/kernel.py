# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, file) for file in src_files]
small_embedding = load('small_embedding', src_files, verbose = True)

import small_embedding
from torch.autograd import Function

def vector_index_accumulate(indexes, source, output_size, custom = True):
    # indexes = [batch_size]
    # source = [batch_size, vector_dim]
    # outputs = [output_size, vector_dim]

    assert len(indexes.shape) == 1
    assert len(source.shape) == 2

    batch_size = indexes.shape[0]
    assert source.shape[0] == batch_size

    if custom:
        if not indexes.is_contiguous():
            indexes = indexes.contiguous()
        if not source.is_contiguous():
            source = source.contiguous()
        if indexes.dtype != torch.int:
            indexes = indexes.to(torch.int)
        if source.dtype != torch.float:
            source = source.float()

        return small_embedding.vector_index_accumulate(indexes, source, output_size)

    vector_dim = source.shape[1]
    outputs = torch.zeros(output_size, vector_dim, dtype = source.dtype, device = source.device)
    indexes = indexes[:, None].repeat(1, vector_dim)
    outputs.scatter_add_(0, indexes, source)

    return outputs

class VectorIndexSelect(Function):
    @staticmethod
    def forward(ctx, indexes, source):
        indexes = indexes.contiguous()
        ctx.save_for_backward(indexes)
        ctx.source_size = source.shape[0]
        return source[indexes, :]

    @staticmethod
    def backward(ctx, grad_outputs):
        indexes, = ctx.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        grad_source = vector_index_accumulate(indexes, grad_outputs, ctx.source_size, custom = True)
        return None, grad_source

def autograd_vector_index_select(indexes, source, custom):
    assert len(source.shape) == 2
    output_shape = list(indexes.shape) + [source.shape[-1]]
    indexes = indexes.reshape(-1)
    if custom:
        outputs = VectorIndexSelect.apply(indexes, source)
    else:
        outputs = source[indexes, :]
    outputs = outputs.reshape(output_shape)
    return outputs
