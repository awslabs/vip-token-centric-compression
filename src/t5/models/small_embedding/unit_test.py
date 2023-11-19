# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from kernel import vector_index_accumulate
from kernel import autograd_vector_index_select

def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1), y.reshape(-1)], dim = 0))[0, 1]

batch_size = random.randrange(1000000, 20000000)
source_size = random.randrange(4, 32)
vector_dim = random.randrange(4, 64)
torch.manual_seed(2)

print("batch_size", batch_size)
print("source_size", source_size)
print("vector_dim", vector_dim)

print("################### vector_index_accumulate check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, )).cuda()
source = torch.randn(batch_size, vector_dim).cuda()

ref = vector_index_accumulate(indexes, source, source_size, custom = False)
out = vector_index_accumulate(indexes, source, source_size, custom = True)
print("max diff", (out - ref).abs().max(), corrcoef(out, ref))
print(ref.reshape(-1)[:20], out.reshape(-1)[:20])
assert corrcoef(out, ref) > 0.99999

print("################### autograd check ##################")
torch.manual_seed(0)

indexes = torch.randint(0, source_size, size = (batch_size, )).cuda()
source = torch.randn(source_size, vector_dim, requires_grad = True).cuda()
target = torch.randn(batch_size, vector_dim).cuda()

outputs = autograd_vector_index_select(indexes, source, custom = False)
loss = ((outputs - target) ** 2).mean()
source.retain_grad()
loss.backward()
ref_A, ref_B = outputs, source.grad

torch.manual_seed(0)

indexes = torch.randint(0, source_size, size = (batch_size, )).cuda()
source = torch.randn(source_size, vector_dim, requires_grad = True).cuda()
target = torch.randn(batch_size, vector_dim).cuda()

outputs = autograd_vector_index_select(indexes, source, custom = True)
loss = ((outputs - target) ** 2).mean()
source.retain_grad()
loss.backward()
out_A, out_B = outputs, source.grad

print("################### output check ##################")
print("max diff", (out_A - ref_A).abs().max(), "coef", corrcoef(out_A, ref_A))
print(out_A.reshape(-1)[:20], ref_A.reshape(-1)[:20])
print(out_A.reshape(-1)[:20] / ref_A.reshape(-1)[:20])
assert corrcoef(out_A, ref_A) > 0.99999

print("################### grad_source check ##################")
print("max diff", (out_B - ref_B).abs().max(), "coef", corrcoef(out_B, ref_B))
print(out_B.reshape(-1)[:20], ref_B.reshape(-1)[:20])
print(out_B.reshape(-1)[:20] / ref_B.reshape(-1)[:20])
assert corrcoef(out_B, ref_B) > 0.99999
