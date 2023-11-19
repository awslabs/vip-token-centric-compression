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

batch_size = 9000000
source_size = 32
vector_dim = 8

print("batch_size", batch_size)
print("source_size", source_size)
print("vector_dim", vector_dim)

print("################### vector_index_accumulate check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, )).cuda()
source = torch.randn(batch_size, vector_dim).cuda()

torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    outputs = vector_index_accumulate(indexes, source, source_size, custom = False)
torch.cuda.synchronize()
t1 = time.time()
torch_t = t1 - t0

torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    outputs = vector_index_accumulate(indexes, source, source_size, custom = True)
torch.cuda.synchronize()
t1 = time.time()
custom_t = t1 - t0

print(f"torch_t={torch_t:.5f}, custom_t={custom_t:.5f}")
