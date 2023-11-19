# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

import kernel
