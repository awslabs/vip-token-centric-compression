# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
from torch.optim import Adam, AdamW
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

def filter_inputs(func, inputs):
    func_args = inspect.signature(func).parameters.keys()
    filtered = {key:val for key, val in inputs.items() if key in func_args}
    return filtered

def get_optimizer(
    optimizer, optim_groups, base_learning_rate,
    adam_w_mode = True, adam_betas = (0.9, 0.98), adam_epsilon = 1e-6, **kwargs
):
    optimizer = optimizer.lower()
    optim_cls = {
        "adam": AdamW if adam_w_mode else Adam,
    }[optimizer]

    args = [optim_groups]
    kwargs = {
        "lr": base_learning_rate,
        "eps": adam_epsilon,
        "betas": adam_betas,
    }
    if optimizer in {"fusedadam", "fusedlamb"}:
        kwargs["adam_w_mode"] = adam_w_mode
    optimizer = optim_cls(*args, **kwargs)

    return optimizer