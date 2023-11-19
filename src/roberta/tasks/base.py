# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytorch_lightning as pl
import torch
from dataclasses import dataclass, field, asdict
import os
import json
import time
from transformers import AutoConfig, PretrainedConfig
from transformers.trainer_pt_utils import get_parameter_names
from src.utils import get_optimizer, get_scheduler, get_parameter_names
from src.base_model_module import BaseModelModule as Base

class BaseModelModule(Base):
    def __init__(self, config, data_module):
        super().__init__(config)
        if hasattr(self.config.model, "pretrained_config"):
            self.model_config = AutoConfig.from_pretrained(self.config.model.pretrained_config)
        else:
            self.model_config = PretrainedConfig()
        for key, val in self.config.model.to_dict().items():
            setattr(self.model_config, key, val)
        print(self.model_config)

    def step(self, batch):
        output = self.model(**batch)
        return output

    def on_save_checkpoint(self, checkpoint):
        save_to_hf = self.config.save_to_hf if hasattr(self.config, "save_to_hf") else True
        if save_to_hf:
            path = os.path.join(self.config.save_dir_path, "hf_ckpts", f"epoch={self.current_epoch:05d}-step={self.global_step:08d}")
            self.model.save_pretrained(path)

    def configure_optimizers(self):

        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        params_decay = [p for n, p in self.named_parameters() if any(nd in n for nd in decay_parameters)]
        params_nodecay = [p for n, p in self.named_parameters() if not any(nd in n for nd in decay_parameters)]

        optim_groups = [
            {"params": params_decay, "weight_decay": self.config.optimizer.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = get_optimizer(optim_groups = optim_groups, **self.config.optimizer.to_dict())

        max_steps = self.trainer.max_steps
        if max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
            print(f"Inferring max_steps: {max_steps}")

        scheduler = get_scheduler(
            self.config.optimizer.lr_scheduler_type,
            optimizer,
            num_warmup_steps = self.config.optimizer.warmup_steps,
            num_training_steps = max_steps,
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "loss",
                }
            ],
        )
