# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytorch_lightning as pl
import torch
from dataclasses import dataclass, field, asdict
import os
import json
import time
import torch.optim

class BaseModelModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.occupy_all_memory_flag = False
        if hasattr(self.config, "occupy_all_memory"):
            self.occupy_all_memory_flag = self.config.occupy_all_memory
        self.called_occupy_all_memory = False

    def try_occupy_all_memory(self):
        if not self.occupy_all_memory_flag or self.called_occupy_all_memory:
            return
        self.called_occupy_all_memory = True
        print("***************** Start: occupying all memory *****************")
        tmp_list = []
        while True:
            try:
                tmp_list.append(torch.ones(1024, 1024, 512, dtype = torch.float32, device = self.model.device))
            except Exception as e:
                print(e)
                break
        for tensor in tmp_list:
            del tensor
        del tmp_list
        print("***************** End:   occupying all memory *****************")

    def training_step(self, batch, batch_idx):
        self.try_occupy_all_memory()
        output = self.step(batch)
        for key, val in self.sync_dict(output).items():
            self.log(f"train.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        output = self.step(batch)
        for key, val in self.sync_dict(output).items():
            self.log(f"val.{dataloader_idx}.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def step(self, batch):
        raise Exception()

    def log(self, *args, **kwargs):
        if self.trainer is None:
            return
        else:
            return super().log(*args, **kwargs)

    def get_world_size(self):
        if self.trainer is None:
            return 1
        else:
            return self.trainer.world_size

    def sync_dict(self, inp):
        world_size = self.get_world_size()
        out = {key:val.detach() / world_size for key, val in inp.items()}

        if world_size == 1:
            return out

        for key in out:
            torch.distributed.all_reduce(out[key])
        return out