# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytorch_lightning as pl
import torch
import os
import json
import time
import datasets
from src.args import import_from_string
from .base import BaseModelModule
from src.utils import filter_inputs

class PretrainModelModule(BaseModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)

        self.tokenizer = data_module.tokenizer
        self.model = import_from_string(self.config.model.model_type)(self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        gradient_checkpointing = self.config.model.gradient_checkpointing if hasattr(self.config.model, "gradient_checkpointing") else False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if hasattr(self.config.model, "load_pretrain"):
            if hasattr(self.config, "resume_ckpt_path") and self.config.resume_ckpt_path is not None:
                return
            print(f"********* Loading pretrained weights: {self.config.model.load_pretrain}")
            checkpoint_model = import_from_string(self.config.model.load_pretrain["model_type"]).from_pretrained(self.config.model.load_pretrain["checkpoint"])
            checkpoint_model.resize_token_embeddings(len(self.tokenizer))
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_model.state_dict(), strict = False)
            print(f"missing_keys = {missing_keys}")
            print(f"unexpected_keys = {unexpected_keys}")

    def training_step(self, batch, batch_idx):
        filtered = filter_inputs(self.model.forward, batch)
        output = self.model(**filtered)
        output = {"loss":output.loss}
        for key, val in self.sync_dict(output).items():
            self.log(f"train.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def profile_encoder(self, batch, batch_idx):
        for module in self.model.modules():
            if hasattr(module, "encoder"):
                encoder = module.encoder
        filtered = filter_inputs(encoder.forward, batch)
        output = encoder(**filtered)
        output = {"loss":(output.last_hidden_state).abs().mean()}
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        filtered = filter_inputs(self.model.forward, batch)
        output = self.model(**filtered)
        output = {"loss":output.loss}
        for key, val in self.sync_dict(output).items():
            self.log(f"val.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output
