# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytorch_lightning as pl
import torch
import os
import json
import time
import datasets
import evaluate
from .pretrain import PretrainModelModule
from collections import defaultdict
from rouge_score import rouge_scorer
from src.utils import filter_inputs
from .downstream import DownstreamModelModule as DownstreamModelModule_

class DownstreamModelModule(DownstreamModelModule_):
    def generate(self, batch):
        generated_output = self.model.generate(
            batch["input_ids"],
            attention_mask = batch["attention_mask"],
            encoder_input_important_mask = batch["encoder_input_important_mask"],
            max_length = self.config.model.max_decoder_length
        )
        return generated_output
