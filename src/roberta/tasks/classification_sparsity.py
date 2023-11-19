# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import json
import time
from transformers import RobertaPreTrainedModel
from collections import defaultdict
from .downstream import DownstreamModelModule
from .metrics import Loss, Accuracy
from src.utils import filter_inputs
from src.args import import_from_string
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from src.base_model_module import get_optimizer

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForSequenceClassificaiton(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.roberta = import_from_string(config.encoder_type)(config)
        self.tunable_classifier = ClassificationHead(config)
        self.loss_fct = Loss()
        self.accu_fct = Accuracy()
        self.post_init()

    def forward(
        self,
        label,
        **kwargs
    ):
        outputs = self.roberta(**filter_inputs(self.roberta.forward, kwargs))
        sequence_output = outputs[0]

        scores = self.tunable_classifier(sequence_output[:, 0, :])
        loss, _ = self.loss_fct(scores, label)
        accu, count = self.accu_fct(scores, label)

        output = {
            "loss":loss, "accu":accu, "count":count
        }
        return output

class SparseSequenceClassificaitonModelModule(DownstreamModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)

        self.tokenizer = data_module.tokenizer
        self.model = RobertaForSequenceClassificaiton(self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if hasattr(self.config.model, "load_pretrain"):
            print(f"********* Loading pretrained weights: {self.config.model.load_pretrain}")
            checkpoint_model = import_from_string(self.config.model.load_pretrain["model_type"]).from_pretrained(self.config.model.load_pretrain["checkpoint"])
            checkpoint_model.resize_token_embeddings(len(self.tokenizer))
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_model.state_dict(), strict = False)
            print(f"missing_keys = {missing_keys}")
            print(f"unexpected_keys = {unexpected_keys}")

    
    def configure_optimizers(self):

        trained_parameters = []
        for name, param in self.model.named_parameters():
            if "tunable" in name:
                trained_parameters.append(param)
                print(name, param.shape)
                
        optim_groups = [
            {"params": trained_parameters, "weight_decay": 0.0},
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
