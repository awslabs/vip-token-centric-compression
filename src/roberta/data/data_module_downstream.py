# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pyarrow as pa
import glob
import logging
import os, sys, json
import numpy as np
import pathlib
import torch
import datasets
from src.args import import_from_string
from transformers import AutoTokenizer

class DownstreamDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config.data
        self.seed = config.seed
        self.batch_size = config.optimizer.batch_size

        os.environ["TOKENIZERS_PARALLELISM"] = "False"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.data_collator = import_from_string(self.config.collator)(tokenizer = self.tokenizer, **self.config.collator_args.to_dict())

    def setup(self, stage = None):
        if stage is not None:
            print(f"Setting data module stage {stage} has no effect")

        self.training_dataset = datasets.load_from_disk(self.config.training_dataset_path)
        if isinstance(self.config.validation_dataset_path, list):
            self.validation_dataset = [
                datasets.load_from_disk(path)
                for path in self.config.validation_dataset_path
            ]
        else:
            self.validation_dataset = datasets.load_from_disk(self.config.validation_dataset_path)

    def train_dataloader(self):

        dl = DataLoader(
            self.training_dataset,
            batch_size = self.batch_size,
            collate_fn = self.data_collator,
            num_workers = self.config.num_workers,
            shuffle = True,
            drop_last = True,
            prefetch_factor = 4
        )

        return dl

    def val_dataloader(self):

        if isinstance(self.validation_dataset, list):
            dl = [
                DataLoader(
                    dataset,
                    batch_size = self.batch_size,
                    collate_fn = self.data_collator,
                    num_workers = self.config.num_workers,
                    drop_last = True,
                    prefetch_factor = 4
                )
                for dataset in self.validation_dataset
            ]
        else:
            dl = DataLoader(
                self.validation_dataset,
                batch_size = self.batch_size,
                collate_fn = self.data_collator,
                num_workers = self.config.num_workers,
                drop_last = True,
                prefetch_factor = 4
            )
        return dl
