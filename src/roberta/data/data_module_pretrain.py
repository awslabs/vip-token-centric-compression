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
import random
from src.args import import_from_string
from transformers import AutoTokenizer

class IndexDataset(Dataset):
    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)

class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        assert hasattr(config.trainer, "use_distributed_sampler") and not config.trainer.use_distributed_sampler
        
        self.config = config.data
        self.seed = config.seed
        self.batch_size = config.optimizer.batch_size

        os.environ["TOKENIZERS_PARALLELISM"] = "False"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.data_collator = import_from_string(self.config.collator)(tokenizer = self.tokenizer, **self.config.collator_args.to_dict())

    def setup(self, stage = None):
        if stage is not None:
            print(f"Setting data module stage {stage} has no effect")

        print("Create memory map")
        training_mmap = pa.memory_map(self.config.training_dataset_path)
        validation_mmap = pa.memory_map(self.config.validation_dataset_path)
        print("MMAP Read ALL")

        self.training_dataset = pa.ipc.open_stream(training_mmap).read_all()
        self.validation_dataset = pa.ipc.open_stream(validation_mmap).read_all()

        assert len(self.training_dataset["text"].chunks) == 1
        assert len(self.validation_dataset["text"].chunks) == 1

    def get_indeces_subset(self, dataset_size):
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        rng = np.random.default_rng(self.seed + self.trainer.current_epoch + 1000 * rank)

        size_per_rank = dataset_size // world_size
        offset_start = rank * size_per_rank
        offset_end = (rank + 1) * size_per_rank
        indices = np.arange(offset_start, offset_end, dtype = np.uint32)

        rng.shuffle(indices)

        return indices

    def train_dataloader(self):

        indices = self.get_indeces_subset(len(self.training_dataset))
        data_collator = self.data_collator
        training_dataset = self.training_dataset

        dl = DataLoader(
            IndexDataset(indices),
            batch_size = self.batch_size,
            collate_fn = lambda indices:data_collator(training_dataset.take(indices)["text"].to_pylist()),
            num_workers = self.config.num_workers,
            drop_last = True,
            persistent_workers = True
        )

        rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        print(f"Initialized Train DataLoader, rank={rank}, world_size={world_size}, size={len(dl)}")

        return dl

    def val_dataloader(self):

        indices = self.get_indeces_subset(len(self.validation_dataset))
        data_collator = self.data_collator
        validation_dataset = self.validation_dataset

        dl = DataLoader(
            IndexDataset(indices),
            batch_size = self.batch_size,
            collate_fn = lambda indices:data_collator(validation_dataset.take(indices)["text"].to_pylist()),
            num_workers = self.config.num_workers,
            drop_last = True,
            persistent_workers = True
        )

        rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        print(f"Initialized Val DataLoader, rank={rank}, world_size={world_size}, size={len(dl)}")

        return dl
