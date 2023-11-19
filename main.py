# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import os
import pytorch_lightning as pl
from src.args import import_config, import_from_string
import argparse
import datetime
import logging
import copy
import sys
import json
import torch
import random
import time
import logging
from pytorch_lightning.loggers import CSVLogger, WandbLogger

def main(config):

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval = "step")]
    if config.save_top_k > 0:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_last = True,
                save_top_k = config.save_top_k,
                dirpath = config.save_dir_path,
                monitor = "step",
                mode = "max",
                filename = "{epoch:05d}-{step:08d}",
                save_on_train_epoch_end = False,
                every_n_epochs = 0 if config.save_top_k == 0 else None
            )
        )

    if config.test_run:
        trainer_logger = CSVLogger(os.path.join(config.save_dir_path, "logs"))
    else:
        trainer_logger = WandbLogger(project = config.project, name = config.run_name, save_dir = "..")

    trainer = pl.Trainer.from_argparse_args(
        config.trainer,
        callbacks = callbacks,
        enable_checkpointing = (config.save_top_k > 0),
        default_root_dir = config.save_dir_path if config.save_top_k > 0 else None,
        accelerator = 'gpu',
        logger = trainer_logger
    )

    if not os.path.exists(config.save_dir_path) and trainer.global_rank == 0:
        os.makedirs(config.save_dir_path)

    if trainer.global_rank == 0:
        print(config)

    if os.path.exists(os.path.join(config.save_dir_path, "last.ckpt")):
        config.seed = config.seed * random.randrange(10000)
        print(f"new seed: {config.seed}")
    pl.utilities.seed.seed_everything(config.seed)

    print(f"*********** initializing data module ***********")
    data = import_from_string(config.data.pl_module)(config)
    
    print(f"*********** initializing model module ***********")
    model = import_from_string(config.model.pl_module)(config, data_module = data)

    print(f"*********** seting up data module ***********")
    data.setup()
    
    if trainer.global_rank == 0:
        print(trainer)
        print(data)
        print(model)

    print(f"*********** finding potential resume checkpoints ***********")
    possible_ckpt_path = os.path.join(config.save_dir_path, "last.ckpt")
    if os.path.exists(possible_ckpt_path):
        print(f"Resuming from checkpoint to {possible_ckpt_path}")
    elif hasattr(config, "resume_ckpt_path"):
        print(f"Resuming from checkpoint to {config.resume_ckpt_path}")
        possible_ckpt_path = config.resume_ckpt_path
    else:
        possible_ckpt_path = None

    if config.val_only:
        print(f"*********** start validation ***********")
        trainer.validate(model = model, datamodule = data, ckpt_path = possible_ckpt_path)
    else:
        print(f"*********** start training ***********")
        trainer.fit(model = model, datamodule = data, ckpt_path = possible_ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    parser.add_argument('--test_run', action = 'store_true')
    parser.add_argument('--val_only', action = 'store_true')
    args = parser.parse_args()

    print(f"args: {args}")

    config = import_config(args.config)
    config.run_name = args.config.replace(os.sep, "-")
    if config.trainer.gpus == -1:
        config.trainer.gpus = torch.cuda.device_count()
    config.save_dir_path = os.path.join(config.output_dir, args.config)
    config.config_path = args.config

    if not hasattr(config, "test_run"):
        config.test_run = args.test_run
    if not hasattr(config, "val_only"):
        config.val_only = args.val_only

    if config.test_run:
        config.trainer.gpus = 1
        config.optimizer.batch_size = min(2, config.optimizer.batch_size)

    main(config)
