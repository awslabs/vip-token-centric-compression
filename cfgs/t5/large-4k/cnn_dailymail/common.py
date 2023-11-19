# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from src.args import Args, Options

seed = 1234
output_dir = "../outputs"
save_top_k = 3
project = "vcc"


trainer = Args()
trainer.strategy = "ddp"
trainer.gradient_clip_val = 1.0
trainer.max_epochs = 10
trainer.gpus = 8
trainer.precision = "bf16"

data = Args()
data.pl_module = "src.t5.data.data_module_downstream.DownstreamDataModule"
data.num_workers = 8
data.training_dataset_path = "/nobackup/zhanpeng/cnn_dailymail/train"
data.validation_dataset_path = "/nobackup/zhanpeng/cnn_dailymail/validation"
data.tokenizer = "t5-base"
data.collator = "src.t5.data.downstream.summarization.CNNDailyMailCollator"
data.collator_args = Args()
data.collator_args.max_encoder_length = 4096
data.collator_args.max_decoder_length = 256

optimizer = Args()
optimizer.optimizer = "adam"
optimizer.batch_size = 4
optimizer.weight_decay = 0.01
optimizer.base_learning_rate = 3e-4
optimizer.min_lr_ratio = 0.001
optimizer.lr_scheduler_type = "linear"
optimizer.warmup_steps = 1000
