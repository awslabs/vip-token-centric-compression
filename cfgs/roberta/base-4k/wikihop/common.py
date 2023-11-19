# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from src.args import Args, Options

seed = 1234
output_dir = "../outputs"
save_top_k = 0
project = "vcc"


trainer = Args()
trainer.strategy = "ddp"
trainer.gradient_clip_val = 1.0
trainer.max_epochs = 10
trainer.gpus = 8
trainer.precision = 16

optimizer = Args()
optimizer.optimizer = "adam"
optimizer.batch_size = 4
optimizer.weight_decay = 0.01
optimizer.base_learning_rate = 5e-5
optimizer.min_lr_ratio = 0.001
optimizer.lr_scheduler_type = "linear"
optimizer.warmup_steps = 1000

model = Args()
model.pl_module = "src.roberta.tasks.multiple_choice.MultipleChoiceModelModule"

data = Args()
data.pl_module = "src.roberta.data.data_module_downstream.DownstreamDataModule"
data.num_workers = 8
data.training_dataset_path = "/nobackup/zhanpeng/wikihop/train"
data.validation_dataset_path = "/nobackup/zhanpeng/wikihop/dev"
data.tokenizer = "roberta-base"
data.collator = "src.roberta.data.downstream.wikihop.WikiHopCollator"
data.collator_args = Args()
data.collator_args.max_sequence_length = 4096
data.collator_args.encode_type = "original"
data.collator_args.max_num_candidates = 128
data.collator_args.shuffle_candidates = True
data.collator_args.shuffle_supports = True
data.collator_args.question_first = True
