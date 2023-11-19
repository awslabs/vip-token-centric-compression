# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from src.args import Args

seed = 111111
output_dir = "../outputs"
save_top_k = 50
project = "vcc"

model = Args()
model.pl_module = "src.t5.tasks.pretrain.PretrainModelModule"
model.model_type = "src.t5.models.t5_wrapper.VccT5ForConditionalGeneration"
model.pretrained_config = "t5-large"
model.load_pretrain = {
    "model_type":"transformers.T5ForConditionalGeneration",
    "checkpoint":"t5-large"
}

model.custom_relative_attention = True

model.short_segment_size = 512
model.num_important_tokens = 224
model.block_size = 16
model.num_fine_blocks = 90
model.include_positions_in_mixed_states = True
model.selector_type = "attention_based_selector"
model.coarser_type = "mean"
model.explore_prob = 0.1
model.scale_correction = True
model.encoder_layers = [
        "src.t5.models.layers.T5Block"
    ] * 6 + [
        "src.t5.models.vcc_layers.VccT5Block"
    ] * 18

trainer = Args()
trainer.strategy = "ddp"
trainer.gradient_clip_val = 1.0
trainer.val_check_interval = 10000
trainer.limit_val_batches = 100
trainer.gpus = 8
trainer.precision = "bf16"
trainer.max_steps = 250000

data = Args()
data.pl_module = "src.t5.data.data_module_pretrain.PretrainDataModule"
data.num_workers = 8
data.training_dataset_path = "/nobackup/zhanpeng/wiki_en_1k/train.arrow"
data.validation_dataset_path = "/nobackup/zhanpeng/wiki_en_1k/val.arrow"
data.tokenizer = "t5-base"
data.collator = "src.t5.data.pretrain.span_corruption.T5DataCollatorForSpanCorruption"
data.collator_args = Args()
data.collator_args.noise_density = 0.148
data.collator_args.mean_noise_span_length = 3.0
data.collator_args.input_length = 4096
data.collator_args.decoder_start_token_id = 0

optimizer = Args()
optimizer.optimizer = "adam"
optimizer.adam_beta1 = 0.9
optimizer.adam_beta2 = 0.98
optimizer.adam_epsilon = 1e-6
optimizer.batch_size = 2
optimizer.weight_decay = 0.01
optimizer.base_learning_rate = 0.001
optimizer.min_lr_ratio = 0.001
optimizer.lr_scheduler_type = "linear"
optimizer.warmup_steps = 5000
