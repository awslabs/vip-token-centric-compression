# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from src.args import Args, Options, import_config
import os

__common = import_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), "common.py"))
seed = __common.seed
output_dir = __common.output_dir
save_top_k = __common.save_top_k
project = __common.project
trainer = __common.trainer
optimizer = __common.optimizer
data = __common.data

model = Args()
model.pl_module = "src.t5.tasks.vcc_t5_downstream.DownstreamModelModule"
model.model_type = "src.t5.models.t5_wrapper.VccT5ForConditionalGeneration"
model.pretrained_config = "t5-large"
# model.load_pretrain = {
#     "model_type":"src.t5.models.t5_wrapper.VccT5ForConditionalGeneration",
#     "checkpoint":"/mnt/zpzeng/hf_ckpts/t5/large-4k/mra-16n-custom-b256-250k-slr-resume.py/epoch=00002-step=00241700"
# }
model.max_decoder_length = data.collator_args.max_decoder_length

model.custom_relative_attention = True

model.short_segment_size = 512
model.block_size = 16
model.num_important_tokens = 6 * model.block_size
model.num_fine_blocks = 70
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
